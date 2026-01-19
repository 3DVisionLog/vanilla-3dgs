import yaml
import os
import torch
import torch.nn.functional as F
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

from src.utils import set_seed
from src.data_loader import load_data
from src.gs.gs_model import GaussianModel
from src.gs.render import gaussians_to_screen, render
from src.gs.ssim import ssim
from src.camera import get_360_poses

def main(config_path, data_dir=None):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])

    save_dir = os.path.join("results", config["exp_name"])
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "config_backup.yaml"), "w") as f:
        yaml.dump(config, f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    base_dir = data_dir or config["base_dir"]
    datas = load_data(base_dir)
    H, W, focal = datas[0]["hwf"]
    H, W = int(H), int(W)

    model = GaussianModel(num_points=500).to(device) # 4000이 최대인듯

    # Optimizer (파라미터별 Learning Rate 차등 적용)
    optimizer = torch.optim.Adam([
        {'params': [model.xyz], 'lr': config["lr"]["xyz"]},
        {'params': [model.sh_coeffs], 'lr': config["lr"]["sh_coeffs"]},
        {'params': [model.opacity_logit], 'lr': config["lr"]["opacity_logit"]},
        {'params': [model.scale_log], 'lr': config["lr"]["scale_log"]},
        {'params': [model.rot_quat], 'lr': config["lr"]["rot_quat"]},
    ], lr=config["lr"]["default"])

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1000,
        gamma=0.5
    )

    cam_centers = []
    for data in datas:
        # c2w의 4번째 열이 카메라 위치
        cam_centers.append(data["c2w"][:3, 3]) 

    cam_centers = torch.stack(cam_centers) # (N_cams, 3)

    # 2. 카메라들이 분포한 구(Sphere)의 반지름을 구합니다.
    center = cam_centers.mean(dim=0)
    dist = (cam_centers - center).norm(dim=1)
    scene_radius = dist.max().item()

    # 3. 이걸 scene_extent로 고정합니다. (보통 1.0 ~ 4.0 정도 나옴)
    scene_extent = scene_radius * 1.1 
    print(f"🎯 고정된 Scene Extent: {scene_extent:.4f}")


    model.train()
    train_loss = []
    print("학습 시작!")
    for step in tqdm(range(config["n_iters"]), desc="Training..."): # 1000번 반복
        # debug_coordinate_system(model, datas, H, W, focal)
        optimizer.zero_grad()

        img_i = np.random.randint(0, len(datas))
        target = datas[img_i]["image"].to(device) # (H, W, 3)
        c2w = datas[img_i]["c2w"].to(device)      # (4, 4)
        # c2w[0:3, 1:3] *= -1
        w2c = torch.linalg.inv(c2w) # World -> Camera (역행렬)

        # World 좌표계 기준 방향 벡터 계산(점 위치 - 카메라 위치)
        # view_dirs = xyz_cam @ w2c[:3, :3].T 일케 했었는데... 느리대 이거
        cam_pos = -w2c[:3, :3].T @ w2c[:3, 3]  # Camera position (World 좌표계)
        view_dirs = model.xyz - cam_pos.to(device)  # (N, 3)
        view_dirs = view_dirs / view_dirs.norm(dim=1, keepdim=True)  # 정규화

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        """
        World space (xyz, Σ)
        ↓ w2c
        Camera space (x, y, z, Σ')
        ↓ proj
        Image plane (x/z, y/z)
        ↓ focal, cx, cy
        Pixel coords (u, v)
        ↓ rasterization + alpha blending
        Image
        """
        # xyz, cov3d, rgb, opacity = g3d
        # means2d, cov2d, rgb, opacity = gaussians2d
        g3d = model(view_dirs) 
        indices, g2d = gaussians_to_screen(g3d, w2c, focal, H, W)
        img = render(indices, g2d, H, W)
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        
        # (3) Loss 계산 (L1 + SSIM)
        # SSIM을 위해 차원 변경: (H, W, 3) -> (1, 3, H, W)
        img_permuted = img.permute(2, 0, 1).unsqueeze(0)
        target_permuted = target.permute(2, 0, 1).unsqueeze(0)

        l1_loss = (img - target).abs().mean()
        ssim_loss = 1.0 - ssim(img_permuted, target_permuted)
        total_loss = (1.0 - config["lambda"]) * l1_loss + config["lambda"] * ssim_loss

        # (4) Backward
        total_loss.backward()
        optimizer.step()
        scheduler.step()
    
        train_loss.append(total_loss.item())

        if step % 200 == 0 and 0.1*config["n_iters"] < step < 0.5*config["n_iters"]:
            # debug1(model, w2c, g2d, W, H)
            print(f"Step {step}: 점 개수 = {model.xyz.shape[0]}")
            # ranges = model.xyz.max(dim=0).values - model.xyz.min(dim=0).values # 축별 범위
            # scene_extent = ranges.max() # 가장 큰 축 기준

            new_gaussian = densify_and_prune(
                model, min_opacity=0.01, threshold_grad=0.0002, scene_extent=scene_extent
            )

            model.xyz = nn.Parameter(new_gaussian["xyz"])
            model.sh_coeffs = nn.Parameter(new_gaussian["sh_coeffs"])
            model.opacity_logit = nn.Parameter(new_gaussian["opacity_logit"])
            model.scale_log = nn.Parameter(new_gaussian["scale_log"])
            model.rot_quat = nn.Parameter(new_gaussian["rot_quat"])

            # 파라미터 텐서 자체가 교체되었으므로 Optimizer를 새로 만들어야 함
            param_list = [
                {'params': [model.xyz], 'lr': config["lr"]["xyz"], 'initial_lr': config["lr"]["xyz"], 'name': 'xyz'},
                {'params': [model.sh_coeffs], 'lr': config["lr"]["sh_coeffs"], 'initial_lr': config["lr"]["sh_coeffs"], 'name': 'sh'},
                {'params': [model.opacity_logit], 'lr': config["lr"]["opacity_logit"], 'initial_lr': config["lr"]["opacity_logit"], 'name': 'opacity'},
                {'params': [model.scale_log], 'lr': config["lr"]["scale_log"], 'initial_lr': config["lr"]["scale_log"], 'name': 'scale'},
                {'params': [model.rot_quat], 'lr': config["lr"]["rot_quat"], 'initial_lr': config["lr"]["rot_quat"], 'name': 'rotation'},
            ]
            optimizer = torch.optim.Adam(param_list, lr=config["lr"]["default"])
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=1000,
                gamma=0.5,
                last_epoch=step 
            )
            if 2*step == config["n_iters"]:
                print(f"✨ [Step {step}] Opacity Reset! 모든 점의 투명도를 1%로 초기화합니다.")
                target_opacity = 0.01 
                reset_logit = inverse_sigmoid(torch.tensor(target_opacity)).to(device)
                model.opacity_logit.data.fill_(reset_logit)
        if torch.isnan(model.xyz).any():
            print(f"\n🚨 [Step {step}] 업데이트 후 model.xyz에 nan 발생!")
            break
    model.eval()
    frames = []
    with torch.no_grad():
        poses = get_360_poses(device=device)
        for c2w in tqdm(poses, desc="[Render]"):
            c2w[0:3, 1:3] *= -1
            w2c = torch.linalg.inv(c2w)
            cam_pos = -w2c[:3, :3].T @ w2c[:3, 3]
            view_dirs = model.xyz - cam_pos.to(device) # unsqueeze(0)
            view_dirs = view_dirs / (view_dirs.norm(dim=1, keepdim=True))

            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            g3d = model(view_dirs) 
            indices, g2d = gaussians_to_screen(g3d, w2c, focal, H, W)
            img = render(indices, g2d, H, W)
            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

            rgb = torch.cat(img, dim=0) # 다시 이미지 모양으로 합치기
            rgb = rgb.reshape(H, W, 3)       # flatten해준거 다시 펴주고~
            rgb = (rgb.clamp(0, 1) * 255).byte().cpu().numpy()
    
            frames.append(Image.fromarray(rgb))

        # GIF 저장
        frames[0].save(
            os.path.join(save_dir,"result.gif"),
            save_all=True,
            append_images=frames[1:],
            optimize=False,
            duration=100,
            loop=0
        )
        print(f"\n✨ 저장 완료! {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--data", type=str, help="Optional data path")

    args = parser.parse_args()

    main(args.config, args.data)
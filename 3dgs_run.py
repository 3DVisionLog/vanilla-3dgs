import yaml
import os
import torch
import torch.nn as nn
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

from src.utils import set_seed
from src.data_loader import load_data
from src.gaussian.model import GaussianModel
from src.gaussian.densify import densify_and_prune
from src.gaussian.projection import gaussians_to_screen
# from src.render.render import render
from src.rasterizer.rasterizer import GaussianRasterizerFunction
from src.ssim import ssim
from src.camera import get_360_poses, get_cameras_extent
from src.utils import get_conic, get_radii

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
    print(f"데이터 로딩 완료 H: {H}, W: {W}, focal: {focal}")

    cameras_extent = get_cameras_extent(datas)
    print(f"Camera Extent 고정 : {cameras_extent:.4f}")

    model = GaussianModel(num_points=config["n_points"]).to(device)

    ply_path = os.path.join(base_dir, "points3D.ply") 
    if os.path.exists(ply_path):
        print(f"📂 [초기화] {ply_path} 파일을 로드하여 가우시안을 생성합니다.")
        # PLY 파일에 맞춰 모델 생성 (GaussianModel 클래스에 load_ply 메서드가 있다고 가정)
        model.load_ply(ply_path)

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

    model.train()
    train_loss = []
    print("학습 시작!")
    
    for step in tqdm(range(config["iters"]["total"]), desc="Training..."):
        optimizer.zero_grad()

        img_i = np.random.randint(0, len(datas))
        target = datas[img_i]["image"].to(device) # (H, W, 3)
        # c2w = datas[img_i]["c2w"].to(device)      # (4, 4)
        """
        datas[img_i]["c2w"]는 리스트에 저장된 텐서의 메모리 주소를 가리킴
        c2w[0:3, 1:3] *= -1하면 해당 인덱스(예: 7번 이미지)의 원본 데이터 자체가 -1이 곱해진 상태로 저장됨
        처음 7번 이미지가 뽑혔을 때는 -1이 곱해져서 정상적으로 +Z (OpenCV 좌표계)가 되는데....
        학습이 진행되다가 나중에 똑같은 7번 이미지가 다시 뽑히면 이미 -1이 곱해져 있던 값에 다시 -1을 곱하게 됨..!!!
        그래서 특정 시점(이미 뽑혔던 인덱스가 다시 뽑힐 때)마다 Z-min이 마이너스로 찍히고 경고 문구가 뜨는 것입니다.
        """
        c2w = datas[img_i]["c2w"].clone().to(device)
        # c2w[0:3, 1:3] *= -1
        w2c = torch.linalg.inv(c2w) # World -> Camera (역행렬)

        # World 좌표계 기준 방향 벡터 계산(점 위치 - 카메라 위치)
        # view_dirs = xyz_cam @ w2c[:3, :3].T 일케 했었는데... 느리대 이거
        cam_pos = -w2c[:3, :3].T @ w2c[:3, 3]  # Camera position (World 좌표계)
        view_dirs = model.xyz - cam_pos.to(device)  # (N, 3)
        view_dirs = view_dirs / view_dirs.norm(dim=1, keepdim=True)  # 정규화

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # g3d: xyz, cov3d, rgb, opacity
        # gaussians2d: means2d, cov2d, rgb, opacity
        g3d = model(view_dirs)
        indices, g2d = gaussians_to_screen(g3d, w2c, focal, H, W)

        uv      = g2d["uv"]
        cov2d   = g2d["cov2d"]
        conics  = get_conic(cov2d)
        rgb     = g2d["rgb"]
        opacity = g2d["opacity"]  
        radii = get_radii(cov2d)
        
        img = GaussianRasterizerFunction.apply(uv, conics, opacity, rgb, radii, H, W)
        img = img.clamp(0.0, 1.0)
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

        if config["iters"]["start"] < step <= 0.8*config["iters"]["total"]:
            if step % config["iters"]["densify"] == 0:
                new_gaussian = densify_and_prune(
                    model, min_opacity=0.01, threshold_grad=0.0002, scene_extent=cameras_extent
                )

                model.xyz = nn.Parameter(new_gaussian["xyz"])
                model.sh_coeffs = nn.Parameter(new_gaussian["sh_coeffs"])
                model.opacity_logit = nn.Parameter(new_gaussian["opacity_logit"])
                model.scale_log = nn.Parameter(new_gaussian["scale_log"])
                model.rot_quat = nn.Parameter(new_gaussian["rot_quat"])
                    
                print(f"Step {step}: 점 개수 = {model.xyz.shape[0]}")
                
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

                if step % config["iters"]["reset"] == 0:
                    new_opa = torch.tensor(0.01)
                    reset_logit = torch.log(new_opa / (1 - new_opa)).to(device)
                    model.opacity_logit.data.fill_(reset_logit)

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

            if config["renderer"] == "cuda":
                uv      = g2d["uv"]
                cov2d   = g2d["cov2d"]
                conics  = get_conic(cov2d)
                rgb     = g2d["rgb"]
                opacity = g2d["opacity"]  
                radii = get_radii(cov2d)
                
                img = GaussianRasterizerFunction.apply(uv, conics, opacity, rgb, radii, H, W)
            else: 
                img = render(indices, g2d, H, W)
                # img = torch.cat(img, dim=0) # 다시 이미지 모양으로 합치기
                # img = img.reshape(H, W, 3)  # flatten해준거 다시 펴주고~
            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            img = img.clamp(0.0, 1.0)
            rgb = (img * 255).byte().cpu().numpy()
    
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
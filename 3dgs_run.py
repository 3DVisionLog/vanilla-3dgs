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

    model.train()
    train_loss = []
    print("학습 시작!")
    for step in tqdm(range(3000), desc="Training..."): # 1000번 반복
        optimizer.zero_grad()

        img_i = np.random.randint(0, len(datas))
        target = datas[img_i]["image"].to(device) # (H, W, 3)
        c2w = datas[img_i]["c2w"].to(device)      # (4, 4)
        c2w[0:3, 1:3] *= -1 # Blender -> OpenCV 좌표계 변환 (Y, Z 뒤집기)

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

    model.eval()
    frames = []
    with torch.no_grad():
        poses = get_360_poses(device=device)
        for c2w in tqdm(poses, desc="[Render]"):
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
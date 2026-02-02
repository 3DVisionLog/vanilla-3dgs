import yaml
import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import glob

from src.utils import set_seed, save_ply, save_gif
from src.data.loader import load_data
from src.data.io import load_points3D_bin
from src.renderer import GaussianRenderer
from src.renderer import render
from src.ssim import ssim
from src.camera import get_360_poses, get_cameras_extent

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
    if glob.glob(os.path.join(base_dir, "*.json")):
        datas = load_data(base_dir, "nerf")
    else: datas = load_data(base_dir)
    H, W, focal = datas[0]["hwf"]
    H, W = int(H), int(W)
    print(f"데이터 로딩 완료 H: {H}, W: {W}, focal: {focal}")

    cameras_extent = get_cameras_extent(datas)
    print(f"Camera Extent 고정 : {cameras_extent:.4f}")

    points_path = os.path.join(base_dir, "sparse", "points3D.bin")
    if os.path.exists(points_path):
        xyz, rgb = load_points3D_bin(base_dir)
        num_points = xyz.shape[0]
    else:
        xyz, rgb = None, None
        num_points = config["n_points"]
        
    renderer = GaussianRenderer(num_points, xyz, rgb, config, device)
    print(f"모델 초기화! 점 개수: {num_points}")
    
    renderer.model.train()
    train_loss = []
    train_points = []
    for step in tqdm(range(config["iters"]["total"]), desc="Training..."):
        renderer.optimizer.zero_grad()

        img_i = np.random.randint(0, len(datas))
        target = datas[img_i]["image"].to(device) # (H, W, 3)
        w2c = datas[img_i]["w2c"].to(device)      # (4, 4)

        img = renderer(w2c, focal, H, W)
        
        # SSIM을 위해 차원 변경: (H, W, 3) -> (1, 3, H, W)
        img_permuted = img.permute(2, 0, 1).unsqueeze(0)
        target_permuted = target.permute(2, 0, 1).unsqueeze(0)

        l1_loss = (img - target).abs().mean()
        ssim_loss = 1.0 - ssim(img_permuted, target_permuted)
        total_loss = (1.0 - config["lambda"]) * l1_loss + config["lambda"] * ssim_loss

        total_loss.backward()
        renderer.optimizer.step()
        renderer.scheduler.step()

        if config["iters"]["start"] <= step <= 0.6 * config["iters"]["total"]:
            if step % config["iters"]["densify"] == 0:
                renderer.densify(cameras_extent)
            if step % config["iters"]["reset"] == 0:
                renderer.model.opacity_reset()

        train_loss.append(total_loss.item())
        train_points.append(renderer.model.xyz.shape[0])

    renderer.model.eval()
    frames = []
    with torch.no_grad():
        poses = get_360_poses(device=device)
        for w2c in tqdm(poses, desc="[Render]"):
            img = renderer(w2c, focal, H, W)

            frames.append(Image.fromarray(
                (img * 255).byte().cpu().numpy())
            )

    save_gif(frames, os.path.join(save_dir,"result.gif"))
    save_ply(renderer.model, os.path.join(save_dir, "point_cloud.ply"))
    print(f"\n결과 저장 완료! {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--data", type=str, help="Optional data path")

    args = parser.parse_args()

    main(args.config, args.data)
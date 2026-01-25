import torch

def gaussians_to_screen(g3d, w2c, focal, H, W):
    xyz     = g3d["xyz"]
    cov3d   = g3d["cov3d"]
    rgb     = g3d["rgb"]
    opacity = g3d["opacity"]
    device  = xyz.device

    # 1. world -> camera (t = W * p) 
    # 초점거리, z 같은거 다 카메라 기준인데 가우시안은 월드 좌표계에 있으니..
    w2c = w2c.to(device)
    xyz_homo = torch.cat([xyz, torch.ones_like(xyz[...,:1])], dim=-1)
    xyz_cam = (w2c @ xyz_homo.T).T # (N, 4)

    if xyz_cam[:, 2].min() < 0 and xyz_cam[:, 2].max() < 0:
        print("c2w[0:3, 1:3] *= -1 이거 확인 ㄱㄱ")
        print(f"DEBUG: Z-min: {xyz_cam[:, 2].min().item():.4f}, Z-max: {xyz_cam[:, 2].max().item():.4f}")
    valid_mask = xyz_cam[:, 2] > 0.2 # 0.2m 보다 뒤에 있는 놈들만 선별
    # if not valid_mask.any(): return torch.tensor([], device=device), {}

    indices_valid = torch.where(valid_mask)[0]
    curr_xyz_cam = xyz_cam[indices_valid]
    curr_cov3d = cov3d[indices_valid]
    curr_rgb = rgb[indices_valid]
    curr_opacity = opacity[indices_valid]
    
    curr_x, curr_y, curr_z = curr_xyz_cam[:, 0], curr_xyz_cam[:, 1], curr_xyz_cam[:, 2]

    # 2. cov3d → cov2d (Σ' = JWΣWᵀJᵀ)
    J = torch.stack([
        torch.stack([focal/curr_z, torch.zeros_like(curr_x), -(focal*curr_x)/(curr_z*curr_z)], dim=1),
        torch.stack([torch.zeros_like(curr_y), focal/curr_z, -(focal*curr_y)/(curr_z*curr_z)], dim=1)
    ], dim=1)
    
    W_mat = w2c[:3, :3]
    cov2d = (J @ W_mat) @ curr_cov3d @ (J @ W_mat).transpose(1, 2)
    
    # EWA 블러링 추가
    cov2d[:, 0, 0] += 0.3
    cov2d[:, 1, 1] += 0.3

    # 3. 점들도 2d로 투영
    u = (curr_x / curr_z) * focal + W / 2
    v = (curr_y / curr_z) * focal + H / 2
    uv = torch.stack([u, v], dim=1)

    # 4. sorting
    sort_indices = torch.argsort(curr_z, descending=False) # 깊은게 뒤쪽
    g2d = {
        "uv": uv[sort_indices],
        "cov2d": cov2d[sort_indices],
        "rgb": curr_rgb[sort_indices],
        "opacity": curr_opacity[sort_indices] 
    }
    
    return indices_valid[sort_indices], g2d
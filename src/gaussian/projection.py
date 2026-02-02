import torch

def _cull_gaussians(xyz_cam):
    # 귀찮으니 간단하게만 구현
    valid_mask = xyz_cam[:, 2] > 0.2 # 0.2m 보다 뒤에 있는 놈들만 선별
    indices_valid = torch.where(valid_mask)[0]
    
    return indices_valid

def _world_to_camera(xyz_world, w2c):
    """
    cullGaussian을 할껀데... 기준은 뭐다? 카메라 기준이어야 하잖아!
    초점거리.. z.. 같은 것도 다 카메라 기준이잖아! 카메라 시점으로 바꾸자 xyz
    """
    xyz_homo = torch.cat([xyz_world, torch.ones_like(xyz_world[...,:1])], dim=-1)
    xyz_cam = (w2c @ xyz_homo.T).T # (N, 4)

    return xyz_cam

def gaussians_to_screen(g3d, w2c, focal, H, W):
    # 1. world -> camera (t = W * p) 
    xyz_cam = _world_to_camera(g3d["xyz"], w2c)
    indices_valid = _cull_gaussians(xyz_cam)

    xyz_cam = xyz_cam[indices_valid]
    cov3d   = g3d["cov3d"][indices_valid]
    rgb     = g3d["rgb"][indices_valid]
    opacity = g3d["opacity"][indices_valid]
    
    x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]

    # 2. cov3d → cov2d (Σ' = JWΣWᵀJᵀ)
    J = torch.stack([
        torch.stack([focal/z, torch.zeros_like(x), -(focal*x)/(z*z)], dim=1),
        torch.stack([torch.zeros_like(y), focal/z, -(focal*y)/(z*z)], dim=1)
    ], dim=1)
    
    W_mat = w2c[:3, :3]
    cov2d = (J @ W_mat) @ cov3d @ (J @ W_mat).transpose(1, 2)
    
    # EWA 블러링 추가
    cov2d[:, 0, 0] += 0.3
    cov2d[:, 1, 1] += 0.3

    # 3. 점들도 2d로 투영
    u = (x / z) * focal + W / 2
    v = (y / z) * focal + H / 2
    uv = torch.stack([u, v], dim=1)

    # 4. sorting
    sort_indices = torch.argsort(z, descending=False) # 깊은게 뒤쪽
    g2d = {
        "uv": uv[sort_indices],
        "cov2d": cov2d[sort_indices],
        "rgb": rgb[sort_indices],
        "opacity": opacity[sort_indices] 
    }
    
    return indices_valid[sort_indices], g2d
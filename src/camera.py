import torch
import numpy as np
        
def get_look_at(eye_pos, center, up):
    """
    카메라가 eye_pos에서 center를 바라보게 하는 View Matrix 생성
    """
    z_axis = eye_pos - center
    z_axis = z_axis / torch.norm(z_axis)
    
    x_axis = torch.cross(up, z_axis, dim=0)
    x_axis = x_axis / torch.norm(x_axis)
    
    y_axis = torch.cross(z_axis, x_axis, dim=0)
    y_axis = y_axis / torch.norm(y_axis)
    
    # R (3x3)
    R = torch.stack([x_axis, y_axis, z_axis], dim=0)
    
    # t (3x1) = -R * eye
    t = -torch.matmul(R, eye_pos.unsqueeze(1)).squeeze()
    
    # 4x4 Matrix
    view_mat = torch.eye(4, device=eye_pos.device)
    view_mat[:3, :3] = R
    view_mat[:3, 3] = t
    
    return view_mat

def get_360_poses(n_frame=30, elevation=30, radius=4.0, device="cpu"):
    poses = []
    up = torch.tensor([0.0, 0.0, 1.0], device=device)
    center = torch.tensor([0.0, 0.0, 0.0], device=device)

    phi = np.deg2rad(elevation) # 고도

    for angle in np.linspace(0, 360, n_frame):
        theta = np.deg2rad(angle) # 방위각
        
        eye_pos = torch.tensor([
            radius * np.cos(phi) * np.cos(theta), 
            radius * np.cos(phi) * np.sin(theta),
            radius * np.sin(phi)
        ], device=device).float()

        c2w = get_look_at(eye_pos, center, up)

        poses.append(c2w)

    return poses
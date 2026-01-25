import torch
import numpy as np
        
def _get_c2w_opengl(eye, center, up) -> torch.Tensor:
    """
    eye: 카메라의 위치, center: 카메라가 바라보는 지점, up: 카메라 기준 up  
    -> c2w: 카메라가 보는걸 world 좌표계로
    """
    z_axis = -(center - eye) # 카메라 시선(center<-eye)의 반대 = 찍고 있는 방향이 -z
    x_axis = torch.cross(up, z_axis, dim=0) # 외적 순서 중요!!
    y_axis = torch.cross(z_axis, x_axis, dim=0)

    # 단위벡터로..
    z_axis = z_axis / torch.norm(z_axis)
    x_axis = x_axis / torch.norm(x_axis)
    y_axis = y_axis / torch.norm(y_axis)

    R = torch.stack([x_axis, y_axis, z_axis], dim=1)
    t = eye # c2w에서의 평행이동 == 걍 카메라 위치
    # => c2w는 연산이 쉬움 걍 [R|t] 잖아

    c2w = torch.eye(4, device=eye.device)
    c2w[:3, :3] = R
    c2w[:3, 3] = t

    return c2w

def get_360_poses(n_frames=30, elevation=30, radius=4.0, device="cpu"):
    poses = []
    up = torch.tensor([0.0, 0.0, 1.0], device=device)
    center = torch.tensor([0.0, 0.0, 0.0], device=device)

    phi = np.deg2rad(elevation) # 고도

    for angle in np.linspace(0, 360, n_frames):
        theta = np.deg2rad(angle) # 방위각
        
        eye_pos = torch.tensor([
            radius * np.cos(phi) * np.cos(theta), 
            radius * np.cos(phi) * np.sin(theta),
            radius * np.sin(phi)
        ], device=device).float()

        c2w = get_c2w_opengl(eye_pos, center, up)

        poses.append(c2w)

    return poses
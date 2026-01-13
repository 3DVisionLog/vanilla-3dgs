import torch
import numpy as np
        
def get_c2w_opengl(eye, center, up):
    """
    y   z      물체(center)에서 카메라(eye)를
    |  /       바라보는 pose matrix 생성!!
    | /        즉, 카메라가 월드 중 어디에 있나
    o ㅡㅡㅡ x (w2c는 view matrix 얜 world를 카메라로 담기)
    """
    # 카메라 시선(center<-eye)의 반대
    # 즉.. 카메라가 찍고 있는 방향을 -z로 하겠다
    z_axis = -(center - eye)
    z_axis = z_axis / torch.norm(z_axis)

    x_axis = torch.cross(z_axis, up, dim=0)
    x_axis = x_axis / torch.norm(x_axis)

    y_axis = torch.cross(x_axis, z_axis, dim=0)

    R = torch.stack([x_axis, y_axis, z_axis], dim=1)
    t = eye # c2w에서의 평행이동 == 걍 카메라 위치
    # => c2w는 연산이 쉬임 걍 [R|t] 잖아

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
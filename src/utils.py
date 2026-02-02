import random
import torch
import numpy as np
from plyfile import PlyData, PlyElement

def set_seed(seed: int = 42):
    random.seed(seed)            # 기본 Python random 고정
    np.random.seed(seed)         # NumPy 랜덤 고정
    torch.manual_seed(seed)      # CPU 연산 랜덤 고정
    torch.cuda.manual_seed(seed) # GPU 모든 디바이스 랜덤 고정
    torch.cuda.manual_seed_all(seed)  # 멀티 GPU일 때

    # 연산 재현성
    torch.backends.cudnn.deterministic = True  # cuDNN 연산을 determinisitc으로 강제
    torch.backends.cudnn.benchmark = False     # CUDA 성능 자동 튜닝 기능 끔 → 완전 재현 가능

def get_conic(cov2d): # 역행렬 계산
    a = cov2d[:, 0, 0]
    b = cov2d[:, 0, 1]
    c = cov2d[:, 1, 1]

    det = a * c - b**2 # (det = ac - b^2)

    inv_xx =  c / det
    inv_yy =  a / det
    inv_xy = -b / det

    return torch.stack([inv_xx, inv_yy, inv_xy], dim=1)

def get_radii(cov2d): # (N, 3)
    a = cov2d[:, 0, 0] # xx
    b = cov2d[:, 0, 1] # xy
    c = cov2d[:, 1, 1] # yy

    # λ = ((a+c) +- sqrt((a-c)² + 4b²)) / 2
    lambda1 = ((a+c) + torch.sqrt(
        ((a-c)**2+4*b**2))
    ) / 2.0
    
    # sqrt(고유값) -> 표준편차 -> 3배해서 99.7%가 반지름 속에 포함되도록..
    radii = torch.ceil(3.0 * torch.sqrt(lambda1)) # .ceil(): 올림

    return radii

def save_ply(model, path):
    xyz = model.xyz.detach().cpu().numpy()
    opacity = model.opacity_logit.detach().cpu().numpy()
    scale = model.scale_log.detach().cpu().numpy()
    rotation = torch.nn.functional.normalize(model.rot_quat).detach().cpu().numpy()
    
    """
    # 내 렌더러는 color = sigmoid(C0 * sh_dc)
    # viewer는 color = 0.5 + C0 * f_dc 라서 역변환 해줘야함... 일단

    # C0 = 0.28209479177387814

    # sh = model.sh_coeffs          # (N, 3, K)
    # N = sh.shape[0]

    # # 1. DC 성분 → 이미지 색 기준으로 변환
    # # PyTorch 렌더러와 동일한 색
    # rgb = torch.sigmoid(C0 * sh[:, :, 0])  # (N, 3) in [0,1]
    # rgb = rgb.detach().cpu().numpy()

    # # viewer 수식 역변환
    # f_dc = (rgb - 0.5) / C0                 # (N, 3)

    # # 2. SH 고차항 제거 (중요)
    # sh_degree = int(np.sqrt(sh.shape[2]) - 1)
    # sh_rest_dim = (sh_degree + 1) ** 2 - 1

    # f_rest = np.zeros((N, sh_rest_dim * 3), dtype=np.float32)
    """
    sh = model.sh_coeffs.detach().cpu().numpy()  # (N, 3, K)
    N, _, K = sh.shape

    # DC 그대로
    f_dc = sh[:, :, 0]            # (N, 3)

    # 고차항 그대로
    f_rest = sh[:, :, 1:].reshape(N, -1)  # (N, 3*(K-1))


    # PLY dtype
    dtype = [
        ('x','f4'),('y','f4'),('z','f4'),
        ('nx','f4'),('ny','f4'),('nz','f4'),
        ('f_dc_0','f4'),('f_dc_1','f4'),('f_dc_2','f4')
    ]

    for i in range(f_rest.shape[1]):
        dtype.append((f'f_rest_{i}','f4'))

    dtype += [
        ('opacity','f4'),
        ('scale_0','f4'),('scale_1','f4'),('scale_2','f4'),
        ('rot_0','f4'),('rot_1','f4'),('rot_2','f4'),('rot_3','f4')
    ]

    normals = np.zeros_like(xyz)

    data = np.concatenate([
        xyz, normals,
        f_dc, f_rest,
        opacity, scale,
        rotation
    ], axis=1)

    elements = np.empty(N, dtype=dtype)
    elements[:] = list(map(tuple, data))

    PlyData([PlyElement.describe(elements, 'vertex')]).write(path)

    print(f"✅ Viewer-compatible PLY saved: {path}")
import random
import torch
import numpy as np

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
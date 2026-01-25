import torch
from src.utils import get_conic

def render(indices, g2d, H, W):
    uv      = g2d["uv"]
    cov2d   = g2d["cov2d"]
    rgb     = g2d["rgb"]
    opacity = g2d["opacity"]

    """ 렌더링 과정 C = ΣTᵢαᵢcᵢ """
    device = uv.device

    # 픽셀 그리드 생성
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
    indexing='ij'
    )
    pixels = torch.stack([grid_x, grid_y], dim=-1).float() # (H, W, 2)
    conics = get_conic(cov2d)

    T = torch.ones(H, W, device=device) # Transmittance (빛이 통과하는 정도, 초기값 1)
    final_image = torch.zeros(H, W, 3, device=device)

    for i in range(len(indices)):
        # dᵀΣ⁻¹d = (p-μ)ᵀΣ⁻¹(p-μ) = ax²+cy²+2bxy
        delta = pixels - uv[i]  # (H, W, 2)
        mahal_dist = (
            delta[..., 0]**2 * conics[i][0] + # x=delta[..., 0]
            delta[..., 1]**2 * conics[i][1] + # y=delta[..., 1]
            2 * delta[..., 0] * delta[..., 1] * conics[i][2]
        )
        alpha = opacity[i] * torch.exp(-0.5 * mahal_dist) # opacity * exp(distance)
        # opacity는 학습, 뒤에 exp(distance) 이건... 걍 가우시안이니까 중앙에가 더 색이 짙을거 아녀

        weight = alpha.unsqueeze(-1) * T.unsqueeze(-1) # (H, W, 1)
        color = rgb[i].view(1, 1, 3)     # (1, 1, 3)
        # 이거... 매번 텐서 더하면 그래프가 엄청 길어질 수도 있대
        final_image += weight * color    # (H, W, 3)

        T = T * (1 - alpha) # 뒤를 위해 투과율 줄이기

        # if T.mean() < 0.001: break

    return final_image
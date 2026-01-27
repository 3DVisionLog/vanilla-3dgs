import math
from numba import cuda

@cuda.jit
def render_numba_kernel(means2d, conics, opacities, rgbs, output_image, H, W):
    # pixels = torch.stack([grid_x, grid_y], dim=-1).float()
    x, y = cuda.grid(2)
    if x >= W or y >= H: return

    T = 1.0  # Transmittance (빛이 통과하는 정도, 초기값 1)
    acc_r = 0.0
    acc_g = 0.0
    acc_b = 0.0

    num_points = means2d.shape[0]

    for i in range(num_points): 
        # dᵀΣ⁻¹d = (p-μ)ᵀΣ⁻¹(p-μ) = ax²+cy²+2bxy
        dx = x - means2d[i, 0] # 가우시안 중심과 내 픽셀(x, y) 거리
        dy = y - means2d[i, 1]
        inv_xx, inv_yy, inv_xy = conics[i]
        mahal_dist = (
            inv_xx * dx ** 2 +
            inv_yy * dy ** 2 +
            2.0 * inv_xy * dx * dy
        )

        alpha = opacities[i] * math.exp(-0.5 * mahal_dist) # opacity * exp(distance)
        # opacity는 학습, 뒤에 exp(distance) 이건... 걍 가우시안이니까 중앙에가 더 색이 짙을거 아녀

        # final_image += weight * color
        weight = T * alpha
        acc_r += weight * rgbs[i, 0]
        acc_g += weight * rgbs[i, 1]
        acc_b += weight * rgbs[i, 2]

        T = T * (1.0 - alpha) # 뒤를 위해 투과율 줄이기

    output_image[y, x, 0] = acc_r
    output_image[y, x, 1] = acc_g
    output_image[y, x, 2] = acc_b
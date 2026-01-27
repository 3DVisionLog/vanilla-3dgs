import math
import numpy as np
import torch
from numba import cuda

@cuda.jit
def render_numba_kernel(means2d, conics, opacities, rgbs, output_image):
    # 현재 스레드가 담당하는 픽셀 좌표 x, y = cuda.grid(2) 일케해도 되나봄
    # pixels = torch.stack([grid_x, grid_y], dim=-1).float()
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    
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

@cuda.jit
def render_backward_kernel(
    means2d, conics, opacities, rgbs, # forward랑 같은거
    final_image, # forward 결과물이 있어야 정확한 미분 가능하대
    grad_image, # 입력값, Loss에서 넘어온 미분값임 (dL_dImage)
    grad_means2d, grad_conics, grad_opacities, grad_rgbs # 출력값, 여기에 atomicAdd 함
):
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    T = 1.0
    acc_r = 0.0
    acc_g = 0.0
    acc_b = 0.0

    num_points = means2d.shape[0]

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # 2. 이 픽셀의 오차(Gradient) 가져오기
    # dL_dPixel: (3,) RGB 벡터
    dLoss_dr = grad_image[y, x, 0]
    dLoss_dg = grad_image[y, x, 1]
    dLoss_db = grad_image[y, x, 2]
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
    for i in range(num_points): 
        dx = x - means2d[i, 0]
        dy = y - means2d[i, 1]
        inv_xx, inv_yy, inv_xy = conics[i]
        mahal_dist = (
            inv_xx * dx ** 2 +
            inv_yy * dy ** 2 +
            2.0 * inv_xy * dx * dy
        )
        alpha = opacities[i] * math.exp(-0.5 * mahal_dist)

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    
        # Alpha 미분이 좀 복잡함 (최종 이미지로 예측을 하니까 그런거임)
        # final_image += alpha * T * rgb => ΣTᵢ*αᵢ*RGBᵢ
        # = Σʲ⁼ⁱ⁻¹TⱼαⱼRGBⱼ(1) + TᵢαᵢRGBᵢ(2) + Σⱼ₌ᵢ₊₁TⱼαⱼRGBⱼ(3)
        # αᵢ에 대해 미분한다고 하면... (1)은 상수라 0이고 (2)는 걍 TᵢRGBᵢ
        # (3)에서 Tⱼ가 문제임! Tᵢ=Πⁱ⁻¹(1-αⱼ)라서 i+1항 이후부턴 T에만 αᵢ가 관여함..
        # Tⱼ=(1-α₁)x(1-α₂)x...x(1-αᵢ)x...니까.. αᵢ로 미분하면.. 
        # 걍 원래 T에 -1/(1-αᵢ)해주면 되넹!!! -1/(1-αᵢ)Σⱼ₌ᵢ₊₁TⱼαⱼRGBⱼ
        # next = final - acc - curr
        next_r = final_image[y, x, 0] - acc_r - (alpha*T*rgbs[i, 0])
        next_g = final_image[y, x, 1] - acc_g - (alpha*T*rgbs[i, 1])
        next_b = final_image[y, x, 2] - acc_b - (alpha*T*rgbs[i, 2])

        d_alpha_r = (rgbs[i, 0] * T) - (next_r / (1.0 - alpha))
        d_alpha_g = (rgbs[i, 1] * T) - (next_g / (1.0 - alpha))
        d_alpha_b = (rgbs[i, 2] * T) - (next_b / (1.0 - alpha))

        # dLoss_dAlpha = grad_image*dImage_dAlpha
        dLoss_dAlpha = (
            dLoss_dr*d_alpha_r + 
            dLoss_dg*d_alpha_g + 
            dLoss_db*d_alpha_b
        )
        
        # mahal_dist: dᵀΣ⁻¹d = (p-μ)ᵀΣ⁻¹(p-μ) = ax²+cy²+2bxy
        dLoss_dDist = dLoss_dAlpha*-0.5*alpha # dLoss_dAlpha * dAlpha_dDist

        # 1. dLoss_dMean2d = dLoss_dDist * dDist_dMean2d
        dDist_du = 2*inv_xx*dx + 2*inv_xy*dy
        dDist_dv = 2*inv_yy*dy + 2*inv_xy*dx

        cuda.atomic.add(grad_means2d, (i, 0), -dLoss_dDist*dDist_du)
        cuda.atomic.add(grad_means2d, (i, 1), -dLoss_dDist*dDist_dv)

        # 2. dLoss_dConics = dLoss_dDist * dDist_dConic
        dDist_dxx = dx*dx # inv_xx(a)
        dDist_dyy = dy*dy # inv_yy(c)
        dDist_dxy = 2*dx*dy # inv_xy(b)
    
        cuda.atomic.add(grad_conics, (i, 0), dLoss_dDist*dDist_dxx)
        cuda.atomic.add(grad_conics, (i, 1), dLoss_dDist*dDist_dyy)
        cuda.atomic.add(grad_conics, (i, 2), dLoss_dDist*dDist_dxy)

        # 3. dLoss_dOpacities
        dAlpha_dOpac = math.exp(-0.5 * mahal_dist)
        cuda.atomic.add(grad_opacities, (i), dLoss_dAlpha*dAlpha_dOpac)

        # 4. dLoss_dRgbs = dLoss/dImage * alpha * T
        weight = alpha * T
        cuda.atomic.add(grad_rgbs, (i, 0), dLoss_dr*weight)
        cuda.atomic.add(grad_rgbs, (i, 1), dLoss_dg*weight)
        cuda.atomic.add(grad_rgbs, (i, 2), dLoss_db*weight)
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        weight = T * alpha
        acc_r += weight * rgbs[i, 0]
        acc_g += weight * rgbs[i, 1]
        acc_b += weight * rgbs[i, 2]

        T = T * (1.0 - alpha)

class GaussianRasterizerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means2d, conics, opacity, rgb, H, W):
        d_means2d = cuda.as_cuda_array(means2d.detach().contiguous())
        d_conics = cuda.as_cuda_array(conics.detach().contiguous())
        d_opacities = cuda.as_cuda_array(opacity.view(-1).detach().contiguous())
        d_rgbs = cuda.as_cuda_array(rgb.detach().contiguous())

        final_image = torch.zeros((H, W, 3), device=means2d.device)
        d_output = cuda.as_cuda_array(final_image)

        threadsperblock = (16, 16)
        blockspergrid_x = int(np.ceil(W / threadsperblock[0]))
        blockspergrid_y = int(np.ceil(H / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        render_numba_kernel[blockspergrid, threadsperblock](
            d_means2d, d_conics, d_opacities, d_rgbs, d_output, H, W
        )

        ctx.save_for_backward(means2d, conics, opacity, rgb, final_image)
        ctx.H = H
        ctx.W = W

        return final_image

    @staticmethod
    def backward(ctx, grad_img):
        # grad_img: Loss에서 넘어온 dLoss_dImage (H, W, 3)

        means2d, conics, opacity, rgb, final_image = ctx.saved_tensors
        H, W = ctx.H, ctx.W

        # Gradient 담을 빈 텐서
        grad_means2d = torch.zeros_like(means2d)
        grad_conics = torch.zeros_like(conics)
        grad_opacity = torch.zeros_like(opacity)
        grad_rgb = torch.zeros_like(rgb)

        d_grad_img = cuda.as_cuda_array(grad_img.contiguous())
        d_grad_means = cuda.as_cuda_array(grad_means2d)
        d_grad_conics = cuda.as_cuda_array(grad_conics)
        d_grad_opac = cuda.as_cuda_array(grad_opacity.view(-1).contiguous())
        d_grad_rgb = cuda.as_cuda_array(grad_rgb)

        # forward 했던 거 .detach() 또 해줘야함
        d_means = cuda.as_cuda_array(means2d.detach())
        d_conics = cuda.as_cuda_array(conics.detach())
        d_opac = cuda.as_cuda_array(opacity.view(-1).detach().contiguous())
        d_rgb = cuda.as_cuda_array(rgb.detach())
        d_final = cuda.as_cuda_array(final_image.detach())

        block = (16, 16)
        grid_x = int(np.ceil(W / 16))
        grid_y = int(np.ceil(H / 16))

        render_backward_kernel[(grid_x, grid_y), block](
            d_means, d_conics, d_opac, d_rgb, d_final,
            d_grad_img, # 입력
            d_grad_means, d_grad_conics, d_grad_opac, d_grad_rgb, # 출력
            H, W
        )

        # forward 인자랑 순서랑 개수 똑같아야 함(H, W는 미분 없으니 None 리턴)
        return grad_means2d, grad_conics, grad_opacity, grad_rgb, None, None
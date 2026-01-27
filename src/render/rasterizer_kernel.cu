#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Forward Kernel: 픽셀당 가우시안들을 루프 돌며 렌더링
__global__ void forward_kernel(
    const float* __restrict__ means2d,   // (N, 2)
    const float* __restrict__ conics,    // (N, 3)
    const float* __restrict__ opacities, // (N,)
    const float* __restrict__ rgbs,      // (N, 3)
    float* __restrict__ output_image,    // (H, W, 3)
    int num_points, int H, int W) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= W || y >= H) return;

    float T = 1.0f;  // Transmittance
    float acc_r = 0.0f;
    float acc_g = 0.0f;
    float acc_b = 0.0f;

    for (int i = 0; i < num_points; i++) {
        // dᵀΣ⁻¹d = (p-μ)ᵀΣ⁻¹(p-μ) = ax² + cy² + 2bxy
        float dx = (float)x - means2d[i * 2 + 0];
        float dy = (float)y - means2d[i * 2 + 1];
        
        /*
        얘가 좀 크리티컬하구나?
        # 거리가 너무 멀면(3시그마 밖) 계산하지 말고 패스 (continue)
        # 대략적인 범위 체크 (생략 가능하지만 넣으면 빠름)
        radius = radii[i]
        if abs(dx) > radius or abs(dy) > radius: continue
        */

        float inv_xx = conics[i * 3 + 0];
        float inv_yy = conics[i * 3 + 1];
        float inv_xy = conics[i * 3 + 2];

        float mahal_dist = (
            inv_xx * dx * dx + 
            inv_yy * dy * dy + 
            2.0f * inv_xy * dx * dy
        );
        
        // alpha = opacity * exp(distance)
        float alpha = opacities[i] * expf(-0.5f * mahal_dist);
        if (alpha < 1.0f / 255.0f): continue

        // final_image += weight * color
        float weight = T * alpha;
        acc_r += weight * rgbs[i * 3 + 0];
        acc_g += weight * rgbs[i * 3 + 1];
        acc_b += weight * rgbs[i * 3 + 2];

        T = T * (1.0f - alpha); // 뒤를 위해 투과율 줄이기
        if (T < 0.0001f) break;
    }

    output_image[(y * W + x) * 3 + 0] = acc_r;
    output_image[(y * W + x) * 3 + 1] = acc_g;
    output_image[(y * W + x) * 3 + 2] = acc_b;
}

// Backward Kernel: 오차 역전파
__global__ void backward_kernel(
    const float* __restrict__ means2d, const float* __restrict__ conics,
    const float* __restrict__ opacities, const float* __restrict__ rgbs,
    const float* __restrict__ final_image, 
    const float* __restrict__ grad_image,
    float* grad_means2d, float* grad_conics, float* grad_opacities, float* grad_rgbs,
    int num_points, int H, int W) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= W || y >= H) return;

    float T = 1.0f;
    float acc_r = 0.0f, acc_g = 0.0f, acc_b = 0.0f;

    // 해당 픽셀의 Gradient ------------------------
    int pix_idx = (y * W + x) * 3;
    float dLoss_dr = grad_image[pix_idx + 0];
    float dLoss_dg = grad_image[pix_idx + 1];
    float dLoss_db = grad_image[pix_idx + 2];
    // -------------------------------------------

    for (int i = 0; i < num_points; i++) {
        float dx = (float)x - means2d[i * 2 + 0];
        float dy = (float)y - means2d[i * 2 + 1];

        float inv_xx = conics[i * 3 + 0];
        float inv_yy = conics[i * 3 + 1];
        float inv_xy = conics[i * 3 + 2];

        float mahal_dist = (
            inv_xx * dx * dx + 
            inv_yy * dy * dy + 
            2.0f * inv_xy * dx * dy
        );
        
        float alpha = opacities[i] * expf(-0.5f * mahal_dist);
        if (alpha < 1.0f / 255.0f): continue
        
        // ---------------- backprop -----------------
        // dLoss/dAlpha 좀 복잡 (next = final - acc - curr)
        float final_r = final_image[pix_idx + 0];
        float final_g = final_image[pix_idx + 1];
        float final_b = final_image[pix_idx + 2];

        float curr_r = alpha * T * rgbs[i * 3 + 0];
        float curr_g = alpha * T * rgbs[i * 3 + 1];
        float curr_b = alpha * T * rgbs[i * 3 + 2];

        float next_r = final_r - acc_r - curr_r;
        float next_g = final_g - acc_g - curr_g;
        float next_b = final_b - acc_b - curr_b;

        float d_alpha_r = (rgbs[i * 3 + 0] * T) - (next_r / (1.0f - alpha + 1e-7f));
        float d_alpha_g = (rgbs[i * 3 + 1] * T) - (next_g / (1.0f - alpha + 1e-7f));
        float d_alpha_b = (rgbs[i * 3 + 2] * T) - (next_b / (1.0f - alpha + 1e-7f));

        // dLoss_dAlpha = grad_image*dImage_dAlpha
        float dLoss_dAlpha = (
            dLoss_dr * d_alpha_r + 
            dLoss_dg * d_alpha_g + 
            dLoss_db * d_alpha_b
        );

        // 1. grad_means2d: dLoss_dMean2d = dLoss_dDist * dDist_dMean2d
        // mahal_dist: dᵀΣ⁻¹d = (p-μ)ᵀΣ⁻¹(p-μ) = ax²+cy²+2bxy
        float dLoss_dDist = dLoss_dAlpha * -0.5f * alpha; // dLoss_dAlpha * dAlpha_dDist
        float dDist_du = 2.0f * inv_xx * dx + 2.0f * inv_xy * dy;
        float dDist_dv = 2.0f * inv_yy * dy + 2.0f * inv_xy * dx;
        atomicAdd(&grad_means2d[i * 2 + 0], -dLoss_dDist * dDist_du);
        atomicAdd(&grad_means2d[i * 2 + 1], -dLoss_dDist * dDist_dv);

        // 2. grad_conics: dLoss_dConics = dLoss_dDist * dDist_dConic
        atomicAdd(&grad_conics[i * 3 + 0], dLoss_dDist * dx * dx);
        atomicAdd(&grad_conics[i * 3 + 1], dLoss_dDist * dy * dy);
        atomicAdd(&grad_conics[i * 3 + 2], dLoss_dDist * 2.0f * dx * dy);

        // 3. grad_opacities: dLoss_dOpacities
        float dAlpha_dOpac = expf(-0.5f * mahal_dist);
        atomicAdd(&grad_opacities[i], dLoss_dAlpha * dAlpha_dOpac);

        // 4. grad_rgbs: dLoss_dRgbs = dLoss_dImage * alpha * T
        float weight = alpha * T;
        atomicAdd(&grad_rgbs[i * 3 + 0], dLoss_dr * weight);
        atomicAdd(&grad_rgbs[i * 3 + 1], dLoss_dg * weight);
        atomicAdd(&grad_rgbs[i * 3 + 2], dLoss_db * weight);
        // -------------------------------------------

        acc_r += curr_r;
        acc_g += curr_g;
        acc_b += curr_b;

        T = T * (1.0f - alpha);
        if (T < 0.0001f) break;
    }
}
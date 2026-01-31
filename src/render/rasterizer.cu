#include "rasterizer.h"
#include "config.h"

__global__ void _forwardKernel(
    const float* __restrict__ means2d,    // (N, 2)
    const float* __restrict__ conics,     // (N, 3)
    const float* __restrict__ opacities,  // (N,)
    const float* __restrict__ rgbs,       // (N, 3)
    const float* __restrict__ radii,      // (N,)
    const int2* __restrict__ tile_ranges, // (N, 2)
    float* __restrict__ output_image,     // (H, W, 3)
    int num_points, int H, int W) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // if (x >= W || y >= H) return;
    // 화면 밖 스레드라도 데이터를 로드하는 역할(협업)은 해야 하므로
    // return을 바로 하지 않고, 연산 부분에서만 체크합니다.
    bool inside = (x < W && y < H);

    float T = 1.0f;
    float acc_r = 0.0f, acc_g = 0.0f, acc_b = 0.0f;

    // ============================================================
    // 타일(블록) 내에서 나(스레드) 인덱스 (0~255) -> 협업 로딩에 쓰임
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // 현재 블록(타일)의 ID
    int tile_id = blockIdx.y * gridDim.x + blockIdx.x;

    // 이 타일이 처리해야 할 가우시안 범위 가져오기
    int2 range = tile_ranges[tile_id];
    int range_start = range.x;
    int range_end = range.y;

    // 처리할 가우시안이 없으면 조기 종료
    if (range_start >= range_end) return;

    // 공유 메모리 선언
    __shared__ float s_means[BLOCK_SIZE][2];
    __shared__ float s_conics[BLOCK_SIZE][3];
    __shared__ float s_opacities[BLOCK_SIZE];
    __shared__ float s_rgbs[BLOCK_SIZE][3];
    __shared__ float s_radii[BLOCK_SIZE];

    for (
        int batch_start = range_start; // 모든 점 ㄴㄴ
        batch_start < range_end;       // 특정 tile 안에 있는 가우시안만
        batch_start += BLOCK_SIZE // 가우시안은 256개 씩 끊어서 처리
    ) { 
        // 1. 협력적 로딩 (Collaborative Loading)
        // 각 스레드가 전역 메모리에서 가우시안 하나씩을 맡아서 공유 메모리로 가져옴
        int global_idx = batch_start + tid;
        
        if (global_idx < range_end) {
            s_means[tid][0] = means2d[global_idx * 2 + 0];
            s_means[tid][1] = means2d[global_idx * 2 + 1];
            
            s_conics[tid][0] = conics[global_idx * 3 + 0];
            s_conics[tid][1] = conics[global_idx * 3 + 1];
            s_conics[tid][2] = conics[global_idx * 3 + 2];
            
            s_opacities[tid] = opacities[global_idx];
            
            s_rgbs[tid][0] = rgbs[global_idx * 3 + 0];
            s_rgbs[tid][1] = rgbs[global_idx * 3 + 1];
            s_rgbs[tid][2] = rgbs[global_idx * 3 + 2];

            s_radii[tid] = radii[global_idx];
        }
        // ============================================================
        __syncthreads(); // 모든 스레드가 로딩을 마칠 때까지 대기 (동기화)

        if (inside && T > 0.0001f) {
            // 이번 배치에서 처리할 개수 (보통 256개, 마지막엔 남은 거)
            int batch_limit = min(BLOCK_SIZE, range_end - batch_start);

            for (int i = 0; i < batch_limit; i++) {
                // 공유 메모리에서 데이터 읽기 (전역 메모리 접근 X -> 매우 빠름)
                float radius = s_radii[i];
                float dx = (float)x - s_means[i][0];
                float dy = (float)y - s_means[i][1];

                if (fabsf(dx) > radius || fabsf(dy) > radius) continue;

                float inv_xx = s_conics[i][0];
                float inv_yy = s_conics[i][1];
                float inv_xy = s_conics[i][2];

                // alpha = opacity * exp(distance)
                float mahal_dist = ( // dᵀΣ⁻¹d = (p-μ)ᵀΣ⁻¹(p-μ) = ax² + cy² + 2bxy
                    inv_xx * dx * dx + 
                    inv_yy * dy * dy + 
                    2.0f * inv_xy * dx * dy
                );
                float alpha = s_opacities[i] * expf(-0.5f * mahal_dist);
                if (alpha < 1.0f / 255.0f) continue;

                // final_image += weight * color
                float weight = T * alpha;
                acc_r += weight * s_rgbs[i][0];
                acc_g += weight * s_rgbs[i][1];
                acc_b += weight * s_rgbs[i][2];

                T = T * (1.0f - alpha); // 뒤를 위해 투과율 줄이기
            }
        }
        __syncthreads();
    }

    // 결과 저장
    if (inside) {
        int pix_idx = (y * W + x) * 3;
        output_image[pix_idx + 0] = acc_r;
        output_image[pix_idx + 1] = acc_g;
        output_image[pix_idx + 2] = acc_b;
    }
}

__global__ void _backwardKernel(
    const float* __restrict__ means2d,   const float* __restrict__ conics,
    const float* __restrict__ opacities, const float* __restrict__ rgbs,
    const float* __restrict__ radii,     const int2* __restrict__ tile_ranges,
    const float* __restrict__ final_image,
    const float* __restrict__ grad_image,
    float* grad_means2d, float* grad_conics, float* grad_opacities, float* grad_rgbs,
    int num_points, int H, int W) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    bool inside = (x < W && y < H);
    float T = 1.0f;
    float acc_r = 0.0f, acc_g = 0.0f, acc_b = 0.0f;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int tile_id = blockIdx.y * gridDim.x + blockIdx.x;
    
    int2 range = tile_ranges[tile_id];
    int range_start = range.x;
    int range_end = range.y;

    if (range_start >= range_end) return;

    __shared__ float s_means[BLOCK_SIZE][2];
    __shared__ float s_conics[BLOCK_SIZE][3];
    __shared__ float s_opacities[BLOCK_SIZE];
    __shared__ float s_rgbs[BLOCK_SIZE][3];
    __shared__ float s_radii[BLOCK_SIZE];
    
    // 해당 픽셀의 Gradient ------------------------
    float dLoss_dr = 0.0f, dLoss_dg = 0.0f, dLoss_db = 0.0f;
    float final_r = 0.0f, final_g = 0.0f, final_b = 0.0f;

    if (inside) {
        int pix_idx = (y * W + x) * 3;
        dLoss_dr = grad_image[pix_idx + 0];
        dLoss_dg = grad_image[pix_idx + 1];
        dLoss_db = grad_image[pix_idx + 2];
        final_r = final_image[pix_idx + 0];
        final_g = final_image[pix_idx + 1];
        final_b = final_image[pix_idx + 2];
    }
    // -------------------------------------------

    for (int batch_start = range_start; batch_start < range_end; batch_start += BLOCK_SIZE) {        
        int global_idx = batch_start + tid;
        if (global_idx < num_points) {
            s_means[tid][0] = means2d[global_idx * 2 + 0];
            s_means[tid][1] = means2d[global_idx * 2 + 1];
            
            s_conics[tid][0] = conics[global_idx * 3 + 0];
            s_conics[tid][1] = conics[global_idx * 3 + 1];
            s_conics[tid][2] = conics[global_idx * 3 + 2];
            
            s_opacities[tid] = opacities[global_idx];
            
            s_rgbs[tid][0] = rgbs[global_idx * 3 + 0];
            s_rgbs[tid][1] = rgbs[global_idx * 3 + 1];
            s_rgbs[tid][2] = rgbs[global_idx * 3 + 2];

            s_radii[tid] = radii[global_idx];
        }
        __syncthreads();

        if (inside && T > 0.0001f) {
            int batch_limit = min(BLOCK_SIZE, range_end - batch_start);
            
            for (int i = 0; i < batch_limit; i++) {
                int curr_global_idx = batch_start + i; // 실제 가우시안 인덱스

                float radius = s_radii[i];
                float dx = (float)x - s_means[i][0];
                float dy = (float)y - s_means[i][1];

                if (fabsf(dx) > radius || fabsf(dy) > radius) continue;

                float inv_xx = s_conics[i][0];
                float inv_yy = s_conics[i][1];
                float inv_xy = s_conics[i][2];
                float mahal_dist = inv_xx * dx * dx + inv_yy * dy * dy + 2.0f * inv_xy * dx * dy;
                float alpha = s_opacities[i] * expf(-0.5f * mahal_dist);

                if (alpha < 1.0f / 255.0f) continue;

                // ---------------- backprop -----------------
                // dLoss/dAlpha 좀 복잡 (next = final - acc - curr)
                float curr_r = alpha * T * s_rgbs[i][0];
                float curr_g = alpha * T * s_rgbs[i][1];
                float curr_b = alpha * T * s_rgbs[i][2];

                float next_r = final_r - acc_r - curr_r;
                float next_g = final_g - acc_g - curr_g;
                float next_b = final_b - acc_b - curr_b;

                float denom = 1.0f - alpha + 1e-7f;
                float d_alpha_r = (s_rgbs[i][0] * T) - (next_r / denom);
                float d_alpha_g = (s_rgbs[i][1] * T) - (next_g / denom);
                float d_alpha_b = (s_rgbs[i][2] * T) - (next_b / denom);

                // dLoss_dAlpha = grad_image*dImage_dAlpha
                float dLoss_dAlpha = (
                    dLoss_dr * d_alpha_r + 
                    dLoss_dg * d_alpha_g + 
                    dLoss_db * d_alpha_b
                );

                // 1. grad_means2d: dLoss_dMean2d = dLoss_dDist * dDist_dMean2d
                // mahal_dist: dᵀΣ⁻¹d = (p-μ)ᵀΣ⁻¹(p-μ) = ax²+cy²+2bxy
                float dLoss_dDist = dLoss_dAlpha * -0.5f * alpha;
                float dDist_du = 2.0f * inv_xx * dx + 2.0f * inv_xy * dy;
                float dDist_dv = 2.0f * inv_yy * dy + 2.0f * inv_xy * dx;
                atomicAdd(&grad_means2d[curr_global_idx * 2 + 0], -dLoss_dDist * dDist_du);
                atomicAdd(&grad_means2d[curr_global_idx * 2 + 1], -dLoss_dDist * dDist_dv);

                // 2. grad_conics: dLoss_dConics = dLoss_dDist * dDist_dConic
                atomicAdd(&grad_conics[curr_global_idx * 3 + 0], dLoss_dDist * dx * dx);
                atomicAdd(&grad_conics[curr_global_idx * 3 + 1], dLoss_dDist * dy * dy);
                atomicAdd(&grad_conics[curr_global_idx * 3 + 2], dLoss_dDist * 2.0f * dx * dy);

                // 3. grad_opacities: dLoss_dOpacities
                float dAlpha_dOpac = expf(-0.5f * mahal_dist);
                atomicAdd(&grad_opacities[curr_global_idx], dLoss_dAlpha * dAlpha_dOpac);

                // 4. grad_rgbs: dLoss_dRgbs = dLoss_dImage * alpha * T
                float weight = alpha * T;
                atomicAdd(&grad_rgbs[curr_global_idx * 3 + 0], dLoss_dr * weight);
                atomicAdd(&grad_rgbs[curr_global_idx * 3 + 1], dLoss_dg * weight);
                atomicAdd(&grad_rgbs[curr_global_idx * 3 + 2], dLoss_db * weight);
                // -------------------------------------------

                acc_r += curr_r;
                acc_g += curr_g;
                acc_b += curr_b;

                T = T * (1.0f - alpha);
            }
        }
        __syncthreads();
    }
}

// C++ Wrapper Functions
torch::Tensor forwardCUDA(
    torch::Tensor means2d,
    torch::Tensor conics,
    torch::Tensor opacities,
    torch::Tensor rgbs,
    torch::Tensor radii,
    torch::Tensor tile_ranges,
    int H, int W
) {
    int num_points = means2d.size(0);
    auto output_image = torch::zeros({H, W, 3}, means2d.options());

    // 타일 크기 16x16 고정
    const dim3 threads(BLOCK_DIM, BLOCK_DIM);
    const dim3 blocks((W + BLOCK_DIM - 1) / BLOCK_DIM, (H + BLOCK_DIM - 1) / BLOCK_DIM);

    _forwardKernel<<<blocks, threads>>>(
        means2d.data_ptr<float>(),
        conics.data_ptr<float>(),
        opacities.data_ptr<float>(),
        rgbs.data_ptr<float>(),
        radii.data_ptr<float>(),
        (int2*)tile_ranges.data_ptr<int>(), // int2 (정수 2개짜리 구조체)
        output_image.data_ptr<float>(),
        num_points, H, W
    );
    return output_image;
}

std::vector<torch::Tensor> backwardCUDA(
    torch::Tensor means2d,
    torch::Tensor conics,
    torch::Tensor opacities,
    torch::Tensor rgbs,
    torch::Tensor radii,
    torch::Tensor tile_ranges,
    torch::Tensor final_image,
    torch::Tensor grad_image
) {
    int num_points = means2d.size(0);
    int H = grad_image.size(0);
    int W = grad_image.size(1);

    auto grad_means2d = torch::zeros_like(means2d);
    auto grad_conics = torch::zeros_like(conics);
    auto grad_opacities = torch::zeros_like(opacities);
    auto grad_rgbs = torch::zeros_like(rgbs);

    const dim3 threads(BLOCK_DIM, BLOCK_DIM);
    const dim3 blocks((W + BLOCK_DIM - 1) / BLOCK_DIM, (H + BLOCK_DIM - 1) / BLOCK_DIM);

    _backwardKernel<<<blocks, threads>>>(
        means2d.data_ptr<float>(),
        conics.data_ptr<float>(),
        opacities.data_ptr<float>(),
        rgbs.data_ptr<float>(),
        radii.data_ptr<float>(),
        (int2*)tile_ranges.data_ptr<int>(),
        final_image.data_ptr<float>(),
        grad_image.data_ptr<float>(),
        grad_means2d.data_ptr<float>(),
        grad_conics.data_ptr<float>(),
        grad_opacities.data_ptr<float>(),
        grad_rgbs.data_ptr<float>(),
        num_points, H, W
    );

    return {grad_means2d, grad_conics, grad_opacities, grad_rgbs};
}
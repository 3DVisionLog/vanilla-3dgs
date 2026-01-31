#include "rasterizer.h"
#include "config.h"

__global__ void _countTilesPerGaussian(
    const float* __restrict__ means2d, // (N, 2)
    const float* __restrict__ radii,   // (N, )
    int* __restrict__ tile_counts,     // 출력용임. 가우시안 별 타일 수
    int num_points, int grid_width, int grid_height
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    float px = means2d[idx * 2 + 0];
    float py = means2d[idx * 2 + 1];
    float r = radii[idx];

    // 타일 좌표계로 변환 ex) 150px / 16 -> tile 9
    int min_x = max(0, (int)((px - r) / BLOCK_DIM));
    int max_x = min(grid_width - 1, (int)((px + r) / BLOCK_DIM));
    int min_y = max(0, (int)((py - r) / BLOCK_DIM));
    int max_y = min(grid_height - 1, (int)((py + r) / BLOCK_DIM));

    if (min_x > max_x || min_y > max_y) {
        tile_counts[idx] = 0; // 화면 밖 가우시안이면 0개
    } else {
        // 가로 타일 개수 * 세로 타일 개수
        tile_counts[idx] = (max_x - min_x + 1) * (max_y - min_y + 1);
    }
}
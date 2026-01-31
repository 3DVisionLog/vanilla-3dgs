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

__global__ void _emitTilePairs(
    const float* __restrict__ means2d,
    const float* __restrict__ radii,
    const int* __restrict__ offsets,   // cumsum 결과
    long* __restrict__ duplicated_tile_ids,   // 출력용
    long* __restrict__ duplicated_point_ids,  // 출력용
    int num_points, int grid_width, int grid_height
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    float px = means2d[idx * 2 + 0];
    float py = means2d[idx * 2 + 1];
    float r = radii[idx];

    int min_x = max(0, (int)((px - r) / BLOCK_DIM));
    int max_x = min(grid_width - 1, (int)((px + r) / BLOCK_DIM));
    int min_y = max(0, (int)((py - r) / BLOCK_DIM));
    int max_y = min(grid_height - 1, (int)((py + r) / BLOCK_DIM));

    if (min_x > max_x || min_y > max_y) return;

    // offset: ex) [1, 2, 1] -> [1, 3, 4]
    int curr_offset = (idx == 0) ? 0 : offsets[idx - 1]; // 현재 점 시작위치
    
    int count = 0;
    // Bounding Box 내의 모든 타일에 대해 키 생성
    for (int y = min_y; y <= max_y; y++) {
        for (int x = min_x; x <= max_x; x++) {
            int tile_id = y * grid_width + x; // 2차원 계산하는거 알지?
            
            // 전역 메모리에 기록
            duplicated_tile_ids[curr_offset + count] = tile_id; // N번 타일
            duplicated_point_ids[curr_offset + count] = idx;    // 원본 점
            count++;
        }
    }
}
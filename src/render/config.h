#pragma once
#include <cuda.h>            // CUDA 드라이버 API
#include <cuda_runtime.h>    // CUDA 런타임 API (커널 실행, 메모리 관리 등)
#include <math.h>            // expf, fabsf 같은 수학 함수

#define BLOCK_DIM 16         // 타일 한 변의 길이 (16픽셀)
#define BLOCK_SIZE (BLOCK_DIM * BLOCK_DIM) // 16*16 = 256. 타일 하나를 담당할 스레드 개수
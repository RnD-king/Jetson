#pragma once
#include <cuda_runtime.h>

// RAW (1,5,N) => GPU 후보 압축 · 정렬(점수 내림차순) · Greedy NMS(완전 GPU)
// 선택된 결과만 Host로 복사한다.
bool RunYoloRawNMS(
    const float* d_raw,   // device ptr, shape = [1,5,N] in memory layout [5,N]
    int N,                // 8400 ...
    int input_w, int input_h,   // 모델 입력(HxW)
    int img_w,   int img_h,     // 원본 이미지 크기
    float conf_t, float iou_t,  // 0.70, 0.45
    int   max_det,              // 300
    cudaStream_t stream,
    // host outputs
    int*   h_num,               // 선택 개수
    float* h_boxes,             // size >= max_det*4 (xyxy, float)
    float* h_scores             // size >= max_det
);

// 1회 초기화/해제(워크스페이스). max_candidates = N(=pred 수), max_det = 300
bool InitRawNMSWorkspace(int max_candidates, int max_det);
void FreeRawNMSWorkspace();

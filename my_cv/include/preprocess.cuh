#ifndef MY_CV_CUDA_PREPROCESS_CUH_
#define MY_CV_CUDA_PREPROCESS_CUH_
#include <cuda_runtime.h>

// YOLO 입력 전처리용 커널 실행 함수 (HWC BGR → CHW RGB + 정규화 + padding 포함)
extern "C" void PreprocessKernelLauncher(
    const uint8_t* input_bgr,   // 원본 이미지 (HWC, BGR, 8-bit)
    int input_width,            // 원본 이미지 너비
    int input_height,           // 원본 이미지 높이
    float* output_chw,          // 출력 버퍼 (CHW, float32)
    int output_width,           // YOLO 입력 사이즈 너비
    int output_height,          // YOLO 입력 사이즈 높이
    int pad_x,                  // 좌우 패딩
    int pad_y,                  // 상하 패딩
    float scale,                // resize 비율
    cudaStream_t stream         // 실행 스트림
);

#endif  // MY_CV_CUDA_PREPROCESS_CUH_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

// 이미지 전처리: BGR→RGB, [0..1] 정규화, padding 114 적용, CHW 형식
__global__ void PreprocessKernel(
    const uint8_t* __restrict__ input_bgr, // 입력 BGR 이미지
    float* output_chw,                     // 출력 CHW 형식 (3채널)
    int input_w,                           // 입력 이미지 너비
    int input_h,                           // 입력 이미지 높이
    int resized_w,                         // 리사이즈된 너비
    int resized_h,                         // 리사이즈된 높이
    int pad_x,                             // x 방향 패딩
    int pad_y                              // y 방향 패딩
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= resized_w || y >= resized_h) return;

    // 원본 이미지에서 픽셀 비율 계산
    float scale_x = input_w / (float)resized_w;
    float scale_y = input_h / (float)resized_h;

    int src_x = min((int)(x * scale_x), input_w - 1);
    int src_y = min((int)(y * scale_y), input_h - 1);

    int input_idx = (src_y * input_w + src_x) * 3;
    uint8_t b = input_bgr[input_idx];
    uint8_t g = input_bgr[input_idx + 1];
    uint8_t r = input_bgr[input_idx + 2];

    // 패딩된 출력 이미지 인덱스 계산
    int dst_x = x + pad_x;
    int dst_y = y + pad_y;
    int out_idx = dst_y * (pad_x * 2 + resized_w) + dst_x;

    // CHW 형식으로 출력 (R, G, B 순서로 [0..1] 정규화)
    int area = (pad_x * 2 + resized_w) * (pad_y * 2 + resized_h);
    output_chw[out_idx] = r / 255.0f;                    // R
    output_chw[out_idx + area] = g / 255.0f;              // G
    output_chw[out_idx + area * 2] = b / 255.0f;          // B
}

// PreprocessLauncher 커널 호출 함수
extern "C" void PreprocessKernelLauncher(
    const uint8_t* input_bgr,      // 입력 이미지
    int input_w,                   // 원본 이미지 너비
    int input_h,                   // 원본 이미지 높이
    float* output_chw,             // 출력 CHW 데이터
    int resized_w,                 // 리사이즈된 이미지 너비
    int resized_h,                 // 리사이즈된 이미지 높이
    int pad_x,                     // x 방향 패딩
    int pad_y,                     // y 방향 패딩
    float scale,                   // 이미지 리사이즈 스케일
    cudaStream_t stream            // CUDA 스트림
) {
    dim3 threads(16, 16);
    dim3 blocks((resized_w + 15) / 16, (resized_h + 15) / 16);
    PreprocessKernel<<<blocks, threads, 0, stream>>>(
        input_bgr, output_chw,
        input_w, input_h,
        resized_w, resized_h,
        pad_x, pad_y
    );
}

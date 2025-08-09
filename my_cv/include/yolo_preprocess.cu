// yolo_preprocess.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>


__global__ void PreprocessKernel(
    const uint8_t* __restrict__ input_bgr,
    float* output_chw,
    int input_w,
    int input_h,
    int resized_w,
    int resized_h,
    int pad_x,
    int pad_y
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= resized_w || y >= resized_h) return;

    float scale_x = input_w / (float)resized_w;
    float scale_y = input_h / (float)resized_h;

    int src_x = min((int)(x * scale_x), input_w - 1);
    int src_y = min((int)(y * scale_y), input_h - 1);

    int input_idx = (src_y * input_w + src_x) * 3;
    uint8_t b = input_bgr[input_idx];
    uint8_t g = input_bgr[input_idx + 1];
    uint8_t r = input_bgr[input_idx + 2];

    int dst_x = x + pad_x;
    int dst_y = y + pad_y;
    int out_idx = dst_y * (pad_x * 2 + resized_w) + dst_x;

    // CHW
    int area = (pad_x * 2 + resized_w) * (pad_y * 2 + resized_h);
    output_chw[out_idx] = r / 255.0f;
    output_chw[out_idx + area] = g / 255.0f;
    output_chw[out_idx + area * 2] = b / 255.0f;
}

extern "C" void PreprocessKernelLauncher(
    const uint8_t* input_bgr,
    int input_w,
    int input_h,
    float* output_chw,
    int resized_w,
    int resized_h,
    int pad_x,
    int pad_y,
    float scale,
    cudaStream_t stream
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


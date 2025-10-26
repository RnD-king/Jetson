#include "yolo.hpp"          // my_cv::Detection 등 네가 가진 헤더
#include "preprocess.cuh"     // PreprocessKernelLauncher (BGR->RGB, [0..1], pad=114, CHW)

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>

#include <fstream>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>         // FP16 지원을 위한 헤더
#include <cmath>
#include <algorithm>
#include <map>
#include <vector>
#include <memory>

using namespace nvinfer1;
using namespace my_cv;

namespace {

// CUDA 에러 확인 함수
inline void checkCuda(cudaError_t e, const char* f, int l) {
    if (e != cudaSuccess) {
        std::cerr << "[CUDA] " << cudaGetErrorString(e) << " at " << f << ":" << l << std::endl;
        std::exit(1);
    }
}
#define CHECK_CUDA(x) checkCuda((x), __FILE__, __LINE__)

// float -> half 변환 커널
__global__ void FloatToHalfKernel(const float* __restrict__ src, __half* __restrict__ dst, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) dst[i] = __float2half_rn(src[i]);
}

// half -> float 변환 커널
__global__ void HalfToFloatKernel(const __half* __restrict__ src, float* __restrict__ dst, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

} // namespace

// ---------------- Logger ----------------
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};
static Logger gLogger;

// ---------------- YoloTRT ----------------
YoloTRT::YoloTRT(const std::string& engine_path, int input_w, int input_h)
: input_w_(input_w), input_h_(input_h) {

    if (!LoadEngine(engine_path)) {
        std::cerr << "Failed to load TensorRT engine!" << std::endl;
        std::exit(1);
    }

    out_floats_ = 1 + 6 * kMaxNumOutputBbox;  // NMS 포함시 6개 (x, y, w, h, confidence, class_id)
    out_bytes_  = out_floats_ * sizeof(float);

    // 출력 버퍼 초기화
    CHECK_CUDA(cudaMalloc(&buffers_[0], input_size_ * sizeof(float)));  // 입력
    CHECK_CUDA(cudaMalloc(&buffers_[1], out_bytes_));                    // 출력 (GPU)
    CHECK_CUDA(cudaMallocHost(&h_out_pinned_, out_bytes_));              // 출력 (Host pinned)

    CHECK_CUDA(cudaStreamCreate(&stream_));  // CUDA 스트림 생성
    CHECK_CUDA(cudaEventCreate(&ready_));
}

YoloTRT::~YoloTRT() {
    // 리소스 정리
    if (gpu_bgr_)           CHECK_CUDA(cudaFree(gpu_bgr_));
    if (gpu_float_input_)   CHECK_CUDA(cudaFree(gpu_float_input_));
    if (gpu_out_float_)     CHECK_CUDA(cudaFree(gpu_out_float_));

    if (buffers_[0])        CHECK_CUDA(cudaFree(buffers_[0])); // input device
    if (buffers_[1])        CHECK_CUDA(cudaFree(buffers_[1])); // output device

    if (h_out_pinned_)      CHECK_CUDA(cudaFreeHost(h_out_pinned_)); // host float

    if (ready_)             CHECK_CUDA(cudaEventDestroy(ready_));
    if (stream_)            CHECK_CUDA(cudaStreamDestroy(stream_));

    context_.reset();
    engine_.reset();
    runtime_.reset();
}

// --- 내부 유틸: I/O 텐서 이름/모드 자동 탐지
void YoloTRT::DiscoverIOTensors_() {
    input_name_.clear();
    output_name_.clear();

    const int n = engine_->getNbIOTensors();
    for (int i = 0; i < n; ++i) {
        const char* name = engine_->getIOTensorName(i);
        auto mode = engine_->getTensorIOMode(name);
        auto dt   = engine_->getTensorDataType(name);
        auto fmt  = engine_->getTensorFormat(name);
        std::cout << ((mode == TensorIOMode::kINPUT) ? "[IN ] " : "[OUT] ")
                  << name << " | dtype=" << (int)dt << " | fmt=" << (int)fmt << std::endl;

        if (mode == TensorIOMode::kINPUT)  input_name_  = name;
        else                               output_name_ = name;
    }
    if (input_name_.empty() || output_name_.empty()) {
        std::cerr << "[ERR] Failed to discover I/O tensors." << std::endl;
        std::exit(1);
    }
}

// --- 엔진 로드 & 버퍼 준비
bool YoloTRT::LoadEngine(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file) return false;
    const size_t size = (size_t)file.tellg();
    engine_blob_.resize(size);
    file.seekg(0, std::ios::beg);
    file.read(engine_blob_.data(), size);

    runtime_.reset(createInferRuntime(gLogger));
    if (!runtime_) return false;

    engine_.reset(runtime_->deserializeCudaEngine(engine_blob_.data(), engine_blob_.size()));
    if (!engine_) return false;

    context_.reset(engine_->createExecutionContext());
    if (!context_) return false;

    // I/O 이름 자동 탐지
    DiscoverIOTensors_();

    // 바인딩 dtype 확인
    auto in_dtype  = engine_->getTensorDataType(input_name_.c_str());
    auto out_dtype = engine_->getTensorDataType(output_name_.c_str());
    in_is_fp16_  = (in_dtype  == DataType::kHALF);
    out_is_fp16_ = (out_dtype == DataType::kHALF);

    // 동적 shape 대비: 입력 shape 설정 (NCHW)
    Dims in_shape; in_shape.nbDims = 4;
    in_shape.d[0] = 1; in_shape.d[1] = 3; in_shape.d[2] = input_h_; in_shape.d[3] = input_w_;
    if (!context_->setInputShape(input_name_.c_str(), in_shape)) {
        std::cerr << "[ERR] setInputShape failed" << std::endl;
        return false;
    }

    // 입력 바이트 계산 및 할당
    const size_t in_elems = (size_t)1 * 3 * input_h_ * input_w_;
    const size_t in_bytes_device = in_elems * (in_is_fp16_ ? sizeof(__half) : sizeof(float));
    CHECK_CUDA(cudaMalloc(&buffers_[0], in_bytes_device));
    input_elems_ = in_elems;

    // FP16 입력이면, float→half 변환용 임시 디바이스 버퍼(float)도 준비
    if (in_is_fp16_) {
        CHECK_CUDA(cudaMalloc(&gpu_float_input_, in_elems * sizeof(float)));
    }

    // 출력 shape 쿼리 (입력 shape 설정 후!)
    out_dims_ = context_->getTensorShape(output_name_.c_str());
    out_elems_ = 1;
    for (int i = 0; i < out_dims_.nbDims; ++i) out_elems_ *= out_dims_.d[i];

    const size_t out_bytes_device = out_elems_ * (out_is_fp16_ ? sizeof(__half) : sizeof(float));
    CHECK_CUDA(cudaMalloc(&buffers_[1], out_bytes_device));

    // 호스트 출력은 float로 통일
    CHECK_CUDA(cudaMallocHost(&h_out_pinned_, out_elems_ * sizeof(float)));

    // FP16 출력이면, half->float 변환용 디바이스 버퍼(float) 준비
    if (out_is_fp16_) {
        CHECK_CUDA(cudaMalloc(&gpu_out_float_, out_elems_ * sizeof(float)));
    }

    // 업로드용 BGR 버퍼
    gpu_bgr_ = nullptr;
    gpu_bgr_size_ = 0;

    std::cout << "[INFO] Engine loaded. input=" << input_name_ << " output=" << output_name_
              << " | in_fp16=" << in_is_fp16_ << " out_fp16=" << out_is_fp16_ << std::endl;
    return true;
}

// --- 전처리: BGR Mat → (RGB, [0..1], pad=114, letterbox, CHW)
void YoloTRT::Preprocess_(const cv::Mat& img_bgr) {
    const int src_w = img_bgr.cols;
    const int src_h = img_bgr.rows;

    // letterbox scale/offset 계산 (저장해서 역보정에 사용)
    scale_ = std::min(input_w_ / (float)src_w, input_h_ / (float)src_h);
    const int resized_w = int(src_w * scale_);
    const int resized_h = int(src_h * scale_);
    pad_x_ = (input_w_ - resized_w) / 2.0f;
    pad_y_ = (input_h_ - resized_h) / 2.0f;

    // 업로드용 BGR 디바이스 버퍼
    size_t required = (size_t)src_w * src_h * 3 * sizeof(uint8_t);
    if (gpu_bgr_ == nullptr || required > gpu_bgr_size_) {
        if (gpu_bgr_) CHECK_CUDA(cudaFree(gpu_bgr_));
        CHECK_CUDA(cudaMalloc(&gpu_bgr_, required));
        gpu_bgr_size_ = required;
    }
    CHECK_CUDA(cudaMemcpyAsync(gpu_bgr_, img_bgr.data, required, cudaMemcpyHostToDevice, stream_));

    // 커널 호출: BGR(Device) → RGB CHW float(Device), [0..1], pad=114, letterbox
    // 출력 위치:
    //  - 입력 바인딩이 FP32: buffers_[0] (float*)
    //  - 입력 바인딩이 FP16: gpu_float_input_ (float*), 이후 float->half 변환
    float* float_dst = in_is_fp16_ ? gpu_float_input_ : reinterpret_cast<float*>(buffers_[0]);

    PreprocessKernelLauncher(
        gpu_bgr_,                 // const uint8_t* BGR
        src_w, src_h,             // 원본 크기
        float_dst,                // float CHW destination (Device)
        resized_w, resized_h,     // 리사이즈 크기
        (int)pad_x_, (int)pad_y_, // 패딩
        scale_,                   // scale
        stream_                   // CUDA stream
    );

    // float → half 변환 (입력 바인딩이 FP16일 때만)
    if (in_is_fp16_) {
        const int n = (int)input_elems_;
        const int BS = 256;
        const int GS = (n + BS - 1) / BS;
        FloatToHalfKernel<<<GS, BS, 0, stream_>>>(gpu_float_input_, reinterpret_cast<__half*>(buffers_[0]), n);
        CHECK_CUDA(cudaGetLastError());
    }
}

// --- NMS 포함 여부 판별 (출력 shape로 판별)
bool YoloTRT::IsNMSOutput_() const {
    // 보편적으로 [1, N, 6]이면 NMS 포함(x,y,w,h,conf,cls)
    return (out_dims_.nbDims == 3 && out_dims_.d[2] == 6);
}

// --- 추론 & 후처리
bool YoloTRT::Infer(const cv::Mat& img_bgr, std::vector<Detection>& detections, int orig_w, int orig_h) {
    using Clock = std::chrono::high_resolution_clock;
    auto t1 = Clock::now();

    // 1) 전처리
    Preprocess_(img_bgr);
    auto t2 = Clock::now();

    // 2) 텐서 주소 지정 & enqueue
    context_->setTensorAddress(input_name_.c_str(),  buffers_[0]);
    context_->setTensorAddress(output_name_.c_str(), buffers_[1]);

    // (동적 shape 엔진이라면 안전하게 매 프레임 shape 보장)
    Dims in_shape; in_shape.nbDims = 4;
    in_shape.d[0]=1; in_shape.d[1]=3; in_shape.d[2]=input_h_; in_shape.d[3]=input_w_;
    context_->setInputShape(input_name_.c_str(), in_shape);

    bool ok = context_->enqueueV3(stream_);
    if (!ok) {
        std::cerr << "[ERR] enqueueV3 failed" << std::endl;
        return false;
    }
    auto t2_5 = Clock::now();

    // 3) 출력 복사 (FP16이면 GPU에서 half->float 변환 후 D2H, 아니면 바로 D2H)
    if (out_is_fp16_) {
        const int n = (int)out_elems_;
        const int BS = 256;
        const int GS = (n + BS - 1) / BS;
        HalfToFloatKernel<<<GS, BS, 0, stream_>>>(
            reinterpret_cast<const __half*>(buffers_[1]),
            gpu_out_float_,
            n
        );
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaMemcpyAsync(h_out_pinned_, gpu_out_float_, n * sizeof(float), cudaMemcpyDeviceToHost, stream_));
    } else {
        CHECK_CUDA(cudaMemcpyAsync(h_out_pinned_, buffers_[1], out_elems_ * sizeof(float), cudaMemcpyDeviceToHost, stream_));
    }
    CHECK_CUDA(cudaEventRecord(ready_, stream_));
    CHECK_CUDA(cudaEventSynchronize(ready_));
    auto t3 = Clock::now();

    // 4) 후처리
    ParseDetections_(reinterpret_cast<const float*>(h_out_pinned_), detections, orig_w, orig_h);
    auto t4 = Clock::now();

    // (선택) 타이밍 로그
    // std::cout << "[TRT ms] pre=" << std::chrono::duration<float,std::milli>(t2-t1).count()
    //           << " inf=" << std::chrono::duration<float,std::milli>(t2_5-t2).count()
    //           << " d2h=" << std::chrono::duration<float,std::milli>(t3-t2_5).count()
    //           << " post="<< std::chrono::duration<float,std::milli>(t4-t3).count() << std::endl;
    return true;
}

// --- 출력 파싱 (NMS 포함/미포함 자동 분기)
void YoloTRT::ParseDetections_(const float* out, std::vector<Detection>& detections, int img_w, int img_h) {
    detections.clear();
    std::vector<Detection> raw;

    const bool has_nms = IsNMSOutput_();

    // 역보정 스케일/패딩 (전처리에서 저장)
    const float sc = scale_;
    const float px = pad_x_;
    const float py = pad_y_;

    if (has_nms) {
        // [1, N, 6] : x y w h conf cls
        const int N = out_dims_.d[1];
        const int K = out_dims_.d[2]; // ==6
        for (int i = 0; i < N; ++i) {
            const float* p = out + i * K;
            float cx=p[0], cy=p[1], w=p[2], h=p[3], conf=p[4];
            int   cls = (int)p[5];
            if (conf < conf_thresh_) continue;

            float x = (cx - 0.5f*w - px)/sc;
            float y = (cy - 0.5f*h - py)/sc;
            float ww = w/sc, hh = h/sc;
            raw.push_back({ cv::Rect((int)x,(int)y,(int)ww,(int)hh), conf, cls });
        }
        // NMS가 이미 포함되어 있어도 살짝 겹침 제거용으로 한 번 더 NMS 하는 걸 선호하면 아래 유지
        ApplyNMS(raw, detections, iou_thresh_);
    } else {
        // [1, N, K] : (x,y,w,h, cls0, cls1, ...)
        const int N = out_dims_.d[1];
        const int K = out_dims_.d[2];
        const int cls_offset = 4; // v8 일반 출력은 보통 obj 없이 클래스 로짓만
        for (int i = 0; i < N; ++i) {
            const float* p = out + i * K;
            float cx=p[0], cy=p[1], w=p[2], h=p[3];

            int best_cls = 0; float best_prob = 0.f;
            for (int c = cls_offset; c < K; ++c) {
                if (p[c] > best_prob) { best_prob = p[c]; best_cls = c - cls_offset; }
            }
            float conf = best_prob;
            if (conf < conf_thresh_) continue;

            float x = (cx - 0.5f*w - px)/sc;
            float y = (cy - 0.5f*h - py)/sc;
            float ww = w/sc, hh = h/sc;
            raw.push_back({ cv::Rect((int)x,(int)y,(int)ww,(int)hh), conf, best_cls });
        }
        ApplyNMS(raw, detections, iou_thresh_);
    }
}

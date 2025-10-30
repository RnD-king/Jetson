#include "yolo.hpp"
#include "preprocess.cuh"

#include <opencv2/cudaimgproc.hpp>
#include <iostream>
#include <fstream>
#include <chrono>

using namespace nvinfer1;
using namespace my_cv;

namespace {
inline void checkCuda(cudaError_t e, const char* f, int l) {
    if (e != cudaSuccess) {
        std::cerr << "[CUDA] " << cudaGetErrorString(e) << " at " << f << ":" << l << std::endl;
        std::exit(1);
    }
}
#define CHECK_CUDA(x) checkCuda((x), __FILE__, __LINE__)
}

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // 너무 시끄러우면 INFO도 걸러도 됨
        if (severity != Severity::kINFO)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
};
static Logger gLogger;

// ---------------------------
// Constructor / Destructor
// ---------------------------
YoloTRT::YoloTRT(const std::string& engine_path, int input_w, int input_h)
: input_w_(input_w), input_h_(input_h) {
    if (!LoadEngine(engine_path)) {
        std::cerr << "Failed to load TensorRT engine!" << std::endl;
        std::exit(1);
    }
    CHECK_CUDA(cudaStreamCreate(&stream_));
    CHECK_CUDA(cudaEventCreate(&ready_));
}

YoloTRT::~YoloTRT() {
    if (gpu_bgr_)    CHECK_CUDA(cudaFree(gpu_bgr_));
    if (d_input_)    CHECK_CUDA(cudaFree(d_input_));
    if (d_out_)      CHECK_CUDA(cudaFree(d_out_));
    if (h_out_)      CHECK_CUDA(cudaFreeHost(h_out_));
    if (ready_)      CHECK_CUDA(cudaEventDestroy(ready_));
    if (stream_)     CHECK_CUDA(cudaStreamDestroy(stream_));
    context_.reset();
    engine_.reset();
    runtime_.reset();
}

// ---------------------------
// Engine Load
// ---------------------------
bool YoloTRT::LoadEngine(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "[ERR] Cannot open engine: " << engine_path << std::endl;
        return false;
    }
    const size_t size = static_cast<size_t>(file.tellg());
    std::vector<char> engine_blob(size);
    file.seekg(0, std::ios::beg);
    file.read(engine_blob.data(), size);

    runtime_.reset(createInferRuntime(gLogger));
    if (!runtime_) {
        std::cerr << "[ERR] createInferRuntime failed\n";
        return false;
    }

    engine_.reset(runtime_->deserializeCudaEngine(engine_blob.data(), size));
    if (!engine_) {
        std::cerr << "[ERR] deserializeCudaEngine failed\n";
        return false;
    }

    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        std::cerr << "[ERR] createExecutionContext failed\n";
        return false;
    }

    // I/O 이름 자동 추출 (입력 1개: "images", 출력 1개: "output0")
    const int n_io = engine_->getNbIOTensors();
    for (int i = 0; i < n_io; ++i) {
        const char* name = engine_->getIOTensorName(i);
        auto mode = engine_->getTensorIOMode(name);
        if (mode == TensorIOMode::kINPUT)  in_name_  = name;
        if (mode == TensorIOMode::kOUTPUT) out_name_ = name;
    }
    if (in_name_.empty() || out_name_.empty()) {
        std::cerr << "[ERR] Failed to discover I/O tensor names\n";
        return false;
    }

    // 입력/출력 shape
    auto in_dims  = engine_->getTensorShape(in_name_.c_str());   // (1,3,640,640) expected
    auto out_dims = engine_->getTensorShape(out_name_.c_str());  // (1,300,6) expected

    // 정적 vs 동적 입력
    bool is_dynamic = false;
    for (int d = 0; d < in_dims.nbDims; ++d) if (in_dims.d[d] == -1) is_dynamic = true;
    if (is_dynamic) {
        // 동적이면 런타임에 입력 셰이프 지정
        nvinfer1::Dims4 fix{1, 3, input_h_, input_w_};
        if (!context_->setInputShape(in_name_.c_str(), fix)) {
            std::cerr << "[ERR] setInputShape failed\n";
            return false;
        }
        in_dims = fix;
    }

    // 입력/출력 버퍼 할당
    size_t in_elems = 1;
    for (int d = 0; d < in_dims.nbDims; ++d) in_elems *= static_cast<size_t>(in_dims.d[d]);
    CHECK_CUDA(cudaMalloc(&d_input_, in_elems * sizeof(float)));

    out_elems_ = 1;
    for (int d = 0; d < out_dims.nbDims; ++d) out_elems_ *= static_cast<size_t>(out_dims.d[d]);
    CHECK_CUDA(cudaMalloc(&d_out_,  out_elems_ * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_out_, out_elems_ * sizeof(float)));

    // Preproc GPU staging 초기화
    gpu_bgr_ = nullptr;
    gpu_bgr_size_ = 0;

    std::cout << "[INFO] Engine loaded: " << engine_path << std::endl;
    std::cout << "       in=" << in_name_  << " shape=(";
    for (int i=0;i<in_dims.nbDims;++i){ std::cout << in_dims.d[i] << (i+1<in_dims.nbDims? "x":""); }
    std::cout << ")\n";
    std::cout << "       out=" << out_name_ << " shape=(";
    for (int i=0;i<out_dims.nbDims;++i){ std::cout << out_dims.d[i] << (i+1<out_dims.nbDims? "x":""); }
    std::cout << "), elems=" << out_elems_ << std::endl;

    return true;
}

// ---------------------------
// Preprocess
// ---------------------------
void YoloTRT::Preprocess(const cv::Mat& img, float* gpu_input) {
    // letterbox 스타일: 커널이 BGR→RGB/정규화/리사이즈/패딩까지 수행한다고 가정
    const int src_w = img.cols, src_h = img.rows;

    // 입력 해상도 기준 스케일/패딩 계산 (커널이 사용)
    const float scale = std::min(input_w_ / (float)src_w, input_h_ / (float)src_h);
    const int new_w = static_cast<int>(src_w * scale);
    const int new_h = static_cast<int>(src_h * scale);
    const int pad_x = (input_w_ - new_w) / 2;
    const int pad_y = (input_h_ - new_h) / 2;

    // 업로드 버퍼 확보 및 복사
    size_t required = static_cast<size_t>(src_w) * src_h * 3; // BGR8
    if (!gpu_bgr_ || required > gpu_bgr_size_) {
        if (gpu_bgr_) CHECK_CUDA(cudaFree(gpu_bgr_));
        CHECK_CUDA(cudaMalloc(&gpu_bgr_, required));
        gpu_bgr_size_ = required;
    }
    CHECK_CUDA(cudaMemcpyAsync(gpu_bgr_, img.data, required, cudaMemcpyHostToDevice, stream_));

    // GPU 커널 호출: (BGR8 → NCHW FP32, 리사이즈/패드/정규화)
    PreprocessKernelLauncher(
        gpu_bgr_, src_w, src_h,
        gpu_input,
        new_w, new_h, pad_x, pad_y, scale,
        stream_);
}

// ---------------------------
// Inference
// ---------------------------
bool YoloTRT::Infer(const cv::Mat& img, std::vector<Detection>& detections, int img_w, int img_h) {
    // 전처리 → d_input_
    Preprocess(img, reinterpret_cast<float*>(d_input_));

    // 바인딩: 입력 1개, 출력 1개만
    if (!context_->setTensorAddress(in_name_.c_str(),  d_input_)) {
        std::cerr << "[ERR] setTensorAddress input failed\n";
        return false;
    }
    if (!context_->setTensorAddress(out_name_.c_str(), d_out_)) {
        std::cerr << "[ERR] setTensorAddress output failed\n";
        return false;
    }

    // 실행
    if (!context_->enqueueV3(stream_)) {
        std::cerr << "[ERR] enqueueV3 failed" << std::endl;
        return false;
    }

    // 출력 복사
    CHECK_CUDA(cudaMemcpyAsync(h_out_, d_out_, out_elems_ * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_));
    CHECK_CUDA(cudaEventRecord(ready_, stream_));
    CHECK_CUDA(cudaEventSynchronize(ready_));

    // 파싱
    Postprocess(detections, img_w, img_h);
    return true;
}

// ---------------------------
// Postprocess (with unpadding & rescale)
// ---------------------------
// h_out_: shape (1, N, 6) : [x1,y1,x2,y2,conf,cls]
// coords are on 640x640 (letterboxed) space → need to unpad to original frame
void YoloTRT::Postprocess(std::vector<Detection>& detections, int img_w, int img_h) {
    detections.clear();
    if (out_elems_ % 6 != 0) return;
    const int num = static_cast<int>(out_elems_ / 6);
    const float* p = h_out_;

    // Letterbox undo
    float scale = std::min(input_w_ / (float)img_w, input_h_ / (float)img_h);
    int new_w = static_cast<int>(img_w * scale);
    int new_h = static_cast<int>(img_h * scale);
    int pad_x = (input_w_ - new_w) / 2;
    int pad_y = (input_h_ - new_h) / 2;

    auto unpad = [&](float x, float y) -> cv::Point2f {
        float xx = (x - pad_x) / scale;
        float yy = (y - pad_y) / scale;
        return {xx, yy};
    };

    for (int i = 0; i < num; ++i) {
        float x1 = p[0], y1 = p[1], x2 = p[2], y2 = p[3];
        float conf = p[4];
        int cls = static_cast<int>(p[5]);
        p += 6;

        if (conf <= 0.60f) continue; // skip low conf

        // unpad and clamp
        cv::Point2f p1 = unpad(x1, y1);
        cv::Point2f p2 = unpad(x2, y2);
        int xi = std::clamp((int)std::round(p1.x), 0, img_w - 1);
        int yi = std::clamp((int)std::round(p1.y), 0, img_h - 1);
        int xj = std::clamp((int)std::round(p2.x), 0, img_w - 1);
        int yj = std::clamp((int)std::round(p2.y), 0, img_h - 1);

        int w = std::max(0, xj - xi);
        int h = std::max(0, yj - yi);
        if (w <= 1 || h <= 1) continue;

        detections.push_back({cv::Rect(xi, yi, w, h), conf, cls});
    }

    // Debug (optional)
    std::cout << "[Postprocess] Detections: " << detections.size() << std::endl;
}

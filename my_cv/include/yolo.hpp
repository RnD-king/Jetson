#ifndef MY_CV_YOLO_HPP_
#define MY_CV_YOLO_HPP_

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

namespace my_cv {

// Detection 구조체: 박스, confidence, class_id
struct Detection {
    cv::Rect box;
    float    confidence;
    int      class_id;
};

class YoloTRT {
public:
    explicit YoloTRT(const std::string& engine_path);
    ~YoloTRT();

    // 이미지 크기(img_w, img_h): 레터박스 역보정용
    bool Infer(const cv::Mat& input, std::vector<Detection>& detections, int img_w, int img_h);

private:
    // --- TensorRT 핵심 ---
    std::shared_ptr<nvinfer1::IRuntime>        runtime_;
    std::shared_ptr<nvinfer1::ICudaEngine>     engine_;
    std::shared_ptr<nvinfer1::IExecutionContext> context_;

    // --- CUDA ---
    cudaStream_t stream_{};
    cudaEvent_t  ready_{};

    // --- preproc 스크래치 (GPU BGR 업로드) ---
    uint8_t* gpu_bgr_ = nullptr;
    size_t   gpu_bgr_size_ = 0;

    // --- input 메타 ---
    int   input_w_ = 0;
    int   input_h_ = 0;
    size_t input_size_ = 0; // 3*H*W

    // --- device buffers ---
    void* d_input_  = nullptr; // float[3*H*W]
    void* d_num_    = nullptr; // int32[1]            (E2E)
    void* d_boxes_  = nullptr; // float[1,max,4]      (E2E)
    void* d_scores_ = nullptr; // float[1,max]        (E2E)
    void* d_raw_    = nullptr; // float[...]          (RAW fallback)

    // --- host pinned buffers ---
    int32_t* h_num_    = nullptr;
    float*   h_boxes_  = nullptr;
    float*   h_scores_ = nullptr;
    float*   h_raw_    = nullptr;

    // --- engine tensor 이름 ---
    std::string in_name_;
    std::string num_name_;
    std::string boxes_name_;
    std::string scores_name_;
    std::string raw_name_;

    // --- 출력 모드 ---
    // 1 = E2E(EfficientNMS_TRT: num/boxes/scores), 2 = RAW(단일 출력)
    int mode_out_ = 0;
    int max_det_  = 0;   // E2E: boxes 두 번째 차원
    int num_preds_ = 0;  // RAW: 마지막 축(예: 8400)

    // --- legacy placeholders (옛 v5/WTS 코드와의 호환을 위해 남김; 실제로는 미사용) ---
    void*  buffers_[2] = {nullptr, nullptr};
    float* h_out_pinned_ = nullptr;
    size_t out_floats_ = 0;
    size_t out_bytes_  = 0;

private:
    bool LoadEngine(const std::string& engine_path);
    void Preprocess(const cv::Mat& img, float* gpu_input);
    void Postprocess(const float* /*unused*/, std::vector<Detection>& detections, int img_w, int img_h);
};

} // namespace my_cv

#endif // MY_CV_YOLO_HPP_

#ifndef MY_CV_YOLO_HPP_
#define MY_CV_YOLO_HPP_

#include <opencv2/opencv.hpp>
#include <NvInfer.h>          // TensorRT
#include <NvOnnxParser.h>     // (옵션)
#include <cuda_runtime_api.h>

#include <string>
#include <vector>
#include <memory>

namespace my_cv {	

struct Detection {
    cv::Rect box;
    float confidence;
    int class_id;
};

class YoloTRT {
public:
    explicit YoloTRT(const std::string& engine_path);
    ~YoloTRT();

    bool Infer(const cv::Mat& input, std::vector<Detection>& detections, int img_w, int img_h);

private:
    void* buffers_[2];                      // 입력/출력 버퍼
    cudaStream_t stream_;                   // CUDA 스트림

    std::shared_ptr<nvinfer1::IRuntime> runtime_;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::shared_ptr<nvinfer1::IExecutionContext> context_;

    uint8_t* gpu_bgr_ = nullptr;
    size_t gpu_bgr_size_ = 0;
    
    int input_index_;
    int output_index_;
    int input_w_;
    int input_h_;
    int input_size_;

    bool LoadEngine(const std::string& engine_path);
    void Preprocess(const cv::Mat& img, float* gpu_input);
    void Postprocess(const float* output, std::vector<Detection>& detections, int img_w, int img_h);
};

}  // namespace my_cv

#endif  // MY_CV_YOLO_HPP_

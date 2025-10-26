#include "yolo.hpp"  // 헤더 파일 (클래스 선언된 곳)
#include "yololayer.h"
#include "preprocess.cuh"

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>  // <- 이것이 resize 포함
#include <opencv2/cudawarping.hpp>

#include <fstream>   // 파일 입출력 (engine 파일 읽을 때 사용)
#include <iostream>  // 표준 출력 (디버깅용)
#include <cuda_runtime.h>  // CUDA 메모리 할당
#include <cmath>     //
#include <algorithm> // std::max, std::min

using namespace nvinfer1;  // TensorRT 네임스페이스
using namespace my_cv;     // Detection, YoloTRT 등이 정의된 네임스페이스

class Logger : public nvinfer1::ILogger { // ----------------------------------
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

namespace my_cv {
    struct Detection; // forward declaration
    void ApplyNMS(const std::vector<Detection>& input, std::vector<Detection>& output, float iou_thresh);
}

static Logger gLogger;  // 전역 객체로 선언-------------------------------

// 생성자: 엔진 로드 + 메모리 초기화
YoloTRT::YoloTRT(const std::string& engine_path) {
    if (!LoadEngine(engine_path)) {
        std::cerr << "Failed to load TensorRT engine!" << std::endl;
        std::exit(1);  // 실패 시 강제 종료
    }

    // 입력/출력 버퍼 초기화
    cudaMalloc(&buffers_[0], input_size_ * sizeof(float));  // 입력
    cudaMalloc(&buffers_[1], 1000 * sizeof(float));         // 출력 (여유 있게 1000)

    cudaStreamCreate(&stream_);  // CUDA 스트림 생성
}

// 소멸자: 리소스 정리
YoloTRT::~YoloTRT() {
    cudaStreamDestroy(stream_);
    cudaFree(buffers_[0]);
    cudaFree(buffers_[1]);
    if (gpu_bgr_) {
        cudaFree(gpu_bgr_);
    }
}

// 엔진 로드: .engine 파일 → 엔진 객체로 디시리얼라이즈
bool YoloTRT::LoadEngine(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file) return false;

    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);

    runtime_ = std::shared_ptr<IRuntime>(createInferRuntime(gLogger));  // 멤버변수 초기화
    engine_ = std::shared_ptr<ICudaEngine>(runtime_->deserializeCudaEngine(engine_data.data(), size));
    context_ = std::shared_ptr<IExecutionContext>(engine_->createExecutionContext());

    
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    const char* tensor_name = engine_->getIOTensorName(i);
    std::cout << "I/O tensor " << i << ": " << tensor_name << std::endl;
}


    // 입력/출력 정보 가져오기
    input_index_ = 0;  // input 이름 (YOLO 엔진마다 다를 수 있음)
    output_index_ = 1; // output 이름

    input_w_ = 640;  // YOLO 입력 해상도 (고정)
    input_h_ = 640;
    input_size_ = 3 * input_w_ * input_h_;  // RGB 3채널

    return true;
}

// 추론 함수
bool YoloTRT::Infer(const cv::Mat& img, std::vector<Detection>& detections, int img_w, int img_h) {
    using Clock = std::chrono::high_resolution_clock;
    auto t1 = Clock::now();
    
    // 1. 전처리
    Preprocess(img, static_cast<float*>(buffers_[0]));
    auto t2 = Clock::now();

    // 2. 추론 실행
    void* bindings[] = {buffers_[0], buffers_[1]};
    context_->setInputTensorAddress("data", bindings[0]);
    context_->setOutputTensorAddress("prob", bindings[1]);
    context_->enqueueV3(stream_);
    auto t2_5 = Clock::now();
    cudaStreamSynchronize(stream_);
    auto t3 = Clock::now();

    // 3. GPU > CPU 복사
    float output[1000];
    cudaMemcpy(output, buffers_[1], sizeof(float) * 1000, cudaMemcpyDeviceToHost); 
    int num_detected = static_cast<int>(output[0]);
    std::cout << "[INFO] YOLO detected " << num_detected << " objects" << std::endl;
    auto t4 = Clock::now();

    // 4. 후처리
    Postprocess(output, detections, img_w, img_h);
    auto t5 = Clock::now();

    // 시간 출력
    float dt_pre = std::chrono::duration<float, std::milli>(t2 - t1).count();
    float dt_infer = std::chrono::duration<float, std::milli>(t2_5 - t2).count();
    float dt_sync = std::chrono::duration<float, std::milli>(t3 - t2_5).count();
    float dt_memcpy = std::chrono::duration<float, std::milli>(t4 - t3).count();
    float dt_post = std::chrono::duration<float, std::milli>(t5 - t4).count();
    float dt_total = std::chrono::duration<float, std::milli>(t5 - t1).count();

    std::cout << "[TRT(ms)] Total:" << dt_total
              << " | Preprocess: " << dt_pre 
              << " | Inference: " << dt_infer
              << " | Syncronize: " << dt_sync
              << " | Memcpy D2H: " << dt_memcpy 
              << " | Postprocess: " << dt_post << std::endl;

    return true;
}


// 전처리: 이미지 → float 배열 (정규화 + 리사이즈)
void YoloTRT::Preprocess(const cv::Mat& img_bgr, float* gpu_input) {
    int input_w = img_bgr.cols;
    int input_h = img_bgr.rows;
    float scale = std::min(input_w_ / (float)input_w, input_h_ / (float)input_h);
    int resized_w = int(input_w * scale);
    int resized_h = int(input_h * scale);
    int pad_x = (input_w_ - resized_w) / 2;
    int pad_y = (input_h_ - resized_h) / 2;

    // (1) 입력 이미지 크기 기반으로 필요한 GPU 메모리 계산
    size_t required_size = input_w * input_h * 3 * sizeof(uint8_t);

    // (2) 최초 또는 크기 변경 시에만 메모리 할당
    if (gpu_bgr_ == nullptr || required_size > gpu_bgr_size_) {
        if (gpu_bgr_) {
            cudaFree(gpu_bgr_);
        }
        cudaMalloc(&gpu_bgr_, required_size);
        gpu_bgr_size_ = required_size;
    }

    // (3) GPU로 이미지 업로드
    cudaMemcpyAsync(gpu_bgr_, img_bgr.data, required_size, cudaMemcpyHostToDevice, stream_);

    // (4) CUDA 전처리 커널 실행
    PreprocessKernelLauncher(
        gpu_bgr_,               // const uint8_t* input_bgr
        input_w, input_h,       // input 이미지 크기
        gpu_input,              // float* CHW 형식 출력 (TensorRT 입력용)
        resized_w, resized_h,   // YOLO에 맞는 크기로 리사이즈
        pad_x, pad_y,           // 패딩
        scale,                  // 스케일 비율
        stream_                 // CUDA 스트림
    );

    // (5) 커널 실행 오류 확인 (선택사항)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[CUDA KERNEL ERROR] " << cudaGetErrorString(err) << std::endl;
    }
}


// 후처리: YOLO 출력 → 바운딩 박스 감지로 변환
void YoloTRT::Postprocess(const float* output, std::vector<Detection>& detections, int img_w, int img_h) {
    detections.clear();
    std::vector<Detection> raw_boxes;  //  임시 저장용

    int num_detections = static_cast<int>(output[0]);
    RawDetection* raw = (RawDetection*)(output + 1);

    const int det_size = 6;

    float scale = std::min(input_w_ / (float)img_w, input_h_ / (float)img_h);
    float pad_x = (input_w_ - img_w * scale) / 2;
    float pad_y = (input_h_ - img_h * scale) / 2;

    for (int i = 0; i < num_detections; ++i) {
        float cx = (raw[i].bbox[0] - pad_x) / scale;
        float cy = (raw[i].bbox[1] - pad_y) / scale;
        float w  = raw[i].bbox[2] / scale;
        float h  = raw[i].bbox[3] / scale;
        float confidence = raw[i].confidence;
        int class_id = static_cast<int>(raw[i].class_id);

        if (confidence < 0.7f) continue;

        int x = static_cast<int>(cx - w / 2);
        int y = static_cast<int>(cy - h / 2);

        raw_boxes.push_back({cv::Rect(x, y, static_cast<int>(w), static_cast<int>(h)), confidence, class_id});
    }
  
    my_cv::ApplyNMS(raw_boxes, detections, 0.45f);

}



static float IoU(const Detection& a, const Detection& b) {
    auto inter = (a.box & b.box).area();
    auto uni = a.box.area() + b.box.area() - inter;
    return static_cast<float>(inter) / (uni + 1e-6f);
}



// 정의
namespace my_cv {
    void ApplyNMS(const std::vector<Detection>& input, std::vector<Detection>& output, float iou_thresh) {
        output.clear();
        std::map<int, std::vector<Detection>> class_map;
        for (const auto& det : input) {
            class_map[det.class_id].push_back(det);
        }

        for (auto& [cls, boxes] : class_map) {
            std::sort(boxes.begin(), boxes.end(), [](const Detection& a, const Detection& b) {
                return a.confidence > b.confidence;
            });

            std::vector<bool> removed(boxes.size(), false);
            for (size_t i = 0; i < boxes.size(); ++i) {
                if (removed[i]) continue;
                output.push_back(boxes[i]);
                for (size_t j = i + 1; j < boxes.size(); ++j) {
                    if (removed[j]) continue;
                    float iou = IoU(boxes[i], boxes[j]);
                    if (iou > iou_thresh) {
                        removed[j] = true;
                    }
                }
            }
        }
    }
}


// # 메모리 터지면 CTL에서
// sudo systemctl stop gdm
// # 커서만 보이면 ctrl + alt + F2 (F1 or F3)
// # 로그인 ID: rnd / PW: rnd 


// # .pt > .onnx 변환 (가상환경에서 .pt 있는 위치로)
// yolo export \
// model=best_8s.pt \
// format=onnx \
// simplify=True \
// dynamic=False \
// imgsz=640 \
// half=True \
// opset=12 \
// nms=True \
// device=0

// dynamic - 동적입력 / imgsz - 해상도 / half - FP32 > FP16 / nms - 자동 NMS


// # .onnx > .engine 빌드 (가상환경에서 같은 위치로)
// /usr/src/tensorrt/bin/trtexec \
// --onnx=best_8s.onnx \
// --saveEngine=best_8s_16.engine \
// --fp16 \
// --memPoolSize=workspace:4096 \
// --noTF32 \
// --verbose

// memPoolSize - 용량

// # CTL 나오기
// sudo systemctl start gdm

#pragma once

#include "config.h"

struct YoloKernel {
    int width;
    int height;
    float anchors[kNumAnchor * 2];
};

struct alignas(float) RawDetection {
    float bbox[4];  // center_x center_y w h
    float confidence;     // bbox_conf * cls_conf
    float class_id;
    // float mask[32]; 메모리 사용의 주범!
};

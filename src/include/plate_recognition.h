#ifndef PLATE_RECOGNITION_H
#define PLATE_RECOGNITION_H

#include "utils.h" // AppConfig

typedef struct {
    char plate_text[64];
    float confidence;
    int vehicle_bbox[4]; // x, y, w, h
    int plate_bbox[4];   // x, y, w, h
    int is_fraud;        // 1: 欺诈, 0: 正常
} DetectionResult;

// 初始化模型
int system_init(AppConfig* config);
// 处理一帧
DetectionResult* process_frame(unsigned char* rgb_data, int width, int height, int* count);
// 清理
void system_cleanup();

#endif
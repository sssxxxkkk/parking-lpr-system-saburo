#ifndef PLATE_RECOGNITION_H
#define PLATE_RECOGNITION_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

// 模型路径配置
#define VEHICLE_MODEL_PATH "models/vehicle_detector.onnx"
#define PLATE_MODEL_PATH "models/plate_detector.onnx"
#define OCR_MODEL_PATH "models/ppocr_rec.onnx"

// 图像参数
#define MAX_IMAGE_WIDTH 1920
#define MAX_IMAGE_HEIGHT 1080
#define MODEL_INPUT_SIZE 640

// 检测结果结构
typedef struct {
    char plate_text[16];          // 识别出的车牌文本
    float confidence;             // 总体置信度
    int vehicle_bbox[4];          // 车辆边界框 [x1,y1,x2,y2]
    int plate_bbox[4];            // 车牌边界框 [x1,y1,x2,y2]
    time_t timestamp;             // 检测时间
    bool is_fraud;                // 是否欺诈
    char fraud_reason[64];        // 欺诈原因
} DetectionResult;

// 系统配置
typedef struct {
    // 模型配置
    char vehicle_model[256];
    char plate_model[256];
    char ocr_model[256];
    
    // 检测阈值
    float vehicle_threshold;
    float plate_threshold;
    float ocr_threshold;
    
    // 性能配置
    int processing_interval;
    int max_detection_per_frame;
    bool enable_anti_fraud;
    
    // 摄像头配置
    char camera_device[64];
    int camera_width;
    int camera_height;
    int fps;
} SystemConfig;

// 函数声明
int system_init(SystemConfig* config);
DetectionResult* process_frame(unsigned char* image_data, int width, int height, int* result_count);
void system_cleanup();
void signal_handler(int sig);

#endif
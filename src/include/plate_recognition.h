#ifndef PLATE_RECOGNITION_H
#define PLATE_RECOGNITION_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

// 模型路径配置
#define VEHICLE_MODEL_PATH "models/yolov5s.onnx"
#define PLATE_DETECTOR_PATH "models/ppocr_det.onnx"
#define OCR_MODEL_PATH "models/ppocr_rec.onnx"

// 图像参数
#define MAX_IMAGE_WIDTH 1920
#define MAX_IMAGE_HEIGHT 1080
#define YOLO_INPUT_SIZE 640

// 检测结果结构
typedef struct {
    char plate_text[16];
    float confidence;
    int vehicle_bbox[4];
    int plate_bbox[4];
    time_t timestamp;
    bool is_fraud;
    char fraud_reason[64];
} DetectionResult;

// 系统配置
typedef struct {
    char vehicle_model[256];
    char plate_detector_model[256];
    char ocr_model[256];
    float vehicle_threshold;
    float plate_threshold;
    float ocr_threshold;
    int processing_interval;
    int max_detection_per_frame;
    bool enable_anti_fraud;
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
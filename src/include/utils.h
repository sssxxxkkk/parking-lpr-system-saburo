#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 宏定义
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define CLAMP(x, min, max) ((x) < (min) ? (min) : ((x) > (max) ? (max) : (x)))

// 配置解析
typedef struct {
    char camera_device[64];
    int camera_width;
    int camera_height;
    char vehicle_model[256];
    char plate_model[256];
    char ocr_model[256];
    float vehicle_threshold;
    float plate_threshold;
    float ocr_threshold;
} AppConfig;

int config_parse(AppConfig* config, const char* config_path);

#endif
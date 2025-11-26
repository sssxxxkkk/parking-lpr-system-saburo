#ifndef COMMON_TYPES_H
#define COMMON_TYPES_H
#include <stdint.h>

// 基础图像结构 (RGB)
typedef struct {
    uint8_t* data;
    int width;
    int height;
    int channels; // 通常为3
} Image;

// 检测框
typedef struct {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
} Detection;

#endif
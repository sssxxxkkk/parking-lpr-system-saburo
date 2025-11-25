#ifndef COMMON_TYPES_H
#define COMMON_TYPES_H

#include <stdint.h>

// 基础图像类型
typedef struct {
    uint8_t* data;
    int width;
    int height;
    int channels;
} Image;

#endif
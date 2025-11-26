#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include "common_types.h"

// 基础函数
Image image_create(int width, int height, int channels);
void image_free(Image* img);
Image image_resize(const Image* src, int new_width, int new_height);
Image image_crop(const Image* src, int x, int y, int width, int height);
float* image_to_float_array(const Image* img);

// 预处理函数
Image preprocess_for_yolo(const unsigned char* image_data, int width, int height, int target_size);
Image preprocess_for_ocr(const Image* src, int target_height);

#endif
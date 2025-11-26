#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include "common_types.h"

// YOLO 预处理 (Resize + Pad + Normalize)
void preprocess_yolo(const unsigned char* src, int w, int h, int target_size, float* dst);
// YOLO 后处理
void postprocess_yolo(float* data, int num_rows, float conf_thres, int img_w, int img_h, Detection* dets, int* count);

// DBNet (车牌定位) 预处理
void preprocess_dbnet(const unsigned char* src, int w, int h, int target_size, float* dst);
// DBNet 后处理 (从热力图找框)
void postprocess_dbnet(float* map, int map_w, int map_h, float thresh, int* x, int* y, int* w, int* h);

// CRNN (文字识别) 预处理
void preprocess_ocr(const unsigned char* src, int w, int h, float* dst);

// 图像裁剪
void crop_image_rgb(const unsigned char* src, int src_w, int src_h, int x, int y, int w, int h, unsigned char* dst);

#endif
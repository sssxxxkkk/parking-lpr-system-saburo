#include "include/image_utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

//----------------------
// 基础图像结构
//----------------------
Image image_create(int width, int height, int channels)
{
    Image img;
    img.width = width;
    img.height = height;
    img.channels = channels;
    img.data = (uint8_t*)malloc(width * height * channels);
    memset(img.data, 0, width * height * channels);
    return img;
}

void image_free(Image* img)
{
    if (img && img->data) {
        free(img->data);
        img->data = NULL;
    }
}

//----------------------
// 简易最近邻 resize
//----------------------
Image image_resize(const Image* src, int new_w, int new_h)
{
    Image out = image_create(new_w, new_h, src->channels);

    for (int y = 0; y < new_h; y++) {
        int sy = y * src->height / new_h;
        for (int x = 0; x < new_w; x++) {
            int sx = x * src->width / new_w;

            for (int c = 0; c < src->channels; c++) {
                out.data[(y * new_w + x) * src->channels + c] =
                    src->data[(sy * src->width + sx) * src->channels + c];
            }
        }
    }
    return out;
}

//----------------------
// YOLO 输入预处理：resize 到 640×640
//----------------------
Image preprocess_for_yolo(const unsigned char* image_data,
                          int width, int height,
                          int target_size)
{
    Image src;
    src.width = width;
    src.height = height;
    src.channels = 3;
    src.data = (uint8_t*)image_data;

    // 直接最近邻缩放到 640×640
    return image_resize(&src, target_size, target_size);
}

//----------------------
// OCR 预处理（可保持简单）
//----------------------
Image preprocess_for_ocr(const Image* src, int target_height)
{
    float scale = (float)target_height / src->height;
    int new_w = (int)(src->width * scale);

    return image_resize(src, new_w, target_height);
}

//----------------------
// 将图像转换为 float 数组 (NCHW)
//----------------------
float* image_to_float_array(const Image* img)
{
    int total = img->width * img->height * img->channels;
    float* out = (float*)malloc(sizeof(float) * total);

    int idx = 0;
    for (int c = 0; c < img->channels; c++) {
        for (int i = 0; i < img->width * img->height; i++) {
            out[idx++] = img->data[i * img->channels + c] / 255.0f;
        }
    }
    return out;
}
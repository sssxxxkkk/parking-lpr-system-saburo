#include "include/image_utils.h"
#include "include/utils.h"
#include <stdlib.h>
#include <string.h>

// 基础图像函数实现
Image image_create(int width, int height, int channels) {
    Image img;
    img.width = width;
    img.height = height;
    img.channels = channels;
    img.data = malloc(width * height * channels * sizeof(uint8_t));
    return img;
}

void image_free(Image* img) {
    if (img && img->data) {
        free(img->data);
        img->data = NULL;
    }
}

Image image_resize(const Image* src, int new_width, int new_height) {
    // 简化实现 - 返回一个空图像
    return image_create(new_width, new_height, src->channels);
}

Image image_crop(const Image* src, int x, int y, int width, int height) {
    // 简化实现 - 返回一个空图像
    (void)x; (void)y;
    return image_create(width, height, src->channels);
}

Image preprocess_for_yolo(const unsigned char* image_data, int width, int height, int target_size) {
    // 简化实现
    (void)image_data; (void)width; (void)height;
    return image_create(target_size, target_size, 3);
}

Image preprocess_for_ocr(const Image* src, int target_height) {
    // 简化实现
    return image_create(src->width, target_height, src->channels);
}
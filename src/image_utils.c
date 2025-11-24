#include "include/image_utils.h"
#include "include/utils.h"
#include <stdlib.h>
#include <string.h>

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
    Image dst = image_create(new_width, new_height, src->channels);
    
    float x_ratio = (float)src->width / new_width;
    float y_ratio = (float)src->height / new_height;
    
    for (int y = 0; y < new_height; y++) {
        for (int x = 0; x < new_width; x++) {
            int src_x = (int)(x * x_ratio);
            int src_y = (int)(y * y_ratio);
            
            for (int c = 0; c < src->channels; c++) {
                int src_idx = (src_y * src->width + src_x) * src->channels + c;
                int dst_idx = (y * new_width + x) * dst.channels + c;
                dst.data[dst_idx] = src->data[src_idx];
            }
        }
    }
    
    return dst;
}

Image image_crop(const Image* src, int x, int y, int width, int height) {
    // 边界检查
    x = MAX(0, MIN(x, src->width - 1));
    y = MAX(0, MIN(y, src->height - 1));
    width = MIN(width, src->width - x);
    height = MIN(height, src->height - y);
    
    Image crop = image_create(width, height, src->channels);
    
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int src_idx = ((y + row) * src->width + (x + col)) * src->channels;
            int dst_idx = (row * width + col) * crop.channels;
            
            memcpy(&crop.data[dst_idx], &src->data[src_idx], src->channels);
        }
    }
    
    return crop;
}

Image preprocess_for_yolo(const unsigned char* image_data, int width, int height, int target_size) {
    // 创建源图像结构
    Image src;
    src.data = (uint8_t*)image_data;
    src.width = width;
    src.height = height;
    src.channels = 3;
    
    // 1. 调整大小 (保持宽高比)
    int new_width, new_height;
    float scale = MIN((float)target_size / width, (float)target_size / height);
    new_width = (int)(width * scale);
    new_height = (int)(height * scale);
    
    Image resized = image_resize(&src, new_width, new_height);
    
    // 2. 创建目标图像 (填充到target_size x target_size)
    Image processed = image_create(target_size, target_size, 3);
    memset(processed.data, 114, target_size * target_size * 3); // 填充灰色
    
    // 3. 将调整大小后的图像放到中心
    int x_offset = (target_size - new_width) / 2;
    int y_offset = (target_size - new_height) / 2;
    
    for (int y = 0; y < new_height; y++) {
        for (int x = 0; x < new_width; x++) {
            int src_idx = (y * new_width + x) * 3;
            int dst_idx = ((y_offset + y) * target_size + (x_offset + x)) * 3;
            
            // BGR转RGB (如果需要)
            processed.data[dst_idx] = resized.data[src_idx + 2];     // R
            processed.data[dst_idx + 1] = resized.data[src_idx + 1]; // G
            processed.data[dst_idx + 2] = resized.data[src_idx];     // B
        }
    }
    
    image_free(&resized);
    return processed;
}

Image preprocess_for_ocr(const Image* src, int target_height) {
    // 计算新宽度 (保持宽高比)
    int new_width = (int)((float)src->width * target_height / src->height);
    Image resized = image_resize(src, new_width, target_height);
    
    // 这里可以添加更多的OCR特定预处理
    // 比如: 灰度化、二值化、对比度增强等
    
    return resized;
}

int image_save(const Image* img, const char* filename) {
    // 简化实现 - 实际应该使用libjpeg或libpng
    FILE* fp = fopen(filename, "wb");
    if (!fp) return -1;
    
    // 这里应该写入正确的图像格式
    fwrite(img->data, 1, img->width * img->height * img->channels, fp);
    fclose(fp);
    return 0;
}
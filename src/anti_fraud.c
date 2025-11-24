#include "include/anti_fraud.h"
#include <string.h>
#include <ctype.h>

bool detect_fraud(const Image* plate_image, const char* plate_text, 
                  float ocr_confidence, char* fraud_reason) {
    
    bool is_fraud = false;
    
    // 检查1: OCR置信度过低
    if (ocr_confidence < 0.5f) {
        is_fraud = true;
        strcpy(fraud_reason, "OCR置信度过低");
        return is_fraud;
    }
    
    // 检查2: 车牌格式验证
    if (!validate_plate_format(plate_text)) {
        is_fraud = true;
        strcpy(fraud_reason, "车牌格式无效");
        return is_fraud;
    }
    
    // 检查3: 图像质量检测
    float image_quality = assess_image_quality(plate_image);
    if (image_quality < 0.3f) {
        is_fraud = true;
        strcpy(fraud_reason, "图像质量差");
        return is_fraud;
    }
    
    strcpy(fraud_reason, "正常");
    return is_fraud;
}

bool validate_plate_format(const char* plate_text) {
    // 验证中国大陆车牌格式
    int len = strlen(plate_text);
    if (len < 7 || len > 8) return false;
    
    // 检查省份简称
    const char* provinces = "京津晋冀蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新渝";
    if (strchr(provinces, plate_text[0]) == NULL) return false;
    
    // 检查后续字符 (字母和数字)
    for (int i = 1; i < len; i++) {
        if (!isalnum(plate_text[i])) return false;
    }
    
    return true;
}

float assess_image_quality(const Image* img) {
    // 简单的图像质量评估
    // 这里可以实现更复杂的质量评估算法
    
    // 简化实现: 计算图像对比度
    long sum = 0;
    long count = img->width * img->height;
    
    for (int i = 0; i < count; i++) {
        sum += img->data[i * img->channels]; // 使用R通道
    }
    
    float avg_brightness = (float)sum / count / 255.0f;
    
    // 亮度在0.3-0.7之间认为质量较好
    if (avg_brightness >= 0.3f && avg_brightness <= 0.7f) {
        return 0.8f;
    } else {
        return 0.3f;
    }
}

bool verify_plate_color(const Image* img) {
    // 验证车牌颜色 (蓝色、黄色、绿色等)
    // 简化实现: 总是返回true
    return true;
}
#ifndef ANTI_FRAUD_H
#define ANTI_FRAUD_H

#include "image_utils.h"
#include <stdbool.h>

// 函数声明
bool detect_fraud(const Image* plate_image, const char* plate_text, 
                  float ocr_confidence, char* fraud_reason);
bool validate_plate_format(const char* plate_text);
float assess_image_quality(const Image* img);
bool verify_plate_color(const Image* img);

#endif
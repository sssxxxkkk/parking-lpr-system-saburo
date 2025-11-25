#ifndef ONNX_INFERENCE_H
#define ONNX_INFERENCE_H

#include <onnxruntime_c_api.h>
#include <stddef.h>
#include "common_types.h"

typedef struct {
    OrtEnv* env;
    OrtSession* session;
    OrtSessionOptions* session_options;
    OrtMemoryInfo* memory_info;
    
    char** input_names;
    char** output_names;
    int input_count;
    int output_count;
    
    int input_width;
    int input_height;
    int input_channels;
} ONNXModel;

// 检测结果
typedef struct {
    float x1, y1, x2, y2;   // 框坐标
    float confidence;        // 置信度
    int class_id;            // 类别
    char class_name[32];     // 类名
} Detection;

// PP-OCR 识别结果
typedef struct {
    char text[32];
    float confidence;
} PPOCRRecResult;

// 基础函数声明
int onnx_model_init(ONNXModel* model, const char* model_path);
int onnx_model_predict(ONNXModel* model, const float* input_data, 
                      size_t input_size, float** output, size_t* output_size);
Detection* yolo_postprocess(float* output, int output_size,
                            int img_w, int img_h,
                            float conf_thresh, int* det_count);
void onnx_model_cleanup(ONNXModel* model);
float* image_to_float_array(const unsigned char* image_data, int width, int height, int channels,
                           float mean[3], float std[3]);

// PP-OCR 专用函数声明
Image preprocess_for_ppocr_det(const unsigned char* image_data, int width, int height);
Image preprocess_for_ppocr_rec(const Image* src);
int* ppocr_det_postprocess(float* output, int output_height, int output_width, 
                          int original_height, int original_width, 
                          float threshold, int* box_count);
PPOCRRecResult ppocr_rec_postprocess(float* output, int output_len, int dict_size);

#endif
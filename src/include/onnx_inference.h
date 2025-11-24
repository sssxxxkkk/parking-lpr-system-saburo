#ifndef ONNX_INFERENCE_H
#define ONNX_INFERENCE_H

#include <onnxruntime_c_api.h>

typedef struct {
    OrtEnv* env;
    OrtSession* session;
    OrtSessionOptions* session_options;
    OrtMemoryInfo* memory_info;
    
    // 输入输出信息
    char** input_names;
    char** output_names;
    int input_count;
    int output_count;
    
    // 模型信息
    int input_width;
    int input_height;
    int input_channels;
} ONNXModel;

// 检测结果
typedef struct {
    float bbox[4];    // [x1, y1, x2, y2]
    float confidence;
    int class_id;
    char class_name[32];
} Detection;

// 函数声明
int onnx_model_init(ONNXModel* model, const char* model_path);
int onnx_model_predict(ONNXModel* model, const float* input_data, 
                      size_t input_size, float** output, size_t* output_size);
Detection* yolo_postprocess(float* output, int output_size, 
                           int original_width, int original_height,
                           float confidence_threshold, int* detection_count);
void onnx_model_cleanup(ONNXModel* model);
float* image_to_float_array(const unsigned char* image_data, int width, int height, int channels,
                           float mean[3], float std[3]);

#endif
#ifndef ONNX_INFERENCE_H
#define ONNX_INFERENCE_H

#include <onnxruntime_c_api.h>
#include <stdint.h>

typedef struct {
    OrtEnv* env;
    OrtSession* session;
    OrtSessionOptions* session_options;
    OrtMemoryInfo* memory_info;
    char** input_names;
    char** output_names;
    size_t input_count;
    size_t output_count;
} ONNXModel;

int onnx_model_init(ONNXModel* model, const char* model_path);
int onnx_model_predict(ONNXModel* model, 
                       const float* input_data, 
                       const int64_t* input_shape, 
                       size_t shape_len, 
                       float** output_data, 
                       size_t* output_size);
void onnx_model_cleanup(ONNXModel* model);

#endif
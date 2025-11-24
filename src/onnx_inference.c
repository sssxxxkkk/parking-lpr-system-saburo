#include "include/onnx_inference.h"
#include "include/utils.h"
#include <stdlib.h>

static void OrtCheck(OrtStatus* status, const char* message) {
    if (status != NULL) {
        const char* msg = OrtGetErrorMessage(status);
        fprintf(stderr, "%s: %s\n", message, msg);
        OrtReleaseStatus(status);
        exit(1);
    }
}

int onnx_model_init(ONNXModel* model, const char* model_path) {
    printf("加载模型: %s\n", model_path);
    
    // 初始化ONNX Runtime环境
    OrtStatus* status = OrtCreateEnv(ORT_LOGGING_LEVEL_WARNING, "PlateRecognition", &model->env);
    OrtCheck(status, "Failed to create ONNX Runtime environment");
    
    // 创建会话选项
    status = OrtCreateSessionOptions(&model->session_options);
    OrtCheck(status, "Failed to create session options");
    
    // 设置优化选项
    OrtSetSessionThreadPoolSize(model->session_options, 1);
    OrtSetInterOpNumThreads(model->session_options, 1);
    OrtSetIntraOpNumThreads(model->session_options, 1);
    
    // 创建会话
    status = OrtCreateSession(model->env, model_path, model->session_options, &model->session);
    OrtCheck(status, "Failed to create ONNX session");
    
    // 获取内存信息
    status = OrtCreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &model->memory_info);
    OrtCheck(status, "Failed to create memory info");
    
    // 获取输入输出信息
    OrtSessionGetInputCount(model->session, &model->input_count);
    OrtSessionGetOutputCount(model->session, &model->output_count);
    
    // 分配输入输出名称数组
    model->input_names = malloc(model->input_count * sizeof(char*));
    model->output_names = malloc(model->output_count * sizeof(char*));
    
    // 获取输入输出名称和维度
    for (size_t i = 0; i < model->input_count; i++) {
        OrtTypeInfo* type_info;
        status = OrtSessionGetInputTypeInfo(model->session, i, &type_info);
        OrtCheck(status, "Failed to get input type info");
        
        const OrtTensorTypeAndShapeInfo* tensor_info;
        OrtCastTypeInfoToTensorInfo(type_info, &tensor_info);
        
        // 获取输入维度
        size_t num_dims;
        OrtGetDimensionsCount(tensor_info, &num_dims);
        int64_t* dims = malloc(num_dims * sizeof(int64_t));
        OrtGetDimensions(tensor_info, dims, num_dims);
        
        if (num_dims == 4) { // NCHW格式
            model->input_height = (int)dims[2];
            model->input_width = (int)dims[3];
            model->input_channels = (int)dims[1];
        }
        
        free(dims);
        OrtReleaseTypeInfo(type_info);
        
        // 获取输入名称
        char* input_name;
        OrtSessionGetInputName(model->session, i, OrtAllocatorDefault, &input_name);
        model->input_names[i] = strdup(input_name);
        OrtAllocatorFree(OrtAllocatorDefault, input_name);
    }
    
    for (size_t i = 0; i < model->output_count; i++) {
        char* output_name;
        OrtSessionGetOutputName(model->session, i, OrtAllocatorDefault, &output_name);
        model->output_names[i] = strdup(output_name);
        OrtAllocatorFree(OrtAllocatorDefault, output_name);
    }
    
    printf("模型加载成功: %s\n", model_path);
    printf("输入尺寸: %dx%dx%d\n", model->input_width, model->input_height, model->input_channels);
    
    return 0;
}

int onnx_model_predict(ONNXModel* model, const float* input_data, 
                      size_t input_size, float** output, size_t* output_size) {
    
    // 创建输入tensor
    int64_t input_shape[] = {1, model->input_channels, model->input_height, model->input_width};
    size_t input_shape_len = sizeof(input_shape) / sizeof(input_shape[0]);
    
    OrtValue* input_tensor = NULL;
    OrtStatus* status = OrtCreateTensorWithDataAsOrtValue(
        model->memory_info, (void*)input_data, input_size * sizeof(float),
        input_shape, input_shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);
    OrtCheck(status, "Failed to create input tensor");
    
    // 准备输出tensor数组
    OrtValue* output_tensors[model->output_count];
    for (size_t i = 0; i < model->output_count; i++) {
        output_tensors[i] = NULL;
    }
    
    // 执行推理
    status = OrtRun(model->session, NULL, 
                   (const char* const*)model->input_names, &input_tensor, 1,
                   (const char* const*)model->output_names, model->output_count, output_tensors);
    OrtCheck(status, "Failed to run inference");
    
    // 获取输出数据
    float* output_data;
    OrtGetTensorMutableData(output_tensors[0], (void**)&output_data);
    
    // 获取输出尺寸
    OrtTypeInfo* type_info;
    OrtSessionGetOutputTypeInfo(model->session, 0, &type_info);
    const OrtTensorTypeAndShapeInfo* tensor_info;
    OrtCastTypeInfoToTensorInfo(type_info, &tensor_info);
    
    size_t num_dims;
    OrtGetDimensionsCount(tensor_info, &num_dims);
    int64_t* dims = malloc(num_dims * sizeof(int64_t));
    OrtGetDimensions(tensor_info, dims, num_dims);
    
    // 计算输出元素数量
    size_t element_count = 1;
    for (size_t i = 0; i < num_dims; i++) {
        element_count *= dims[i];
    }
    
    // 分配输出内存
    *output = malloc(element_count * sizeof(float));
    *output_size = element_count;
    memcpy(*output, output_data, element_count * sizeof(float));
    
    // 清理资源
    free(dims);
    OrtReleaseTypeInfo(type_info);
    OrtReleaseValue(input_tensor);
    for (size_t i = 0; i < model->output_count; i++) {
        OrtReleaseValue(output_tensors[i]);
    }
    
    return 0;
}

Detection* yolo_postprocess(float* output, int output_size, 
                           int original_width, int original_height,
                           float confidence_threshold, int* detection_count) {
    
    // 简化的YOLO输出解析
    // 实际实现需要根据具体的YOLO输出格式调整
    
    int max_detections = 100;
    Detection* detections = malloc(max_detections * sizeof(Detection));
    int count = 0;
    
    // 假设输出格式为: [batch, num_detections, 6] 其中6为 [x1,y1,x2,y2,conf,class]
    for (int i = 0; i < output_size && count < max_detections; i += 6) {
        float confidence = output[i + 4];
        
        if (confidence > confidence_threshold) {
            Detection det;
            det.bbox[0] = output[i] * original_width;     // x1
            det.bbox[1] = output[i + 1] * original_height; // y1
            det.bbox[2] = output[i + 2] * original_width;  // x2
            det.bbox[3] = output[i + 3] * original_height; // y2
            det.confidence = confidence;
            det.class_id = (int)output[i + 5];
            
            // 根据class_id设置类别名称
            if (det.class_id == 0) strcpy(det.class_name, "vehicle");
            else if (det.class_id == 1) strcpy(det.class_name, "license_plate");
            else strcpy(det.class_name, "unknown");
            
            detections[count++] = det;
        }
    }
    
    *detection_count = count;
    return detections;
}

void onnx_model_cleanup(ONNXModel* model) {
    if (model->session) OrtReleaseSession(model->session);
    if (model->session_options) OrtReleaseSessionOptions(model->session_options);
    if (model->memory_info) OrtReleaseMemoryInfo(model->memory_info);
    if (model->env) OrtReleaseEnv(model->env);
    
    // 释放名称数组
    for (size_t i = 0; i < model->input_count; i++) {
        free(model->input_names[i]);
    }
    for (size_t i = 0; i < model->output_count; i++) {
        free(model->output_names[i]);
    }
    
    free(model->input_names);
    free(model->output_names);
}

float* image_to_float_array(const unsigned char* image_data, int width, int height, int channels,
                           float mean[3], float std[3]) {
    size_t total_pixels = width * height * channels;
    float* float_data = malloc(total_pixels * sizeof(float));
    
    for (size_t i = 0; i < total_pixels; i++) {
        int channel = i % channels;
        float_data[i] = ((float)image_data[i] / 255.0f - mean[channel]) / std[channel];
    }
    
    return float_data;
}
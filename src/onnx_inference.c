#include "include/onnx_inference.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static const OrtApi* g_ort = NULL;

int onnx_model_init(ONNXModel* m, const char* path) {
    if (!g_ort) g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    
    if (g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &m->env) != NULL) return -1;
    if (g_ort->CreateSessionOptions(&m->session_options) != NULL) return -1;
    if (g_ort->CreateSession(m->env, path, m->session_options, &m->session) != NULL) {
        printf("无法加载模型: %s\n", path);
        return -1;
    }
    
    OrtAllocator* allocator;
    g_ort->GetAllocatorWithDefaultOptions(&allocator);
    
    // 输入名称 (简化，只取第1个)
    char* name;
    g_ort->SessionGetInputName(m->session, 0, allocator, &name);
    m->input_names = malloc(sizeof(char*));
    m->input_names[0] = strdup(name);
    allocator->Free(allocator, name);
    
    // 输出名称
    g_ort->SessionGetOutputName(m->session, 0, allocator, &name);
    m->output_names = malloc(sizeof(char*));
    m->output_names[0] = strdup(name);
    allocator->Free(allocator, name);

    return 0;
}

int onnx_model_predict(ONNXModel* m, const float* in_data, const int64_t* in_shape, size_t dim, float** out_data, size_t* out_size) {
    
    OrtMemoryInfo* mem_info;
    g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info);
    
    size_t in_len = 1; 
    for(size_t i=0; i<dim; i++) in_len *= in_shape[i];
    
    OrtValue* input_tensor = NULL;
    // 使用 CreateTensorWithDataAsOrtValue 避免内部拷贝
    g_ort->CreateTensorWithDataAsOrtValue(mem_info, (void*)in_data, in_len * sizeof(float), 
                                          in_shape, dim, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 
                                          &input_tensor);
    
    OrtValue* output_tensor = NULL;
    g_ort->Run(m->session, NULL, (const char* const*)m->input_names, &input_tensor, 1, 
               (const char* const*)m->output_names, 1, &output_tensor);
    
    float* raw_out;
    g_ort->GetTensorMutableData(output_tensor, (void**)&raw_out);
    
    // 获取大小
    OrtTensorTypeAndShapeInfo* info;
    g_ort->GetTensorTypeAndShape(output_tensor, &info);
    size_t elem_cnt;
    g_ort->GetTensorShapeElementCount(info, &elem_cnt);
    *out_size = elem_cnt;
    
    *out_data = malloc(elem_cnt * sizeof(float));
    memcpy(*out_data, raw_out, elem_cnt * sizeof(float));
    g_ort->ReleaseValue(input_tensor);
    g_ort->ReleaseValue(output_tensor);

    if (info != NULL) {
        // g_ort->ReleaseTypeInfo(info);
    }

    g_ort->ReleaseMemoryInfo(mem_info);
    return 0;
}

void onnx_model_cleanup(ONNXModel* m) {
    if(m->session) g_ort->ReleaseSession(m->session);
    if(m->env) g_ort->ReleaseEnv(m->env);
}
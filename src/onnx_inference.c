#include "include/onnx_inference.h"
#include "include/image_utils.h"
#include "include/utils.h"

#include <onnxruntime_c_api.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// 全局 API 指针
static const OrtApi* ort = NULL;

// 错误检查
static void OrtCheck(OrtStatus* status, const char* message) {
    if (status != NULL) {
        const char* msg = ort->GetErrorMessage(status);
        fprintf(stderr, "%s: %s\n", message, msg);
        ort->ReleaseStatus(status);
        exit(1);
    }
}

// 初始化 API（新方式）
static void init_ort_api() {
    if (ort == NULL) {
        const OrtApiBase* base = OrtGetApiBase();
        ort = base->GetApi(ORT_API_VERSION);
    }
}

int onnx_model_init(ONNXModel* model, const char* model_path) {
    init_ort_api();
    printf("加载模型: %s\n", model_path);

    OrtStatus* status = NULL;

    // 创建 Env
    status = ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "PlateRecognition", &model->env);
    OrtCheck(status, "CreateEnv failed");

    // 创建 SessionOptions
    status = ort->CreateSessionOptions(&model->session_options);
    OrtCheck(status, "CreateSessionOptions failed");

    // 设置线程（新写法）
    ort->SetIntraOpNumThreads(model->session_options, 1);
    ort->SetInterOpNumThreads(model->session_options, 1);

    // 加载 ONNX 模型
    status = ort->CreateSession(model->env, model_path, model->session_options, &model->session);
    OrtCheck(status, "CreateSession failed");

    // 创建 CPU 内存信息
    status = ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &model->memory_info);
    OrtCheck(status, "CreateCpuMemoryInfo failed");

    // 获取输入输出数量
    ort->SessionGetInputCount(model->session, &model->input_count);
    ort->SessionGetOutputCount(model->session, &model->output_count);

    model->input_names = malloc(model->input_count * sizeof(char*));
    model->output_names = malloc(model->output_count * sizeof(char*));

    // 分配器
    OrtAllocator* allocator = NULL;
    ort->GetAllocatorWithDefaultOptions(&allocator);

    // 读取输入信息
    for (size_t i = 0; i < model->input_count; i++) {
        // 名称（已使用新 API）
        char* input_name = NULL;
        status = ort->SessionGetInputName(model->session, i, allocator, &input_name);
        OrtCheck(status, "GetInputName failed");
        model->input_names[i] = strdup(input_name);
        allocator->Free(allocator, input_name);

        // 尺寸
        OrtTypeInfo* type_info = NULL;
        status = ort->SessionGetInputTypeInfo(model->session, i, &type_info);
        OrtCheck(status, "InputTypeInfo failed");

        const OrtTensorTypeAndShapeInfo* shape;
        status = ort->CastTypeInfoToTensorInfo(type_info, &shape);
        OrtCheck(status, "CastTypeInfo failed");

        size_t dims_count = 0;
        ort->GetDimensionsCount(shape, &dims_count);

        int64_t* dims = malloc(sizeof(int64_t) * dims_count);
        ort->GetDimensions(shape, dims, dims_count);

        if (dims_count == 4) {
            model->input_height = dims[2];
            model->input_width  = dims[3];
            model->input_channels = dims[1];
        }

        free(dims);
        ort->ReleaseTypeInfo(type_info);
    }

    // 输出名称
    for (size_t i = 0; i < model->output_count; i++) {
        char* output_name = NULL;
        status = ort->SessionGetOutputName(model->session, i, allocator, &output_name);
        OrtCheck(status, "GetOutputName failed");
        model->output_names[i] = strdup(output_name);
        allocator->Free(allocator, output_name);
    }

    printf("模型加载成功: %s\n", model_path);
    printf("输入尺寸: %dx%dx%d\n",
           model->input_width, model->input_height, model->input_channels);

    return 0;
}

// -------------------------
// 简化推理：返回假数据
// -------------------------
int onnx_model_predict(ONNXModel* model, const float* input_data,
                       size_t input_size, float** output, size_t* output_size)
{
    printf("执行模型推理(简化版)...\n");

    *output_size = 1000;
    *output = malloc(*output_size * sizeof(float));
    for (size_t i = 0; i < *output_size; i++) {
        (*output)[i] = (float)i / 1000.0f;
    }
    return 0;
}

// -------------------------
// YOLO 后处理（简单）
// -------------------------
Detection* yolo_postprocess(float* output, int output_size,
                            int ow, int oh,
                            float conf, int* det_count) {
    printf("执行YOLO后处理 (简化)...\n");

    Detection* d = malloc(sizeof(Detection));
    d[0].bbox[0] = 100;
    d[0].bbox[1] = 100;
    d[0].bbox[2] = 300;
    d[0].bbox[3] = 200;
    d[0].confidence = 0.95;
    d[0].class_id = 0;
    strcpy(d[0].class_name, "vehicle");

    *det_count = 1;
    return d;
}

// 图像预处理函数，支持PP-OCR
// Image preprocess_for_ppocr_det(const unsigned char* img, int w, int h) {
//     printf("PP-OCR 检测预处理...\n");
//     return image_create(960, 960, 3);
// }
Image preprocess_for_ppocr_det(const unsigned char* image_data, int width, int height) {
    // PP-OCR检测模型预处理
    // 调整图像尺寸到模型输入尺寸，保持宽高比
    
    int target_size = 960;
    float scale = (float)target_size / MAX(width, height);
    int new_width = (int)(width * scale);
    int new_height = (int)(height * scale);
    
    // 创建源图像结构
    Image src;
    src.data = (uint8_t*)image_data;
    src.width = width;
    src.height = height;
    src.channels = 3;
    
    Image resized = image_resize(&src, new_width, new_height);
    
    // 创建目标图像（填充到固定尺寸）
    Image processed = image_create(target_size, target_size, 3);
    memset(processed.data, 0, target_size * target_size * 3); // 填充黑色
    
    // 将调整大小后的图像放到左上角
    for (int y = 0; y < new_height; y++) {
        for (int x = 0; x < new_width; x++) {
            int src_idx = (y * new_width + x) * 3;
            int dst_idx = (y * target_size + x) * 3;
            
            processed.data[dst_idx] = resized.data[src_idx];     // R
            processed.data[dst_idx + 1] = resized.data[src_idx + 1]; // G
            processed.data[dst_idx + 2] = resized.data[src_idx + 2]; // B
        }
    }
    
    image_free(&resized);
    return processed;
}

// Image preprocess_for_ppocr_rec(const Image* src) {
//     printf("PP-OCR 识别预处理...\n");
//     return image_create(src->width, 32, src->channels);
// }
Image preprocess_for_ppocr_rec(const Image* src) {
    // PP-OCR识别模型预处理
    // 调整高度到32像素，宽度按比例缩放
    
    int target_height = 32;
    int new_width = (int)((float)src->width * target_height / src->height);
    
    Image resized = image_resize(src, new_width, target_height);
    
    // 这里可以添加更多的识别预处理，如归一化等
    return resized;
}

// 新增：PP-OCR检测后处理函数
// int* ppocr_det_postprocess(float* out, int oh, int ow,
//                            int H, int W,
//                            float th, int* count) {
//     printf("PP-OCR 检测后处理...\n");
//     int* box = malloc(4 * sizeof(int));
//     box[0] = 100; box[1] = 50; box[2] = 300; box[3] = 150;
//     *count = 1;
//     return box;
// }
int* ppocr_det_postprocess(float* output, int output_height, int output_width, 
                          int original_height, int original_width, 
                          float threshold, int* box_count) {
    // 简化版的PP-OCR检测后处理
    // 实际应该实现完整的DB后处理（包括二值化、收缩、NMS等）
    
    int max_boxes = 100;
    int* boxes = malloc(max_boxes * 4 * sizeof(int));
    int count = 0;
    
    // 简单的阈值处理找到文本区域
    for (int y = 0; y < output_height && count < max_boxes; y++) {
        for (int x = 0; x < output_width && count < max_boxes; x++) {
            float score = output[y * output_width + x];
            if (score > threshold) {
                // 转换为原始图像坐标
                int x1 = (int)((float)x / output_width * original_width);
                int y1 = (int)((float)y / output_height * original_height);
                int x2 = (int)((float)(x + 1) / output_width * original_width);
                int y2 = (int)((float)(y + 1) / output_height * original_height);
                
                boxes[count * 4] = x1;
                boxes[count * 4 + 1] = y1;
                boxes[count * 4 + 2] = x2;
                boxes[count * 4 + 3] = y2;
                count++;
            }
        }
    }
    
    *box_count = count;
    return boxes;
}

// 新增：PP-OCR识别后处理函数
// PPOCRRecResult ppocr_rec_postprocess(float* out, int len, int dict) {
//     printf("PP-OCR 识别后处理...\n");
//     PPOCRRecResult r;
//     strcpy(r.text, "京A12345");
//     r.confidence = 0.85;
//     return r;
// }
PPOCRRecResult ppocr_rec_postprocess(float* output, int output_len, int dict_size) {
    PPOCRRecResult result = {"", 0.0f};
    float total_confidence = 0.0f;
    int char_count = 0;
    
    // 简化的识别后处理（实际应该实现CTC解码）
    // 假设输出形状为 [seq_len, dict_size]
    int seq_len = output_len / dict_size;
    
    for (int t = 0; t < seq_len; t++) {
        int max_index = 0;
        float max_prob = 0.0f;
        
        // 找到每个时间步概率最大的字符
        for (int i = 0; i < dict_size; i++) {
            float prob = output[t * dict_size + i];
            if (prob > max_prob) {
                max_prob = prob;
                max_index = i;
            }
        }
        
        // 忽略空白字符和重复字符（简化处理）
        if (max_index > 0 && max_prob > 0.5f) {
            if (char_count == 0 || result.text[char_count-1] != max_index) {
                // 这里应该根据字典映射索引到实际字符
                // 简化：直接使用索引作为字符
                result.text[char_count] = (char)('0' + (max_index % 10));
                total_confidence += max_prob;
                char_count++;
            }
        }
    }
    
    result.text[char_count] = '\0';
    if (char_count > 0) {
        result.confidence = total_confidence / char_count;
    }
    
    return result;
}

void onnx_model_cleanup(ONNXModel* model) {
    printf("清理ONNX资源...\n");

    if (model->session) ort->ReleaseSession(model->session);
    if (model->session_options) ort->ReleaseSessionOptions(model->session_options);
    if (model->memory_info) ort->ReleaseMemoryInfo(model->memory_info);
    if (model->env) ort->ReleaseEnv(model->env);

    if (model->input_names) {
        for (size_t i = 0; i < model->input_count; i++)
            free(model->input_names[i]);
        free(model->input_names);
    }
    if (model->output_names) {
        for (size_t i = 0; i < model->output_count; i++)
            free(model->output_names[i]);
        free(model->output_names);
    }

    printf("ONNX模型资源清理完成\n");
}

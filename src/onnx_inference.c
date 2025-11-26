#include "include/onnx_inference.h"
#include "include/image_utils.h"
#include "include/utils.h"
#include <onnxruntime_c_api.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct {
    float x1, y1, x2, y2;
    float score;
    int class_id;
} YoloBox;

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
        // 名称
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
// 真实推理
// -------------------------
// 推理函数
int onnx_model_predict(ONNXModel* model, const float* input_data, size_t input_size, float** output, size_t* output_size) {
    printf("执行模型推理...\n");

    OrtStatus* status = NULL;

    // 创建输入张量
    OrtValue* input_tensor = NULL;
    OrtMemoryInfo* memory_info = model->memory_info;
    size_t input_data_size = input_size * sizeof(float);
    printf("输入数据大小: %zu\n", input_data_size);

    // 创建张量
    status = ort->CreateTensorWithDataAsOrtValue(memory_info, (void*)input_data, input_data_size, 
                                                 model->input_names, model->input_count, 
                                                 ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);
    OrtCheck(status, "CreateTensorWithDataAsOrtValue failed");

    // 创建输出张量
    OrtValue* output_tensor = NULL;

    // 执行推理
    status = ort->Run(model->session, NULL, // 使用默认的运行选项
                      model->input_names,    // 输入张量的名称
                      &input_tensor, 1,       // 传入的输入张量及其数量
                      model->output_names,     // 输出张量的名称
                      model->output_count,     // 输出张量的数量
                      &output_tensor);         // 输出张量数组
    OrtCheck(status, "Run failed");

    // 获取输出张量的类型和形状信息
    const OrtTensorTypeAndShapeInfo* shape_info;
    status = ort->GetTensorTypeAndShape(output_tensor, &shape_info);
    OrtCheck(status, "GetTensorTypeAndShape failed");

    // 获取输出张量的维度数量
    size_t dims_count = 0;
    status = ort->GetDimensionsCount(shape_info, &dims_count);
    OrtCheck(status, "GetDimensionsCount failed");

    // 获取输出张量的总元素数
    size_t num_elements = 1;
    int64_t* dims = malloc(dims_count * sizeof(int64_t));
    status = ort->GetDimensions(shape_info, dims, dims_count);
    OrtCheck(status, "GetDimensions failed");

    for (size_t i = 0; i < dims_count; i++) {
        num_elements *= dims[i];
    }
    free(dims);

    // 设置输出大小
    *output_size = num_elements;

    // 获取输出数据（使用 OrtGetTensorData 获取张量数据）
    void* output_data = NULL;
    // void* output_data = ort->GetTensorMutableData(output_tensor，&output_data);  // 获取原始数据指针
    status = ort->GetTensorMutableData(output_tensor,&output_data);
    OrtCheck(status, "GetTensorMutableData failed");  
    if (output_data == NULL) {
        fprintf(stderr, "获取输出数据失败\n");
        return -1;
    }

    // 将输出数据拷贝到输出参数中
    *output = malloc(*output_size * sizeof(float));
    memcpy(*output, output_data, *output_size * sizeof(float));

    // 释放张量
    ort->ReleaseValue(input_tensor);
    ort->ReleaseValue(output_tensor);

    return 0;
}









// -------------------------
// YOLO
// -------------------------

int max(int a, int b) { return a > b ? a : b; }
int min(int a, int b) { return a < b ? a : b; }

// Sigmoid
static inline float sigmoid(float x) {
    return 1.f / (1.f + expf(-x));
}

// 解析 YOLOv5 输出，提取候选框（不含NMS）
int yolo_decode(const float* output, int num_anchors, int num_classes,
                float conf_thresh,
                YoloBox** out_boxes, int* out_count)
{
    int max_boxes = 3000;
    YoloBox* boxes = malloc(sizeof(YoloBox) * max_boxes);
    int count = 0;

    for (int i = 0; i < num_anchors; i++) {

        const float* ptr = output + i * (5 + num_classes);

        float obj = sigmoid(ptr[4]);
        if (obj < conf_thresh)
            continue;

        // 找最大 class prob
        int best_class = -1;
        float best_class_prob = 0;

        for (int c = 0; c < num_classes; c++) {
            float cls = sigmoid(ptr[5 + c]);
            if (cls > best_class_prob) {
                best_class_prob = cls;
                best_class = c;
            }
        }

        float score = obj * best_class_prob;
        if (score < conf_thresh)
            continue;

        float cx = ptr[0];
        float cy = ptr[1];
        float w  = ptr[2];
        float h  = ptr[3];

        float x1 = cx - w / 2.f;
        float y1 = cy - h / 2.f;
        float x2 = cx + w / 2.f;
        float y2 = cy + h / 2.f;

        boxes[count].x1 = x1;
        boxes[count].y1 = y1;
        boxes[count].x2 = x2;
        boxes[count].y2 = y2;
        boxes[count].score = score;
        boxes[count].class_id = best_class;

        count++;
        if (count >= max_boxes) break;
    }

    *out_boxes = boxes;
    *out_count = count;
    return count;
}

void yolo_map_to_original(YoloBox* boxes, int count,
                          int orig_w, int orig_h,
                          int input_w, int input_h)
{
    for (int i = 0; i < count; i++) {
        boxes[i].x1 = boxes[i].x1 / input_w * orig_w;
        boxes[i].y1 = boxes[i].y1 / input_h * orig_h;
        boxes[i].x2 = boxes[i].x2 / input_w * orig_w;
        boxes[i].y2 = boxes[i].y2 / input_h * orig_h;

        // 限制边界
        if (boxes[i].x1 < 0) boxes[i].x1 = 0;
        if (boxes[i].y1 < 0) boxes[i].y1 = 0;
        if (boxes[i].x2 > orig_w) boxes[i].x2 = orig_w;
        if (boxes[i].y2 > orig_h) boxes[i].y2 = orig_h;
    }
}

// IOU 计算
float iou(Detection a, Detection b) {
    float x1 = max(a.x1, b.x1);
    float y1 = max(a.y1, b.y1);
    float x2 = min(a.x2, b.x2);
    float y2 = min(a.y2, b.y2);

    float w = max(0, x2 - x1);
    float h = max(0, y2 - y1);
    float inter = w * h;
    float union_area = (a.x2 - a.x1)*(a.y2 - a.y1) + (b.x2 - b.x1)*(b.y2 - b.y1) - inter;
    return union_area > 0 ? inter / union_area : 0;
}

// 简化 NMS
int nms(Detection* dets, int det_count, float iou_thresh) {
    int keep_count = 0;
    for (int i = 0; i < det_count; i++) {
        int keep = 1;
        for (int j = 0; j < keep_count; j++) {
            if (iou(dets[i], dets[j]) > iou_thresh) {
                keep = 0;
                break;
            }
        }
        if (keep) {
            dets[keep_count++] = dets[i];
        }
    }
    return keep_count;
}

Detection* yolo_postprocess(float* output, int output_size,
                            int img_w, int img_h,
                            float conf_thresh, int* det_count) {
    // 假设 output 形状为 [num_boxes, 6] : [x, y, w, h, conf, class_id]
    int max_dets = 100;
    Detection* dets = malloc(max_dets * sizeof(Detection));
    int count = 0;

    int num_boxes = output_size / 6;
    for (int i = 0; i < num_boxes && count < max_dets; i++) {
        float conf = output[i*6 + 4];
        if (conf < conf_thresh) continue;

        dets[count].x1 = output[i*6 + 0] - output[i*6 + 2]/2; // cx - w/2
        dets[count].y1 = output[i*6 + 1] - output[i*6 + 3]/2; // cy - h/2
        dets[count].x2 = output[i*6 + 0] + output[i*6 + 2]/2;
        dets[count].y2 = output[i*6 + 1] + output[i*6 + 3]/2;
        dets[count].confidence = conf;
        dets[count].class_id = (int)output[i*6 + 5];
        snprintf(dets[count].class_name, sizeof(dets[count].class_name), "vehicle"); // 只有车辆
        count++;
    }

    // NMS
    int final_count = nms(dets, count, 0.5f);
    *det_count = final_count;
    return dets;
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

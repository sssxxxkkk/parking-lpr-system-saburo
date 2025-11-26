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
// 真实推理 (修复预处理逻辑：HWC->NCHW + /255.0)
// -------------------------
int onnx_model_predict(ONNXModel* model, const float* input_data, size_t input_size, float** output, size_t* output_size) {
    // 假设 model->input_width 和 height 仍然是原始的 640x480
    // 而模型实际需要的是 640x640 (model_h/w)
    
    OrtStatus* status = NULL;
    OrtMemoryInfo* memory_info = model->memory_info;

    // 1. 模型要求的固定尺寸
    int64_t model_h = 640;
    int64_t model_w = 640;
    int64_t channels = 3;
    size_t model_pixels = model_h * model_w;
    size_t model_data_bytes = channels * model_pixels * sizeof(float);

    // 2. 分配 NCHW 格式的内存 (并初始化为0/黑色)
    float* nchw_data = (float*)calloc(channels * model_pixels, sizeof(float));
    if (!nchw_data) {
        fprintf(stderr, "内存分配失败\n");
        return -1;
    }

    // 3. 【核心修复】手动进行预处理
    // 输入图像尺寸 (真实摄像头尺寸)
    int src_h = 480; 
    int src_w = 640; 
    
    // 指向 NCHW 三个通道的起始位置
    float* ptr_r = nchw_data;
    float* ptr_g = nchw_data + model_pixels;
    float* ptr_b = nchw_data + model_pixels * 2;

    // 原始数据指针 (假设输入 input_data 是 HWC 格式的 raw float，如果还没除255)
    // 注意：如果你的摄像头给的是 unsigned char (uint8)，这里 input_data 应该转成 float
    const float* src_ptr = input_data; 

    for (int y = 0; y < src_h; y++) {
        for (int x = 0; x < src_w; x++) {
            // HWC 索引: (y * width + x) * 3
            int src_idx = (y * src_w + x) * 3;
            
            // NCHW 索引: (y * model_w + x)  (注意这里用 model_w 计算偏移，实现 Padding 后的对齐)
            int dst_idx = y * model_w + x;

            // 取出 RGB (假设输入是 RGB 顺序，如果是 BGR 需要换一下)
            // float r = src_ptr[src_idx + 0];
            // float g = src_ptr[src_idx + 1];
            // float b = src_ptr[src_idx + 2];
            float b = src_ptr[src_idx + 0]; // Blue
            float g = src_ptr[src_idx + 1]; // Green
            float r = src_ptr[src_idx + 2]; // Red

            // 赋值并归一化 (0-255 -> 0.0-1.0)
            // 如果你的 input_data 已经是 0-1 的 float，就去掉 "/ 255.0f"
            ptr_r[dst_idx] = r / 255.0f;
            ptr_g[dst_idx] = g / 255.0f;
            ptr_b[dst_idx] = b / 255.0f;
        }
    }

    // 4. 构建 Dimensions
    int64_t input_node_dims[4] = {1, channels, model_h, model_w};

    // 5. 创建 Tensor
    OrtValue* input_tensor = NULL;
    status = ort->CreateTensorWithDataAsOrtValue(
                memory_info, 
                (void*)nchw_data, 
                model_data_bytes, 
                input_node_dims, 4, 
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);
    
    if (status != NULL) {
        printf("CreateTensor Failed: %s\n", ort->GetErrorMessage(status));
        free(nchw_data);
        return -1;
    }

    // 6. Run (推理)
    OrtValue* output_tensor = NULL;
    status = ort->Run(model->session, NULL, 
                      (const char* const*)model->input_names, &input_tensor, 1, 
                      (const char* const*)model->output_names, model->output_count, 
                      &output_tensor);

    // 7. 打印一下 Output Shape (用于调试下一步的后处理)
    const OrtTensorTypeAndShapeInfo* shape_info;
    ort->GetTensorTypeAndShape(output_tensor, &shape_info);
    int64_t* out_dims = NULL;
    size_t out_num_dims = 0;
    ort->GetDimensionsCount(shape_info, &out_num_dims);
    out_dims = malloc(sizeof(int64_t) * out_num_dims);
    ort->GetDimensions(shape_info, out_dims, out_num_dims);
    
    printf("DEBUG: 模型输出形状: [");
    size_t num_elements = 1;
    for(size_t i=0; i<out_num_dims; i++) {
        printf("%ld, ", out_dims[i]);
        num_elements *= out_dims[i];
    }
    printf("]\n");
    free(out_dims);

    // 8. 拷贝输出数据
    *output_size = num_elements;
    void* output_raw;
    ort->GetTensorMutableData(output_tensor, &output_raw);
    *output = malloc(num_elements * sizeof(float));
    memcpy(*output, output_raw, num_elements * sizeof(float));

    // 9. 清理
    ort->ReleaseValue(input_tensor);
    ort->ReleaseValue(output_tensor);
    free(nchw_data); // 释放刚才分配的 NCHW 数据

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

// 辅助函数：Sigmoid

// 辅助函数：IOU
static float compute_iou(float* box_a, float* box_b) {
    float x1 = fmaxf(box_a[0], box_b[0]);
    float y1 = fmaxf(box_a[1], box_b[1]);
    float x2 = fminf(box_a[2], box_b[2]);
    float y2 = fminf(box_a[3], box_b[3]);
    float inter_w = fmaxf(0.0f, x2 - x1);
    float inter_h = fmaxf(0.0f, y2 - y1);
    float inter_area = inter_w * inter_h;
    float area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]);
    float area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]);
    return inter_area / (area_a + area_b - inter_area + 1e-6f);
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

// -------------------------------------------------
// 核心修复：针对 25200x85 输出的后处理
// -------------------------------------------------
Detection* yolo_postprocess(float* output, int output_size,
                            int img_w, int img_h,
                            float conf_thresh, int* det_count) {
    
    // 1. 基础参数
    int num_anchors = 25200; // 模型固定的
    int num_classes = 80;    // 模型固定的 (85 - 5)
    int step = 5 + num_classes; // 85

    // 临时存储所有通过阈值的框 [x1, y1, x2, y2, score, class_id]
    // 预分配大一点，比如最多允许 1000 个候选
    int max_candidates = 1000;
    float* candidates = (float*)malloc(max_candidates * 6 * sizeof(float));
    int count = 0;

    // 2. 遍历所有 25200 个锚框
    for (int i = 0; i < num_anchors; i++) {
        const float* row = output + i * step; // 指向当前框的起始位置

        // 获取 Object Confidence (是否包含物体)
        // 注意：如果你之前的 conf > 1，说明模型输出的是 raw logits，必须做 sigmoid
        float obj_conf = sigmoid(row[4]); 

        // 第一层过滤：如果物体置信度太低，直接跳过 (性能优化)
        if (obj_conf < conf_thresh) continue;

        // 寻找最大概率的类别
        float max_cls_prob = 0.0f;
        int cls_id = -1;
        
        for (int c = 0; c < num_classes; c++) {
            // 同样需要 sigmoid
            float cls_prob = sigmoid(row[5 + c]);
            if (cls_prob > max_cls_prob) {
                max_cls_prob = cls_prob;
                cls_id = c;
            }
        }

        // 最终得分 = 物体置信度 * 类别概率
        float final_score = obj_conf * max_cls_prob;

        // 第二层过滤
        if (final_score < conf_thresh) continue;

        // 解析坐标 (cx, cy, w, h) -> (x1, y1, x2, y2)
        // 这里的坐标是基于 640x640 的
        float cx = row[0]; // 这里的 xywh 有些模型是 raw，有些是已经处理过的
        float cy = row[1]; // YOLOv5 ONNX export 默认通常是像素坐标，不需要 sigmoid
        float w  = row[2];
        float h  = row[3];

        // 转换为左上角/右下角坐标
        float x1 = cx - w * 0.5f;
        float y1 = cy - h * 0.5f;
        float x2 = cx + w * 0.5f;
        float y2 = cy + h * 0.5f;

        // 【关键】坐标映射回原始 640x480
        // 我们之前做的是 Top-Aligned 的 Padding，所以 (0,0) 对齐
        // 且没有缩放 (因为宽都是 640)，所以 x 和 y 不需要缩放
        // 只需要过滤掉落在 Padding 区域 (黑色区域, y > 480) 的框
        if (y1 > 480) continue; 

        // 限制边界
        if (x1 < 0) x1 = 0;
        if (y1 < 0) y1 = 0;
        if (x2 > 640) x2 = 640;
        if (y2 > 480) y2 = 480;

        if (count < max_candidates) {
            candidates[count * 6 + 0] = x1;
            candidates[count * 6 + 1] = y1;
            candidates[count * 6 + 2] = x2;
            candidates[count * 6 + 3] = y2;
            candidates[count * 6 + 4] = final_score;
            candidates[count * 6 + 5] = (float)cls_id;
            count++;
        }
    }

    // 3. 执行 NMS (非极大值抑制) 去除重叠框
    int* keep_indices = (int*)malloc(count * sizeof(int));
    int keep_count = 0;
    int* suppressed = (int*)calloc(count, sizeof(int));

    // 简单的 NMS 实现
    float nms_thresh = 0.45f;
    for (int i = 0; i < count; i++) {
        if (suppressed[i]) continue;
        
        keep_indices[keep_count++] = i;
        
        for (int j = i + 1; j < count; j++) {
            if (suppressed[j]) continue;
            
            // 检查重叠度
            float iou = compute_iou(&candidates[i*6], &candidates[j*6]);
            
            // 如果重叠严重，且分数较低的那个被抑制 (这里简化为抑制后面的)
            if (iou > nms_thresh) {
                suppressed[j] = 1;
            }
        }
    }

    // 4. 打包最终结果
    Detection* results = (Detection*)malloc(keep_count * sizeof(Detection));
    for (int k = 0; k < keep_count; k++) {
        int idx = keep_indices[k];
        float* ptr = candidates + idx * 6;
        
        results[k].x1 = ptr[0];
        results[k].y1 = ptr[1];
        results[k].x2 = ptr[2];
        results[k].y2 = ptr[3];
        results[k].confidence = ptr[4];
        results[k].class_id = (int)ptr[5];
        
        // 调试打印
        printf("检测到目标: Class=%d, Conf=%.2f, Rect=[%.0f, %.0f, %.0f, %.0f]\n", 
               results[k].class_id, results[k].confidence,
               results[k].x1, results[k].y1, results[k].x2, results[k].y2);
    }

    *det_count = keep_count;

    // 清理
    free(candidates);
    free(keep_indices);
    free(suppressed);

    return results;
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

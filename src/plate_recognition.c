#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "include/plate_recognition.h"
#include "include/onnx_inference.h"
#include "include/image_utils.h"

static ONNXModel g_net_vehicle;
static ONNXModel g_net_plate;
static ONNXModel g_net_ocr;

// --- OCR 字典相关 ---
static char** g_keys = NULL;
static int g_keys_count = 0;

// 加载字典文件
int load_ocr_keys(const char* filename) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        printf("错误: 无法打开字典文件 %s\n", filename);
        return -1;
    }

    // 1. 统计行数
    int count = 0;
    char line[64];
    while (fgets(line, sizeof(line), f)) count++;
    rewind(f);

    g_keys_count = count;
    g_keys = malloc(count * sizeof(char*));

    // 2. 读取内容
    int i = 0;
    while (fgets(line, sizeof(line), f) && i < count) {
        // 去掉换行符
        line[strcspn(line, "\r\n")] = 0;
        g_keys[i] = strdup(line);
        i++;
    }
    fclose(f);
    printf("[System] OCR字典加载完成，共 %d 个字符\n", g_keys_count);
    return 0;
}

// 释放字典
void free_ocr_keys() {
    if (g_keys) {
        for(int i=0; i<g_keys_count; i++) free(g_keys[i]);
        free(g_keys);
    }
}

static void save_plate_debug(const char* filename, unsigned char* rgb, int w, int h) {
    FILE* f = fopen(filename, "wb");
    if (!f) return;
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    fwrite(rgb, 1, w * h * 3, f);
    fclose(f);
    printf("[DEBUG] 车牌图片已保存: %s (%dx%d)\n", filename, w, h);
}

int system_init(AppConfig* config) {
    if(onnx_model_init(&g_net_vehicle, config->vehicle_model) != 0) return -1;
    if(onnx_model_init(&g_net_plate, config->plate_model) != 0) return -1;
    if(onnx_model_init(&g_net_ocr, config->ocr_model) != 0) return -1;
    if(load_ocr_keys("models/ppocr_keys_v1.txt") != 0) return -1;
    return 0;
}

void system_cleanup() {
    onnx_model_cleanup(&g_net_vehicle);
    onnx_model_cleanup(&g_net_plate);
    onnx_model_cleanup(&g_net_ocr);
    free_ocr_keys();
}


// ---------------------------------------------------------
// 真正的 CTC 解码函数
// data: 模型输出的概率矩阵 [1, 40, 6625] (Batch, Seq, Class)
// seq_len: 序列长度 (通常是 40 或 48，取决于模型输出维度)
// num_classes: 类别总数 (6625)
// ---------------------------------------------------------
// void decode_ocr_real(float* data, int seq_len, int num_classes, char* buffer) {
//     buffer[0] = '\0';
//     int last_index = -1; 

//     for (int t = 0; t < seq_len; t++) {
//         float max_score = -10000.0f;
//         int max_idx = 0;
        
//         // 指针偏移 (这里 num_classes 必须正确，否则全乱)
//         float* current_step_data = data + t * num_classes;

//         for (int c = 0; c < num_classes; c++) {
//             if (current_step_data[c] > max_score) {
//                 max_score = current_step_data[c];
//                 max_idx = c;
//             }
//         }

//         // 置信度阈值 (可选)：防止输出太差的字
//         if (max_score < 0.5f) continue;

//         if (max_idx != 0 && max_idx != last_index) {
//             int dict_idx = max_idx - 1;
            
//             // 【关键修复】增加越界检查！
//             // 只有当模型识别出的索引在你的字典范围内时，才转换
//             if (dict_idx >= 0 && dict_idx < g_keys_count) {
//                 strcat(buffer, g_keys[dict_idx]);
//             } else {
//                 // 如果字典不全，模型识别出了字典外的字，用 ? 代替
//                 // strcat(buffer, "?"); 
//             }
//         }
//         last_index = max_idx;
//     }
// }
// 修改 src/plate_recognition.c

// 修改 src/plate_recognition.c

void decode_ocr_real(float* data, int seq_len, int num_classes, char* buffer) {
    buffer[0] = '\0';
    int last_index = -1; 
    
    // 调试缓冲区
    char debug_buf[256] = {0};
    int has_valid_char = 0;

    for (int t = 0; t < seq_len; t++) {
        float max_score = -10000.0f;
        int max_idx = 0;
        
        float* current_step_data = data + t * num_classes;
        for (int c = 0; c < num_classes; c++) {
            if (current_step_data[c] > max_score) {
                max_score = current_step_data[c];
                max_idx = c;
            }
        }

        // CTC 去重逻辑
        if (max_idx != 0 && max_idx != last_index) {
            int dict_idx = max_idx - 1;
            
            // 调试打印：记录下看到了什么索引
            char tmp[32];
            snprintf(tmp, 32, "%d(%.2f) ", dict_idx, max_score);
            strcat(debug_buf, tmp);
            has_valid_char = 1;

            if (dict_idx >= 0 && dict_idx < g_keys_count) {
                strcat(buffer, g_keys[dict_idx]);
            } else {
                strcat(buffer, "?"); 
            }
        }
        last_index = max_idx;
    }
    
    if (has_valid_char) {
        printf("[OCR DEBUG] 原始索引序列: %s\n", debug_buf);
    } else {
        printf("[OCR DEBUG] 模型认为全是空白 (Blank)\n");
    }
}

// // 模拟 OCR 解码
// void decode_ocr_dummy(float* ocr_out, int seq_len, int dict_size, char* buffer) {
//     strcpy(buffer, "苏A88888"); 
// }

DetectionResult* process_frame(unsigned char* img_data, int w, int h, int* count) {
    *count = 0;
    if(!img_data) return NULL;
    
    DetectionResult* results = calloc(5, sizeof(DetectionResult));
    
    // -----------------------------------------------------------
    // Step 1: 车辆检测 (YOLO)
    // -----------------------------------------------------------
    float* v_in = malloc(1*3*640*640*sizeof(float));
    
    // 注意：preprocess_yolo 必须是保持比例的 resize (Letterbox)
    // 此时 scale = min(640/w, 640/h)
    preprocess_yolo(img_data, w, h, 640, v_in);
    
    int64_t v_shape[] = {1,3,640,640};
    float* v_out = NULL; 
    size_t v_len = 0;

    if(onnx_model_predict(&g_net_vehicle, v_in, v_shape, 4, &v_out, &v_len) == 0) {
        Detection cars[100]; 
        int car_cnt = 0;
        
        // 后处理：置信度先放低一点，防止漏检
        postprocess_yolo(v_out, v_len/85, 0.25f, w, h, cars, &car_cnt); 
        
        // NMS 去重
        nms_yolo(cars, &car_cnt, 0.45f);

        // 遍历每一辆车
        for(int i=0; i<car_cnt && *count < 5; i++) {
            // YOLO 原始坐标
            int raw_cx = (int)cars[i].x1;
            int raw_cy = (int)cars[i].y1;
            int raw_cw = (int)(cars[i].x2 - cars[i].x1);
            int raw_ch = (int)(cars[i].y2 - cars[i].y1);

            // 过滤过小的误检
            if(raw_cw < 50 || raw_ch < 50) continue;

            // ========================================================
            // 【核心修复 1】: 车辆框扩张 (ROI Expansion)
            // 目的是把车牌（可能在车框边缘）给包进来
            // ========================================================
            int pad_w = (int)(raw_cw * 0.25f); // 宽度左右各扩 15%
            int pad_h = (int)(raw_ch * 0.25f); // 高度上下各扩 15%

            int cx = raw_cx - pad_w;
            int cy = raw_cy - pad_h;
            int cw = raw_cw + 2 * pad_w;
            int ch = raw_ch + 2 * pad_h;

            // 边界检查 (非常重要，否则抠图会崩)
            if (cx < 0) cx = 0;
            if (cy < 0) cy = 0;
            if (cx + cw > w) cw = w - cx;
            if (cy + ch > h) ch = h - cy;
            
            // 打印修正后的车辆坐标，用于调试
            printf("[DEBUG] 车辆 #%d 修正坐标: x=%d y=%d w=%d h=%d\n", i, cx, cy, cw, ch);

            // ========================================================
            // Step 2: 车辆抠图 & 车牌定位 (DBNet)
            // ========================================================
            unsigned char* car_img = malloc(cw * ch * 3);
            crop_image_rgb(img_data, w, h, cx, cy, cw, ch, car_img);

            // 保存这张图！看看是不是包含了完整的车牌
            char debug_name[64];
            snprintf(debug_name, 64, "debug_car_%d.ppm", i);
            save_plate_debug(debug_name, car_img, cw, ch);

            // 车牌定位输入尺寸 (建议设为 640 以提高小目标检出率)
            int det_size = 640; 
            float* p_in = malloc(1*3*det_size*det_size*sizeof(float));
            preprocess_dbnet(car_img, cw, ch, det_size, p_in);
            
            int64_t p_shape[] = {1,3,det_size,det_size};
            float* p_out = NULL; size_t p_len = 0;
            
            if(onnx_model_predict(&g_net_plate, p_in, p_shape, 4, &p_out, &p_len) == 0) {
                // 2.1 从热力图中找车牌框
                int px, py, pw, ph;
                // 注意：这里是在“车辆小图”里找车牌
                postprocess_dbnet(p_out, det_size, det_size, 0.3f, &px, &py, &pw, &ph);
                
                if(pw > 0 && ph > 0) {
                    // 2.2 坐标映射: 小图 -> 大图
                    float scale = fminf((float)det_size/cw, (float)det_size/ch);
                    
                    // 【注意】这里的 cx, cy 必须是上面【扩张后】的车辆左上角
                    int gx = cx + (int)(px / scale);
                    int gy = cy + (int)(py / scale);
                    int gw = (int)(pw / scale);
                    int gh = (int)(ph / scale);
                    
                    // ====================================================
                    // 【核心修复 2】: 车牌框二次扩张
                    // DBNet 找出的框是收缩的(Shrunk)，必须放大才能包含完整文字
                    // ====================================================
                    float plate_expand_w = 1.8f; // 宽扩 1.2 倍
                    float plate_expand_h = 2.0f; // 高扩 1.5 倍

                    int center_x = gx + gw/2;
                    int center_y = gy + gh/2;
                    int new_gw = (int)(gw * plate_expand_w);
                    int new_gh = (int)(gh * plate_expand_h);
                    
                    gx = center_x - new_gw/2;
                    gy = center_y - new_gh/2;
                    gw = new_gw;
                    gh = new_gh;

                    // 边界检查
                    if(gx < 0) gx = 0;
                    if(gy < 0) gy = 0;
                    if(gx + gw > w) gw = w - gx;
                    if(gy + gh > h) gh = h - gy;

                    // ====================================================

                    // 防欺诈逻辑
                    if (gw < cw * 0.9) {
                        // --- Step 3: 车牌识别 (OCR Rec) ---
                        unsigned char* plate_img = malloc(gw * gh * 3);
                        crop_image_rgb(img_data, w, h, gx, gy, gw, gh, plate_img);
                        
                        // 保存最终车牌图，用于确认
                        snprintf(debug_name, 64, "debug_plate_%d.ppm", i);
                        save_plate_debug(debug_name, plate_img, gw, gh);

                        float* ocr_in = malloc(1*3*48*320*sizeof(float));
                        preprocess_ocr(plate_img, gw, gh, ocr_in);
                        
                        int64_t ocr_shape[] = {1,3,48,320};
                        float* ocr_out = NULL; size_t ocr_len = 0;
                        
                        if(onnx_model_predict(&g_net_ocr, ocr_in, ocr_shape, 4, &ocr_out, &ocr_len) == 0) {
                            results[*count].confidence = cars[i].confidence;
                            results[*count].vehicle_bbox[0] = cx;
                            results[*count].vehicle_bbox[1] = cy;
                            results[*count].vehicle_bbox[2] = cw;
                            results[*count].vehicle_bbox[3] = ch;
                            results[*count].is_fraud = 0;
                            
                            // 3.3 真实解码
                            int model_num_classes = 6625; 
                            if (ocr_len % 6625 == 0) model_num_classes = 6625;
                            else if (ocr_len % 97 == 0) model_num_classes = 97;
                            
                            int seq_len = ocr_len / model_num_classes;
                            
                            decode_ocr_real(ocr_out, seq_len, model_num_classes, results[*count].plate_text);
                            
                            if (strlen(results[*count].plate_text) == 0) {
                                strcpy(results[*count].plate_text, "无法识别");
                            }

                            (*count)++;
                            free(ocr_out);
                        }
                        
                        free(ocr_in);
                        free(plate_img);
                    }
                }
                free(p_out);
            }
            free(p_in);
            free(car_img);
        }
        free(v_out);
    }
    free(v_in);
    return results;
}

// DetectionResult* process_frame(unsigned char* img_data, int w, int h, int* count) {
//     *count = 0;
//     if(!img_data) return NULL;
    
//     DetectionResult* results = calloc(5, sizeof(DetectionResult));
    
//     // --- Step 1: 车辆检测 (YOLO) ---
//     float* v_in = malloc(1*3*640*640*sizeof(float));
//     preprocess_yolo(img_data, w, h, 640, v_in);
    
//     int64_t v_shape[] = {1,3,640,640};
//     float* v_out = NULL; 
//     size_t v_len = 0;

//     if(onnx_model_predict(&g_net_vehicle, v_in, v_shape, 4, &v_out, &v_len) == 0) {
//         Detection cars[20]; 
//         int car_cnt = 0;
//         postprocess_yolo(v_out, 25200, 0.5f, w, h, cars, &car_cnt); // 0.5 置信度
        
//         //NMS 去重。阈值0.45，表示如果两个框重叠面积超过 45%，就认为是同一个车
//         nms_yolo(cars, &car_cnt, 0.45f);

//         // 只有检测到车，才进行下一步
//         for(int i=0; i<car_cnt && *count < 5; i++) {
//             int cx = (int)cars[i].x1;
//             int cy = (int)cars[i].y1;
//             int cw = (int)(cars[i].x2 - cars[i].x1);
//             int ch = (int)(cars[i].y2 - cars[i].y1);
            

//             // 过滤过小的车
//             if(cw < 50 || ch < 50) continue;

//             // --- Step 2: 车辆抠图 & 车牌定位 (PP-OCR Det) ---
//             unsigned char* car_img = malloc(cw * ch * 3);
//             crop_image_rgb(img_data, w, h, cx, cy, cw, ch, car_img);

//             save_plate_debug("debug_plate.ppm", car_img, cw, ch);

//             int det_size = 640; // Det 模型输入通常是 32 倍数
//             float* p_in = malloc(1*3*det_size*det_size*sizeof(float));
//             preprocess_dbnet(car_img, cw, ch, det_size, p_in);
            
//             int64_t p_shape[] = {1,3,det_size,det_size};
//             float* p_out = NULL; size_t p_len = 0;
            
//             if(onnx_model_predict(&g_net_plate, p_in, p_shape, 4, &p_out, &p_len) == 0) {
//                 // 后处理找框
//                 int px, py, pw, ph;
//                 postprocess_dbnet(p_out, det_size, det_size, 0.3f, &px, &py, &pw, &ph);
                
//                 if(pw > 0 && ph > 0) {
//                     // 坐标映射： Det图 -> 车辆图 -> 原图
//                     float scale = fminf((float)det_size/cw, (float)det_size/ch);
//                     int gx = cx + (int)(px / scale);
//                     int gy = cy + (int)(py / scale);
//                     int gw = (int)(pw / scale);
//                     int gh = (int)(ph / scale);
                    
//                     // 防欺诈检测：车牌是否过大(比如占了车辆宽度的90%以上，可能是人拿牌子)
//                     if (gw < cw * 0.9) {
//                         // --- Step 3: 车牌识别 (OCR Rec) ---
//                         unsigned char* plate_img = malloc(gw * gh * 3);
//                         crop_image_rgb(img_data, w, h, gx, gy, gw, gh, plate_img);

//                         float* ocr_in = malloc(1*3*48*320*sizeof(float));
//                         preprocess_ocr(plate_img, gw, gh, ocr_in);
                        
//                         int64_t ocr_shape[] = {1,3,48,320};
//                         float* ocr_out = NULL; size_t ocr_len = 0;
                        
//                         if(onnx_model_predict(&g_net_ocr, ocr_in, ocr_shape, 4, &ocr_out, &ocr_len) == 0) {
//                             results[*count].confidence = cars[i].confidence;
//                             results[*count].vehicle_bbox[0] = cx;
//                             results[*count].vehicle_bbox[1] = cy;
//                             results[*count].vehicle_bbox[2] = cw;
//                             results[*count].vehicle_bbox[3] = ch;
//                             results[*count].is_fraud = 0;
                            
//                             // 1. 确定模型的真实类别数 (Model Class Size)
//                             // 绝大多数 PP-OCR 中文模型(v2/v3/v4) 都是 6625 类
//                             int model_num_classes = 6625; 
                            
//                             // 简单的自动推断：如果总长度能被 6625 整除，那就是 6625
//                             if (ocr_len % 6625 == 0) {
//                                 model_num_classes = 6625;
//                             } 
//                             // 如果是英文数字模型，可能是 97
//                             else if (ocr_len % 97 == 0) {
//                                 model_num_classes = 97;
//                             }
//                             else {
//                                 // 兜底：如果都不是，可能计算有误，打印警告
//                                 printf("[Warn] OCR输出尺寸异常: %ld\n", ocr_len);
//                             }

//                             // 2. 计算序列长度 (Sequence Length)
//                             int seq_len = ocr_len / model_num_classes;

//                             // 3. 解码
//                             // 注意：这里传给解码器的必须是 model_num_classes (用于指针步长)
//                             // 而不是 g_keys_count (那个只用于查表)
//                             decode_ocr_real(ocr_out, seq_len, model_num_classes, results[*count].plate_text);
                            
//                             // 如果解码为空，说明没识别出来
//                             if (strlen(results[*count].plate_text) == 0) {
//                                 strcpy(results[*count].plate_text, "未知");
//                             }

//                             (*count)++;
//                             free(ocr_out);
//                         }
                        
//                         free(ocr_in);
//                         free(plate_img);
//                     } else {
//                         printf("警告：疑似欺诈！车牌尺寸异常。\n");
//                     }
//                 }
//                 free(p_out);
//             }
//             free(p_in);
//             free(car_img);
//         }
//         free(v_out);
//     }
//     free(v_in);
//     return results;
// }
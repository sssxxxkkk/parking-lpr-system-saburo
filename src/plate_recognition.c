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

void clean_plate_text(char* text) {
    if (!text) return;
    
    char* src = text; // 读取指针
    char* dst = text; // 写入指针
    
    while (*src) {
        unsigned char c = (unsigned char)*src;
        
        // 1. 处理 UTF-8 的中间点 '·' (0xC2 0xB7)
        if (c == 0xC2 && (unsigned char)*(src+1) == 0xB7) {
            src += 2; // 跳过这2个字节
            continue;
        }
        
        // 2. 处理 ASCII 的分隔符 (点、横杠、空格)
        if (c == '.' || c == '-' || c == ' ') {
            src++; // 跳过1个字节
            continue;
        }
        
        // 3. 保留有效字符 (数字、字母、汉字)
        *dst = *src;
        dst++;
        src++;
    }
    
    *dst = '\0';
}

// 定义省份首字
const char* VALID_PROVINCES[] = {
    "京","沪","津","渝","冀","晋","蒙","辽","吉","黑","苏",
    "浙","皖","闽","赣","鲁","豫","鄂","湘","粤","桂","琼",
    "川","贵","云","藏","陕","甘","青","宁","新", 
    "港", "澳", "使", "领", "学", "警"
};

// 判断是否以合法的省份开头
int is_valid_province(const char* txt) {
    char first_char[4] = {0};
    if (strlen(txt) < 3) return 0;
    
    memcpy(first_char, txt, 3);
    
    int num_provinces = sizeof(VALID_PROVINCES) / sizeof(VALID_PROVINCES[0]);
    for(int i=0; i<num_provinces; i++) {
        if (memcmp(first_char, VALID_PROVINCES[i], 3) == 0) {
            return 1;
        }
    }
    return 0;
}

// 检查字符是否为 数字 或 大写字母
int is_valid_alphanum(char c) {
    if (c >= '0' && c <= '9') return 1;
    if (c >= 'A' && c <= 'Z') return 1;
    return 0;
}

// 序号位不使用字母 O 和 I
void optimize_char_confusion(char* text) {
    if (!text || strlen(text) < 7) return;

    // 1. 修正发牌机关 (第2位字符，即 index 3)
    if (text[3] == '0') {
        text[3] = 'O';
    }

    int i = 4;
    while (text[i] != '\0') {
        // 将 O 修正为 0
        if (text[i] == 'O') {
            text[i] = '0';
        }
        // 将 I 修正为 1
        else if (text[i] == 'I') {
            text[i] = '1';
        }

        i++;
    }
}

// 识别符合车牌规则
int fix_and_validate_plate(char* plate_text) {
    if (!plate_text || strlen(plate_text) < 7) return 0;

    // 1. 检查首字是否为省份
    if (!is_valid_province(plate_text)) {
        return 0;
    }

    // 2. 检查后续字符
    int i = 3; 
    char clean_buf[32] = {0};
    
    memcpy(clean_buf, plate_text, 3);
    int clean_idx = 3;

    while (plate_text[i] != '\0') {
        char c = plate_text[i];
        
        if (is_valid_alphanum(c)) {
            clean_buf[clean_idx++] = c;
        }
        else if ((unsigned char)c > 127) {

        }

        if ((unsigned char)c > 127) i += 3;
        else i++;
    }
    clean_buf[clean_idx] = '\0';
    
    // 3. 长度校验
    int final_len = strlen(clean_buf);
    if (final_len < 7 || final_len > 9) return 0;
    
    // 覆盖回原字符串
    strcpy(plate_text, clean_buf);
    return 1;
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
    
    // if (has_valid_char) {
    //     printf("[OCR DEBUG] 原始索引序列: %s\n", debug_buf);
    // } else {
    //     printf("[OCR DEBUG] 模型认为全是空白 (Blank)\n");
    // }
}

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
            // printf("[DEBUG] 车辆 #%d 修正坐标: x=%d y=%d w=%d h=%d\n", i, cx, cy, cw, ch);

            // ========================================================
            // Step 2: 车辆抠图 & 车牌定位 (DBNet)
            // ========================================================
            unsigned char* car_img = malloc(cw * ch * 3);
            crop_image_rgb(img_data, w, h, cx, cy, cw, ch, car_img);

            // 保存图 完整车牌
            // char debug_name[64];
            // snprintf(debug_name, 64, "debug_car_%d.ppm", i);
            // save_plate_debug(debug_name, car_img, cw, ch);

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
                        // snprintf(debug_name, 64, "debug_plate_%d.ppm", i);
                        // save_plate_debug(debug_name, plate_img, gw, gh);

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

                            // 去除点号
                            clean_plate_text(results[*count].plate_text);

                            // 混淆修正
                            optimize_char_confusion(results[*count].plate_text);

                            // 强规则校验和清洗
                            int is_valid = fix_and_validate_plate(results[*count].plate_text);
                            
                            if (is_valid) {
                                (*count)++;
                            } else { 

                            }
                            
                            if (strlen(results[*count].plate_text) == 0) {
                                strcpy(results[*count].plate_text, "无法识别");
                            }

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
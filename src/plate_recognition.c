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

int system_init(AppConfig* config) {
    if(onnx_model_init(&g_net_vehicle, config->vehicle_model) != 0) return -1;
    if(onnx_model_init(&g_net_plate, config->plate_model) != 0) return -1;
    if(onnx_model_init(&g_net_ocr, config->ocr_model) != 0) return -1;
    return 0;
}

void system_cleanup() {
    onnx_model_cleanup(&g_net_vehicle);
    onnx_model_cleanup(&g_net_plate);
    onnx_model_cleanup(&g_net_ocr);
}

// 模拟 OCR 解码 (实际需要查几千字的字典)
void decode_ocr_dummy(float* ocr_out, int seq_len, int dict_size, char* buffer) {
    // 这里简单地写死，实际需要 ArgMax + CTC Decode + 查字典
    strcpy(buffer, "苏A88888"); 
}

DetectionResult* process_frame(unsigned char* img_data, int w, int h, int* count) {
    *count = 0;
    if(!img_data) return NULL;
    
    DetectionResult* results = calloc(5, sizeof(DetectionResult));
    
    // --- Step 1: 车辆检测 (YOLO) ---
    float* v_in = malloc(1*3*640*640*sizeof(float));
    preprocess_yolo(img_data, w, h, 640, v_in);
    
    int64_t v_shape[] = {1,3,640,640};
    float* v_out = NULL; 
    size_t v_len = 0;
 
    // if(onnx_model_predict(&g_net_vehicle, v_in, v_shape, 4, &v_out, &v_len) == 0) {
        
    //     // ----------------- DEBUG 修改开始 -----------------
    //     printf("\n[DEBUG] ----- 开始分析 YOLO 输出 (Total size: %ld) -----\n", v_len);
        
    //     int rows = 25200; // YOLOv5s 640x640 的标准输出行数
    //     int dimensions = 85; // 坐标(4) + 置信度(1) + 类别(80)
        
    //     int detected_count = 0;
        
    //     for(int i = 0; i < rows; i++) {
    //         // 获取当前锚框的数据指针
    //         float* row = v_out + i * dimensions;
            
    //         // 原始置信度 (Object Confidence)
    //         // 注意：如果你的 postprocess 内部做了 sigmoid，这里也要做。
    //         // 通常 ONNX 导出的 YOLO 输出已经是 sigmoid 过的，或者需要手动做。
    //         // 这里我们假设需要手动做 sigmoid (1 / (1 + exp(-x)))

    //         // float raw_conf = row[4];
    //         // float conf = 1.0f / (1.0f + expf(-raw_conf));
    //         float conf = row[4]; 

    //         // 我们只看置信度 > 0.1 的，防止刷屏，但要足够低以发现问题
    //         if (conf > 0.1f) {
    //             // 找概率最大的类别
    //             int max_class_id = 0;
    //             float max_class_score = 0.0f;
                
    //             // 遍历 80 个类别 (从 row[5] 开始)
    //             for (int c = 0; c < 80; c++) {
    //                 // float class_raw = row[5 + c];
    //                 // float class_score = 1.0f / (1.0f + expf(-class_raw));
    //                 float class_score = row[5 + c]; 

    //                 if (class_score > max_class_score) {
    //                     max_class_score = class_score;
    //                     max_class_id = c;
    //                 }
    //             }
                
    //             // 最终得分 = obj_conf * class_conf
    //             float final_score = conf * max_class_score;

    //             // 打印任何稍微可信的目标
    //             if (final_score > 0.1f && (max_class_id == 2 || max_class_id == 5 || max_class_id == 7)) {
    //                 printf("  >> [目标 %d] Class: %d | Conf: %.4f (Obj: %.2f, Cls: %.2f)\n", 
    //                        detected_count++, max_class_id, final_score, conf, max_class_score);
                           
    //                 // COCO 数据集 ID 参考:
    //                 // 0: Person (人)
    //                 // 2: Car (轿车)
    //                 // 5: Bus (公交车)
    //                 // 7: Truck (卡车)
    //             }
    //         }
    //     }
        
    //     if (detected_count == 0) {
    //         printf("[DEBUG] YOLO 没看到任何东西 (Conf > 0.1)。请检查图片预处理。\n");
    //     }
    //     printf("[DEBUG] ----------------------------------------------\n");
    //     // ----------------- DEBUG 修改结束 -----------------
    // }


    if(onnx_model_predict(&g_net_vehicle, v_in, v_shape, 4, &v_out, &v_len) == 0) {
        Detection cars[10]; 
        int car_cnt = 0;
        postprocess_yolo(v_out, 25200, 0.5f, w, h, cars, &car_cnt); // 0.5 置信度
        
        // 只有检测到车，才进行下一步
        for(int i=0; i<car_cnt && *count < 5; i++) {
            int cx = (int)cars[i].x1;
            int cy = (int)cars[i].y1;
            int cw = (int)(cars[i].x2 - cars[i].x1);
            int ch = (int)(cars[i].y2 - cars[i].y1);
            

            // 过滤过小的车
            if(cw < 50 || ch < 50) continue;

            // --- Step 2: 车辆抠图 & 车牌定位 (PP-OCR Det) ---
            unsigned char* car_img = malloc(cw * ch * 3);
            crop_image_rgb(img_data, w, h, cx, cy, cw, ch, car_img);

            int det_size = 320; // Det 模型输入通常是 32 倍数
            float* p_in = malloc(1*3*det_size*det_size*sizeof(float));
            preprocess_dbnet(car_img, cw, ch, det_size, p_in);
            
            int64_t p_shape[] = {1,3,det_size,det_size};
            float* p_out = NULL; size_t p_len = 0;
            
            if(onnx_model_predict(&g_net_plate, p_in, p_shape, 4, &p_out, &p_len) == 0) {
                // 后处理找框
                int px, py, pw, ph;
                postprocess_dbnet(p_out, det_size, det_size, 0.3f, &px, &py, &pw, &ph);
                
                if(pw > 0 && ph > 0) {
                    // 坐标映射： Det图 -> 车辆图 -> 原图
                    float scale = fminf((float)det_size/cw, (float)det_size/ch);
                    int gx = cx + (int)(px / scale);
                    int gy = cy + (int)(py / scale);
                    int gw = (int)(pw / scale);
                    int gh = (int)(ph / scale);
                    
                    // 防欺诈检测：车牌是否过大？(比如占了车辆宽度的90%以上，可能是人拿牌子)
                    if (gw < cw * 0.9) {
                        // --- Step 3: 车牌识别 (OCR Rec) ---
                        unsigned char* plate_img = malloc(gw * gh * 3);
                        crop_image_rgb(img_data, w, h, gx, gy, gw, gh, plate_img);
                        
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
                            
                            decode_ocr_dummy(ocr_out, 0, 0, results[*count].plate_text);
                            (*count)++;
                            free(ocr_out);
                        }
                        
                        free(ocr_in);
                        free(plate_img);
                    } else {
                        printf("警告：疑似欺诈！车牌尺寸异常。\n");
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
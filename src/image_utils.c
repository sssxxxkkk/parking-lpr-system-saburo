//图像预处理 后处理
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "include/image_utils.h"

// static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

void crop_image_rgb(const unsigned char* src, int sw, int sh, int x, int y, int w, int h, unsigned char* dst) {
    if (x < 0) x = 0; if (y < 0) y = 0;
    if (x + w > sw) w = sw - x; if (y + h > sh) h = sh - y;
    if (w <= 0 || h <= 0) return;

    for(int i=0; i<h; i++) {
        memcpy(dst + i*w*3, src + ((y+i)*sw + x)*3, w*3);
    }
}

void preprocess_yolo(const unsigned char* src, int w, int h, int target, float* dst) {
    float scale = fminf((float)target/w, (float)target/h);
    int nw = (int)(w * scale);
    int nh = (int)(h * scale);
    memset(dst, 0, 3 * target * target * sizeof(float)); // Padding 0

    for(int r=0; r<nh; r++) {
        for(int c=0; c<nw; c++) {
            int sx = (int)(c / scale);
            int sy = (int)(r / scale);
            if(sx >= w) sx = w-1; if(sy >= h) sy = h-1;
            int idx = (sy * w + sx) * 3;
            // NCHW, Normalize 0-1
            dst[0*target*target + r*target + c] = src[idx+0] / 255.0f;
            dst[1*target*target + r*target + c] = src[idx+1] / 255.0f;
            dst[2*target*target + r*target + c] = src[idx+2] / 255.0f;
        }
    }
}

// 简化版 DBNet 预处理 (与 YOLO 类似，但通常需要 Normalize mean=[0.485...] std=[0.229...])
// 这里为了简化，使用 1/255.0f - 0.5f
void preprocess_dbnet(const unsigned char* src, int w, int h, int target, float* dst) {
    float scale = fminf((float)target/w, (float)target/h);
    int nw = (int)(w * scale);
    int nh = (int)(h * scale);
    memset(dst, 0, 3 * target * target * sizeof(float));

    for(int r=0; r<nh; r++) {
        for(int c=0; c<nw; c++) {
            int sx = (int)(c / scale);
            int sy = (int)(r / scale);
            if(sx >= w) sx = w-1; if(sy >= h) sy = h-1;
            int idx = (sy * w + sx) * 3;
            
            // PP-OCR Det Mean/Std: (Pixel/255 - 0.485) / 0.229
            dst[0*target*target + r*target + c] = (src[idx+0]/255.0f - 0.485f) / 0.229f;
            dst[1*target*target + r*target + c] = (src[idx+1]/255.0f - 0.456f) / 0.224f;
            dst[2*target*target + r*target + c] = (src[idx+2]/255.0f - 0.406f) / 0.225f;
        }
    }
}

// void postprocess_yolo(float* data, int rows, float conf_thres, int w, int h, Detection* dets, int* count) {
//     *count = 0;
//     float scale = fminf(640.0f/w, 640.0f/h);
    
//     // YOLOv5 Output: [1, 25200, 85] (cx, cy, w, h, obj_conf, cls_conf...)
//     for(int i=0; i<rows; i++) {
//         float* row = data + i * 85;
//         // float obj = sigmoid(row[4]); // Obj Confidence
//         float obj = row[4]; // Obj Confidence
//         if (obj > conf_thres) {
//             float cx = row[0];
//             float cy = row[1];
//             float bw = row[2];
//             float bh = row[3];
            
//             dets[*count].x1 = (cx - bw/2) / scale;
//             dets[*count].y1 = (cy - bh/2) / scale;
//             dets[*count].x2 = (cx + bw/2) / scale;
//             dets[*count].y2 = (cy + bh/2) / scale;
//             dets[*count].confidence = obj;
//             (*count)++;
//             if(*count >= 10) break;
//         }
//     }
// }
void postprocess_yolo(float* data, int rows, float conf_thres, int w, int h, Detection* dets, int* count) {
    *count = 0;
    // 计算缩放比例，将 640x640 的坐标还原回 实际分辨率(例如 640x480)
    float scale = fminf(640.0f/w, 640.0f/h);
    
    // 偏移量计算 (用于居中填充的情况)
    // 如果你的 preprocess 只是 resize 没有 padding，这里可以是 0
    // 如果是标准的 letterbox padding，需要减去偏移
    // 简化起见，假设是左上角对齐或拉伸，暂不处理复杂 padding
    
    // YOLOv5 Output: [1, 25200, 85] 
    // 0-3: box, 4: obj_conf, 5-84: class_conf
    for(int i=0; i<rows; i++) {
        float* row = data + i * 85;
        
        // 1. 获取物体置信度 (已有 sigmoid，直接读)
        float obj_conf = row[4]; 
        
        // 只有物体置信度足够高，才去算类别，节省 CPU
        if (obj_conf > conf_thres) {
            
            // 2. 找出概率最大的类别
            float max_cls_conf = 0.0f;
            int cls_id = -1;
            
            // 遍历 80 个类别 (从 index 5 开始)
            for(int c=0; c<80; c++) {
                float current_cls_conf = row[5 + c];
                if(current_cls_conf > max_cls_conf) {
                    max_cls_conf = current_cls_conf;
                    cls_id = c;
                }
            }
            
            // 3. 计算最终得分 = 物体概率 * 类别概率
            float final_score = obj_conf * max_cls_conf;
            
            // 4. 再次过滤最终得分 + 类别筛选
            // COCO ID: 2=Car, 5=Bus, 7=Truck
            if (final_score > conf_thres && (cls_id == 2 || cls_id == 5 || cls_id == 7)) {
                
                float cx = row[0];
                float cy = row[1];
                float bw = row[2];
                float bh = row[3];
                
                // 还原坐标到原图尺寸
                // 注意：这里假设 preprocess 是 keep aspect ratio 的 resize
                // 如果坐标偏了，后面再微调
                dets[*count].x1 = (cx - bw/2) / scale;
                dets[*count].y1 = (cy - bh/2) / scale;
                dets[*count].x2 = (cx + bw/2) / scale;
                dets[*count].y2 = (cy + bh/2) / scale;
                
                dets[*count].confidence = final_score;
                dets[*count].class_id = cls_id; 
                
                (*count)++;
                if(*count >= 20) break; // 最多保留 20 个目标
            }
        }
    }
}

void postprocess_dbnet(float* map, int mw, int mh, float thresh, int* x, int* y, int* w, int* h) {
    // 简易热力图寻找包围盒
    int min_x = mw, min_y = mh, max_x = 0, max_y = 0;
    int cnt = 0;
    
    // map 是 sigmoid 后的概率图? 如果模型没自带 sigmoid，需要这里做
    // 假设模型输出已经经过 sigmoid 或者 raw output
    // 这里简单假设 raw output，需要 sigmoid
    for(int i=0; i<mh*mw; i++) {
        float val = (map[i]); // 如果模型是 raw logits，用 sigmoid(map[i])
        if (val > thresh) {
            int r = i / mw;
            int c = i % mw;
            if(c < min_x) min_x = c;
            if(c > max_x) max_x = c;
            if(r < min_y) min_y = r;
            if(r > max_y) max_y = r;
            cnt++;
        }
    }
    
    if (cnt < 20) { *w=0; *h=0; return; }
    
    *x = min_x; *y = min_y;
    *w = max_x - min_x;
    *h = max_y - min_y;
}

void preprocess_ocr(const unsigned char* src, int w, int h, float* dst) {
    int tw = 320; int th = 48;
    float sx = (float)w / tw;
    float sy = (float)h / th;
    
    for(int r=0; r<th; r++) {
        for(int c=0; c<tw; c++) {
            int ox = (int)(c * sx);
            int oy = (int)(r * sy);
            if(ox>=w) ox=w-1; if(oy>=h) oy=h-1;
            int idx = (oy*w + ox)*3;
            // PP-OCR Rec Norm: (x/255 - 0.5)/0.5
            for(int k=0; k<3; k++)
                dst[k*th*tw + r*tw + c] = (src[idx+k]/255.0f - 0.5f) / 0.5f;
        }
    }
}
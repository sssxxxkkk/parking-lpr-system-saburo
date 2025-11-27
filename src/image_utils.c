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

// DBNet 预处理
void preprocess_dbnet(const unsigned char* src, int src_w, int src_h, int target_size, float* dst) {
    // 1. 计算缩放 (Resize with Padding)
    float scale = fminf((float)target_size/src_w, (float)target_size/src_h);
    int new_w = (int)(src_w * scale);
    int new_h = (int)(src_h * scale);
    
    // 初始化为 0
    memset(dst, 0, 3 * target_size * target_size * sizeof(float));
    
    // ImageNet 标准均值和方差 (PP-OCR 专用)
    float mean[] = {0.485f, 0.456f, 0.406f};
    float std[]  = {0.229f, 0.224f, 0.225f};

    int plane_size = target_size * target_size;

    for(int r = 0; r < new_h; r++) {
        for(int c = 0; c < new_w; c++) {
            int src_x = (int)(c / scale);
            int src_y = (int)(r / scale);
            if (src_x >= src_w) src_x = src_w - 1;
            if (src_y >= src_h) src_y = src_h - 1;
            
            int src_idx = (src_y * src_w + src_x) * 3;
            int dst_idx = r * target_size + c;

            // 注意：这里假设 src 是 RGB 顺序 (如果是 BGR 请交换 0 和 2)
            // 根据之前的调试，你的 src 应该是正确的 RGB
            float r_val = src[src_idx + 0] / 255.0f;
            float g_val = src[src_idx + 1] / 255.0f;
            float b_val = src[src_idx + 2] / 255.0f;

            // 归一化公式： (val - mean) / std
            dst[dst_idx + 0 * plane_size] = (r_val - mean[0]) / std[0];
            dst[dst_idx + 1 * plane_size] = (g_val - mean[1]) / std[1];
            dst[dst_idx + 2 * plane_size] = (b_val - mean[2]) / std[2];
        }
    }
}

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
    int min_x = mw, min_y = mh, max_x = 0, max_y = 0;
    int cnt = 0;
    
    for(int i=0; i<mh*mw; i++) {
        // 如果模型输出没有经过 Sigmoid，这里需要手动做
        // 绝大多数 PP-OCR v3 导出的 ONNX 已经是 Sigmoid 过的概率值
        // 如果你发现热力图很暗，可能需要在这里加 sigmoid
        float val = map[i]; 
        
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
    
    // 如果点太少，认为是噪点
    if (cnt < 50) { 
        *w=0; *h=0; 
        // printf("[DBNet] 点太少 (%d), 未找到车牌\n", cnt);
        return; 
    }
    
    *x = min_x; 
    *y = min_y;
    *w = max_x - min_x;
    *h = max_y - min_y;
    printf("[DBNet] 找到目标: x=%d y=%d w=%d h=%d (Score > %.2f 的点数: %d)\n", *x, *y, *w, *h, thresh, cnt);
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

// 辅助函数：计算两个框的 IoU (交并比)
static float compute_iou(Detection* a, Detection* b) {
    float area_a = (a->x2 - a->x1) * (a->y2 - a->y1);
    float area_b = (b->x2 - b->x1) * (b->y2 - b->y1);

    float xx1 = fmaxf(a->x1, b->x1);
    float yy1 = fmaxf(a->y1, b->y1);
    float xx2 = fminf(a->x2, b->x2);
    float yy2 = fminf(a->y2, b->y2);

    float w = fmaxf(0.0f, xx2 - xx1);
    float h = fmaxf(0.0f, yy2 - yy1);
    float inter = w * h;

    return inter / (area_a + area_b - inter + 1e-6f);
}

// 辅助函数：排序用的比较函数 (降序)
static int compare_dets(const void* a, const void* b) {
    float diff = ((Detection*)b)->confidence - ((Detection*)a)->confidence;
    if (diff > 0) return 1;
    else if (diff < 0) return -1;
    return 0;
}

// NMS 核心函数
void nms_yolo(Detection* dets, int* count, float iou_thres) {
    if (*count <= 0) return;

    // 1. 按置信度从高到低排序
    qsort(dets, *count, sizeof(Detection), compare_dets);

    // 2. 标记需要删除的框 (使用 keep 数组，0表示删除，1表示保留)
    int* keep = malloc(*count * sizeof(int));
    for(int i=0; i<*count; i++) keep[i] = 1;

    for(int i=0; i<*count; i++) {
        if(keep[i] == 0) continue; // 已经被剔除的忽略

        for(int j=i+1; j<*count; j++) {
            if(keep[j] == 0) continue;

            float iou = compute_iou(&dets[i], &dets[j]);
            // 如果两个框重叠严重 (IoU > 阈值)，删掉置信度较低的那个(j)
            if(iou > iou_thres) {
                keep[j] = 0;
            }
        }
    }

    // 3. 压缩数组 (移除被标记删除的)
    int new_count = 0;
    for(int i=0; i<*count; i++) {
        if(keep[i] == 1) {
            dets[new_count] = dets[i]; // 结构体拷贝
            new_count++;
        }
    }
    
    *count = new_count; // 更新数量
    free(keep);
}
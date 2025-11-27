#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include "include/plate_recognition.h"
#include "include/video_capture.h"
#include "include/utils.h"

static int g_running = 1;
void handle_sig(int sig) { (void)sig; g_running = 0; }

int main() {
    signal(SIGINT, handle_sig);

    // 1. 加载配置
    AppConfig config = {
        .device = "/dev/video0",
        .vehicle_model = "models/yolov5s.onnx",
        .plate_model = "models/ppocr_det_v4.onnx",
        .ocr_model = "models/ppocr_rec_v4.onnx"
    };

    // 2. 初始化 AI 系统
    if (system_init(&config) != 0) return -1;

    // 3. 初始化摄像头
    CameraContext cam;
    if (camera_init(&cam, config.device, 1280, 720) != 0) {
        system_cleanup();
        return -1;
    }

    printf("========= 停车道闸车牌系统启动 =========\n");

    int loop_count = 0;

    while (g_running) {
        
        unsigned char* frame = NULL;
        int ret = camera_capture(&cam, &frame);
        if (ret == 0) {
            loop_count++ ;


            if (loop_count % 30 == 0) {
                printf("."); 
                fflush(stdout);
            }

            if (loop_count % 5 != 0) {
                usleep(1000); 
                continue; 
            }

            int count = 0;
            // 核心调用
            DetectionResult* results = process_frame(frame, cam.width, cam.height, &count);
            if (count > 0) {
                printf(">>> 帧检测: %d 辆车\n", count);
                for (int i = 0; i < count; i++) {
                    printf("   [车辆 %d] 车牌: %s | 欺诈: %s\n", 
                           i, 
                           results[i].plate_text, 
                           results[i].is_fraud ? "YES (拦截)" : "NO (放行)");
                }
            } else {
                // printf("."); fflush(stdout);
            }

            if (results) free(results);
        } else {
            // loop_count++ ;
            usleep(10000); // 10ms
        }
    }

    camera_close(&cam);
    system_cleanup();
    printf("\n系统退出。\n");
    return 0;
}
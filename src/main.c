#include "include/plate_recognition.h"
#include "include/common_types.h"      // 首先包含基础类型
#include "include/image_utils.h"       // 然后图像工具
#include "include/onnx_inference.h"    // ONNX推理
#include "include/video_capture.h"     // 视频捕获
#include "include/anti_fraud.h"        // 防欺诈
#include "include/utils.h"             // 工具函数
#include <signal.h>
#include <unistd.h>

// 全局变量
static ONNXModel g_vehicle_model;
static ONNXModel g_plate_detector_model;
static ONNXModel g_ocr_model;
static volatile sig_atomic_t g_shutdown_requested = 0;

void signal_handler(int sig) {
    printf("接收到信号 %d, 准备关闭系统...\n", sig);
    g_shutdown_requested = 1;
}

int system_init(SystemConfig* config) {
    printf("初始化车牌识别系统...\n");
    
    // 初始化车辆检测模型
    if (onnx_model_init(&g_vehicle_model, config->vehicle_model) != 0) {
        fprintf(stderr, "车辆检测模型初始化失败\n");
        return -1;
    }
    
    // 初始化车牌检测模型
    if (onnx_model_init(&g_plate_detector_model, config->plate_detector_model) != 0) {
        fprintf(stderr, "车牌检测模型初始化失败\n");
        onnx_model_cleanup(&g_vehicle_model);
        return -1;
    }
    
    // 初始化OCR模型
    if (onnx_model_init(&g_ocr_model, config->ocr_model) != 0) {
        fprintf(stderr, "OCR模型初始化失败\n");
        onnx_model_cleanup(&g_vehicle_model);
        onnx_model_cleanup(&g_plate_detector_model);
        return -1;
    }
    
    printf("所有模块初始化成功\n");
    return 0;
}

// 简化的处理函数
DetectionResult* process_frame(unsigned char* image_data, int width, int height, int* result_count) {
    (void)image_data;  // 标记未使用参数
    (void)width;
    (void)height;
    
    // 简化实现 - 返回一个测试结果
    DetectionResult* results = malloc(sizeof(DetectionResult));
    strcpy(results[0].plate_text, "豫A12345");
    results[0].confidence = 0.9f;
    results[0].vehicle_bbox[0] = 100;
    results[0].vehicle_bbox[1] = 100;
    results[0].vehicle_bbox[2] = 500;
    results[0].vehicle_bbox[3] = 300;
    results[0].plate_bbox[0] = 200;
    results[0].plate_bbox[1] = 150;
    results[0].plate_bbox[2] = 400;
    results[0].plate_bbox[3] = 200;
    results[0].timestamp = time(NULL);
    results[0].is_fraud = false;
    strcpy(results[0].fraud_reason, "正常");
    
    *result_count = 1;
    return results;
}

void system_cleanup() {
    printf("清理系统资源...\n");
    onnx_model_cleanup(&g_vehicle_model);
    onnx_model_cleanup(&g_plate_detector_model);
    onnx_model_cleanup(&g_ocr_model);
    printf("系统清理完成\n");
}

int main(int argc, char* argv[]) {
    (void)argc;  // 标记未使用参数
    (void)argv;
    
    // 注册信号处理
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    printf("车牌识别系统启动\n");
    printf("这是一个简化版本，用于测试编译\n");
    
    // 使用硬编码配置进行测试
    SystemConfig config;
    strcpy(config.vehicle_model, "models/yolov5s.onnx");
    strcpy(config.plate_detector_model, "models/ppocr_det.onnx");
    strcpy(config.ocr_model, "models/ppocr_rec.onnx");
    
    // 初始化系统
    if (system_init(&config) != 0) {
        fprintf(stderr, "系统初始化失败\n");
        return -1;
    }
    
    printf("系统初始化成功，开始模拟运行...\n");
    
    //运行测试
    int frame_count = 0;
    CameraContext cam;
    camera_init(&cam, "/dev/video0", 640, 480);

    while (!g_shutdown_requested && frame_count < 10) {
        unsigned char* frame = NULL;
        camera_capture_frame(&cam, &frame);
        printf("获取摄像头图像");

        // 1. 预处理成 640×640
        Image resized = preprocess_for_yolo(frame, cam.width, cam.height, 640);

        // 2. 转成 float 数组 (NCHW)
        float* input_data = image_to_float_array(&resized);

        // 3. 送入 ONNX 推理
        // run_yolo_inference(session, input_tensor);
        float* output = NULL;
        size_t output_size = 0;
        onnx_model_predict(&g_vehicle_model, input_data, 640*480*3, &output, &output_size);

        // YOLO 后处理
        int det_count = 0;
        Detection* dets = yolo_postprocess(output, output_size, 640, 480, 0.5f, &det_count);

        printf("帧 %d 检测到 %d 辆车\n", frame_count, det_count);
        for (int i = 0; i < det_count; i++) {
            printf("  车辆框: [%.1f, %.1f, %.1f, %.1f], conf=%.2f\n",
                dets[i].x1, dets[i].y1, dets[i].x2, dets[i].y2,
                dets[i].confidence);
        }

        free(input_data);
        free(output);
        free(dets);

        frame_count++;
    }
    camera_cleanup(&cam);


    // 模拟运行
    // int frame_count = 0;
    // while (!g_shutdown_requested && frame_count < 5) {
    //     printf("处理帧 %d\n", frame_count);
        
    //     // 模拟处理
    //     int result_count;
    //     unsigned char dummy_data[100] = {0};
    //     DetectionResult* results = process_frame(dummy_data, 640, 480, &result_count);
        
    //     if (result_count > 0) {
    //         printf("检测到车牌: %s (置信度: %.2f)\n", 
    //                results[0].plate_text, results[0].confidence);
    //     }
        
    //     free(results);
    //     frame_count++;
    //     sleep(1);
    // }
    
    system_cleanup();
    printf("系统正常退出\n");
    return 0;
}
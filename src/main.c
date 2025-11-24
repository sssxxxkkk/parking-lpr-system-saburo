#include "include/plate_recognition.h"
#include "include/onnx_inference.h"
#include "include/image_utils.h"
#include "include/video_capture.h"
#include "include/anti_fraud.h"
#include "include/utils.h"
#include <signal.h>
#include <unistd.h>

// 全局变量
static ONNXModel g_vehicle_model;
static ONNXModel g_plate_model;
static ONNXModel g_ocr_model;
static CameraContext g_camera;
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
    if (onnx_model_init(&g_plate_model, config->plate_model) != 0) {
        fprintf(stderr, "车牌检测模型初始化失败\n");
        onnx_model_cleanup(&g_vehicle_model);
        return -1;
    }
    
    // 初始化OCR模型
    if (onnx_model_init(&g_ocr_model, config->ocr_model) != 0) {
        fprintf(stderr, "OCR模型初始化失败\n");
        onnx_model_cleanup(&g_vehicle_model);
        onnx_model_cleanup(&g_plate_model);
        return -1;
    }
    
    // 初始化摄像头
    if (camera_init(&g_camera, config->camera_device, 
                   config->camera_width, config->camera_height) != 0) {
        fprintf(stderr, "摄像头初始化失败\n");
        onnx_model_cleanup(&g_vehicle_model);
        onnx_model_cleanup(&g_plate_model);
        onnx_model_cleanup(&g_ocr_model);
        return -1;
    }
    
    printf("所有模块初始化成功\n");
    return 0;
}

// OCR输出解析函数 (简化版本)
float parse_ocr_output(float* output, size_t output_size, char* plate_text) {
    // 简化实现: 这里需要根据OCR模型的输出格式来解析
    // 实际实现可能涉及CTC解码、字符映射等
    
    float confidence = 0.8f; // 示例置信度
    strcpy(plate_text, "京A12345"); // 示例文本
    
    // 实际实现需要:
    // 1. 解析字符概率
    // 2. 使用CTC解码或argmax
    // 3. 映射到实际字符
    // 4. 计算总体置信度
    
    return confidence;
}

DetectionResult* process_frame(unsigned char* image_data, int width, int height, int* result_count) {
    int max_results = 10; // 每帧最大检测结果数
    DetectionResult* results = malloc(max_results * sizeof(DetectionResult));
    int result_idx = 0;
    
    // YOLO预处理参数
    float mean[3] = {0.485, 0.456, 0.406};
    float std[3] = {0.229, 0.224, 0.225};
    
    // 步骤1: 车辆检测
    Image yolo_input = preprocess_for_yolo(image_data, width, height, MODEL_INPUT_SIZE);
    float* input_data = image_to_float_array(yolo_input.data, yolo_input.width, 
                                           yolo_input.height, yolo_input.channels,
                                           mean, std);
    
    float* vehicle_output;
    size_t vehicle_output_size;
    onnx_model_predict(&g_vehicle_model, input_data, 
                      yolo_input.width * yolo_input.height * yolo_input.channels,
                      &vehicle_output, &vehicle_output_size);
    
    int vehicle_count;
    Detection* vehicles = yolo_postprocess(vehicle_output, vehicle_output_size,
                                          width, height, 
                                          0.5f, &vehicle_count);
    
    free(input_data);
    image_free(&yolo_input);
    free(vehicle_output);
    
    // 步骤2: 对每个检测到的车辆进行车牌检测
    for (int i = 0; i < vehicle_count && result_idx < max_results; i++) {
        Detection vehicle = vehicles[i];
        
        // 提取车辆区域
        int x1 = MAX(0, (int)vehicle.bbox[0]);
        int y1 = MAX(0, (int)vehicle.bbox[1]);
        int x2 = MIN(width - 1, (int)vehicle.bbox[2]);
        int y2 = MIN(height - 1, (int)vehicle.bbox[3]);
        
        // 创建车辆ROI图像
        Image vehicle_roi;
        vehicle_roi.width = x2 - x1;
        vehicle_roi.height = y2 - y1;
        vehicle_roi.channels = 3;
        vehicle_roi.data = malloc(vehicle_roi.width * vehicle_roi.height * 3);
        
        // 提取车辆区域数据
        for (int y = y1; y < y2; y++) {
            for (int x = x1; x < x2; x++) {
                int src_idx = (y * width + x) * 3;
                int dst_idx = ((y - y1) * vehicle_roi.width + (x - x1)) * 3;
                memcpy(&vehicle_roi.data[dst_idx], &image_data[src_idx], 3);
            }
        }
        
        // 在车辆区域内检测车牌
        Image plate_input = preprocess_for_yolo(vehicle_roi.data, vehicle_roi.width, 
                                               vehicle_roi.height, MODEL_INPUT_SIZE);
        float* plate_input_data = image_to_float_array(plate_input.data, plate_input.width,
                                                      plate_input.height, plate_input.channels,
                                                      mean, std);
        
        float* plate_output;
        size_t plate_output_size;
        onnx_model_predict(&g_plate_model, plate_input_data,
                          plate_input.width * plate_input.height * plate_input.channels,
                          &plate_output, &plate_output_size);
        
        int plate_count;
        Detection* plates = yolo_postprocess(plate_output, plate_output_size,
                                            vehicle_roi.width, vehicle_roi.height,
                                            0.5f, &plate_count);
        
        free(plate_input_data);
        image_free(&plate_input);
        free(plate_output);
        
        // 步骤3: 对每个检测到的车牌进行OCR识别
        for (int j = 0; j < plate_count && result_idx < max_results; j++) {
            Detection plate = plates[j];
            
            // 提取车牌区域 (相对于车辆ROI的坐标)
            int plate_x1 = MAX(0, (int)plate.bbox[0]);
            int plate_y1 = MAX(0, (int)plate.bbox[1]);
            int plate_x2 = MIN(vehicle_roi.width - 1, (int)plate.bbox[2]);
            int plate_y2 = MIN(vehicle_roi.height - 1, (int)plate.bbox[3]);
            
            Image plate_roi = image_crop(&vehicle_roi, plate_x1, plate_y1, 
                                        plate_x2 - plate_x1, plate_y2 - plate_y1);
            
            // OCR预处理和识别
            Image ocr_input = preprocess_for_ocr(&plate_roi, 48);
            
            float* ocr_input_data = image_to_float_array(ocr_input.data, ocr_input.width,
                                                        ocr_input.height, ocr_input.channels,
                                                        mean, std);
            float* ocr_output;
            size_t ocr_output_size;
            
            onnx_model_predict(&g_ocr_model, ocr_input_data,
                              ocr_input.width * ocr_input.height * ocr_input.channels,
                              &ocr_output, &ocr_output_size);
            
            // 解析OCR输出
            char plate_text[16] = {0};
            float ocr_confidence = parse_ocr_output(ocr_output, ocr_output_size, plate_text);
            
            // 步骤4: 防欺诈检测
            bool is_fraud = false;
            char fraud_reason[64] = {0};
            
            is_fraud = detect_fraud(&plate_roi, plate_text, ocr_confidence, fraud_reason);
            
            // 填充结果
            DetectionResult result;
            strncpy(result.plate_text, plate_text, sizeof(result.plate_text) - 1);
            result.confidence = plate.confidence * ocr_confidence;
            result.vehicle_bbox[0] = x1; result.vehicle_bbox[1] = y1;
            result.vehicle_bbox[2] = x2; result.vehicle_bbox[3] = y2;
            result.plate_bbox[0] = x1 + plate_x1; result.plate_bbox[1] = y1 + plate_y1;
            result.plate_bbox[2] = x1 + plate_x2; result.plate_bbox[3] = y1 + plate_y2;
            result.timestamp = time(NULL);
            result.is_fraud = is_fraud;
            strncpy(result.fraud_reason, fraud_reason, sizeof(result.fraud_reason) - 1);
            
            results[result_idx++] = result;
            
            // 清理资源
            free(ocr_input_data);
            free(ocr_output);
            image_free(&plate_roi);
            image_free(&ocr_input);
        }
        
        free(plates);
        free(vehicle_roi.data);
    }
    
    free(vehicles);
    *result_count = result_idx;
    return results;
}

void system_cleanup() {
    printf("清理系统资源...\n");
    camera_cleanup(&g_camera);
    onnx_model_cleanup(&g_vehicle_model);
    onnx_model_cleanup(&g_plate_model);
    onnx_model_cleanup(&g_ocr_model);
    printf("系统清理完成\n");
}

int main(int argc, char* argv[]) {
    // 注册信号处理
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // 解析配置
    AppConfig app_config;
    if (config_parse(&app_config, "config/system.conf") != 0) {
        fprintf(stderr, "使用默认配置\n");
    }
    
    // 转换为系统配置
    SystemConfig config;
    strcpy(config.vehicle_model, app_config.vehicle_model);
    strcpy(config.plate_model, app_config.plate_model);
    strcpy(config.ocr_model, app_config.ocr_model);
    strcpy(config.camera_device, app_config.camera_device);
    config.camera_width = app_config.camera_width;
    config.camera_height = app_config.camera_height;
    config.vehicle_threshold = app_config.vehicle_threshold;
    config.plate_threshold = app_config.plate_threshold;
    config.ocr_threshold = app_config.ocr_threshold;
    config.processing_interval = 100;
    config.max_detection_per_frame = 10;
    config.enable_anti_fraud = true;
    config.fps = 15;
    
    // 初始化系统
    if (system_init(&config) != 0) {
        fprintf(stderr, "系统初始化失败\n");
        return -1;
    }
    
    printf("开始车牌识别...\n");
    int frame_count = 0;
    
    while (!g_shutdown_requested) {
        unsigned char* frame_data = NULL;
        int frame_size = camera_capture_frame(&g_camera, &frame_data);
        
        if (frame_size > 0) {
            // 处理帧
            int result_count;
            DetectionResult* results = process_frame(frame_data, 
                                                   config.camera_width,
                                                   config.camera_height, 
                                                   &result_count);
            
            // 输出结果
            if (result_count > 0) {
                printf("帧 %d: 检测到 %d 个车牌\n", frame_count, result_count);
                for (int i = 0; i < result_count; i++) {
                    printf("  车牌 %d: %s (置信度: %.2f)\n", 
                           i + 1, results[i].plate_text, results[i].confidence);
                    if (results[i].is_fraud) {
                        printf("    警告: %s\n", results[i].fraud_reason);
                    }
                }
            }
            
            free(results);
            frame_count++;
        }
        
        // 控制处理频率
        usleep(1000000 / config.fps);
    }
    
    system_cleanup();
    printf("系统正常退出\n");
    return 0;
}
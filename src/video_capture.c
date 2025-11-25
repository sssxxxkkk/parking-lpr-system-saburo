#include "include/video_capture.h"
#include "include/utils.h"
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>

// 简化的视频捕获实现
int camera_init(CameraContext* ctx, const char* device, int width, int height) {
    printf("初始化模拟摄像头: %s (%dx%d)\n", device, width, height);
    
    // 简化实现 - 不实际打开设备
    ctx->fd = -1;  // 标记为模拟模式
    ctx->width = width;
    ctx->height = height;
    ctx->buffer_count = 0;
    
    printf("模拟摄像头初始化完成\n");
    return 0;
}

int camera_capture_frame(CameraContext* ctx, unsigned char** frame_data) {
    // 简化实现 - 返回模拟图像数据
    static int frame_counter = 0;
    
    if (ctx->fd == -1) {
        // 模拟模式 - 生成假数据
        printf("捕获模拟帧 %d\n", frame_counter);
        
        // 分配帧数据内存 (RGB格式)
        static unsigned char dummy_frame[640*480*3];
        *frame_data = dummy_frame;
        
        // 生成一些简单的测试图案
        for (int y = 0; y < 480; y++) {
            for (int x = 0; x < 640; x++) {
                int idx = (y * 640 + x) * 3;
                
                // 创建简单的渐变背景
                dummy_frame[idx] = (x + frame_counter) % 256;     // R
                dummy_frame[idx + 1] = (y + frame_counter) % 256; // G  
                dummy_frame[idx + 2] = (x + y + frame_counter) % 256; // B
                
                // 在中心添加一个矩形模拟车辆
                if (x > 200 && x < 440 && y > 150 && y < 330) {
                    dummy_frame[idx] = 255;     // 红色车辆
                    dummy_frame[idx + 1] = 0;
                    dummy_frame[idx + 2] = 0;
                }
                
                // 在车辆区域添加一个白色矩形模拟车牌
                if (x > 280 && x < 360 && y > 280 && y < 310) {
                    dummy_frame[idx] = 255;     // 白色车牌
                    dummy_frame[idx + 1] = 255;
                    dummy_frame[idx + 2] = 255;
                }
            }
        }
        
        frame_counter++;
        return 640 * 480 * 3;  // 返回数据大小
    }
    
    // 如果是真实设备，这里会有实际的捕获代码
    return -1;
}

void camera_cleanup(CameraContext* ctx) {
    printf("清理摄像头资源...\n");
    // 模拟模式不需要清理
    if (ctx->fd != -1) {
        close(ctx->fd);
    }
    printf("摄像头资源清理完成\n");
}

// 简化的格式转换函数（如果需要）
int convert_yuyv_to_rgb(const unsigned char* yuyv, unsigned char* rgb, int width, int height) {
    // 简化实现 - 直接复制数据（假设已经是RGB）
    memcpy(rgb, yuyv, width * height * 3);
    return 0;
}

int convert_mjpeg_to_rgb(const unsigned char* mjpeg, unsigned char* rgb, int width, int height) {
    // 简化实现 - 直接复制数据（假设已经是RGB）
    memcpy(rgb, mjpeg, width * height * 3);
    return 0;
}
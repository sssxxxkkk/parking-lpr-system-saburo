#ifndef VIDEO_CAPTURE_H
#define VIDEO_CAPTURE_H

typedef struct {
    int fd;
    int width;
    int height;
    unsigned char* buffer_rgb; // 转换后的RGB缓存
} CameraContext;

int camera_init(CameraContext* ctx, const char* device, int w, int h);
int camera_capture(CameraContext* ctx, unsigned char** frame_data);
void camera_close(CameraContext* ctx);

#endif
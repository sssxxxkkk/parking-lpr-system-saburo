#ifndef VIDEO_CAPTURE_H
#define VIDEO_CAPTURE_H

#include <linux/videodev2.h>

typedef struct {
    int fd;
    struct v4l2_format format;
    unsigned char* buffers[4];
    unsigned int buffer_count;
    int width;
    int height;
} CameraContext;

// 函数声明
int camera_init(CameraContext* ctx, const char* device, int width, int height);
int camera_capture_frame(CameraContext* ctx, unsigned char** frame_data);
void camera_cleanup(CameraContext* ctx);

#endif
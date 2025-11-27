#include "include/video_capture.h"
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <string.h>
#include <unistd.h>

struct buffer { void* start; size_t length; };
static struct buffer g_bufs[4];

// YUYV -> RGB 转换
static void yuyv_to_rgb(const unsigned char* yuyv, unsigned char* rgb, int width, int height) {
    int z = 0;
    
    // 定义一个简单的宏来限制范围在 0-255 (Clamp)
    // 这是标准的 C 语言写法
    #define CLP(x) ((x)<0?0:((x)>255?255:(x)))

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j += 2) {
            int pos = (i * width + j) * 2;
            int y1 = yuyv[pos];
            int u  = yuyv[pos + 1];
            int y2 = yuyv[pos + 2];
            int v  = yuyv[pos + 3];

            // YUV 转 RGB 的整数运算公式
            int c1 = y1 - 16; 
            int c2 = y2 - 16;
            int d = u - 128; 
            int e = v - 128;

            int r1 = (298 * c1 + 409 * e + 128) >> 8;
            int g1 = (298 * c1 - 100 * d - 208 * e + 128) >> 8;
            int b1 = (298 * c1 + 516 * d + 128) >> 8;
            
            int r2 = (298 * c2 + 409 * e + 128) >> 8;
            int g2 = (298 * c2 - 100 * d - 208 * e + 128) >> 8;
            int b2 = (298 * c2 + 516 * d + 128) >> 8;

            // 使用宏进行赋值
            rgb[z++] = (unsigned char)CLP(r1); 
            rgb[z++] = (unsigned char)CLP(g1); 
            rgb[z++] = (unsigned char)CLP(b1);
            
            rgb[z++] = (unsigned char)CLP(r2); 
            rgb[z++] = (unsigned char)CLP(g2); 
            rgb[z++] = (unsigned char)CLP(b2);
        }
    }
    
    // 解除宏定义，养成好习惯
    #undef CLP
}

int camera_init(CameraContext* ctx, const char* dev, int w, int h) {
    ctx->fd = open(dev, O_RDWR);
    if(ctx->fd < 0) {
        perror("无法打开摄像头设备");
        return -1;
    }
    ctx->width = w; 
    ctx->height = h;
    ctx->buffer_rgb = malloc(w * h * 3);

    struct v4l2_format fmt = {0};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = w;
    fmt.fmt.pix.height = h;
    // 大多数 USB 摄像头默认支持 YUYV
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV; 
    if (ioctl(ctx->fd, VIDIOC_S_FMT, &fmt) < 0) {
        perror("设置像素格式失败");
        return -1;
    }

    struct v4l2_requestbuffers req = {0};
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (ioctl(ctx->fd, VIDIOC_REQBUFS, &req) < 0) {
        perror("申请缓冲区失败");
        return -1;
    }

    for (int i = 0; i < 4; ++i) {
        struct v4l2_buffer buf = {0};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        ioctl(ctx->fd, VIDIOC_QUERYBUF, &buf);
        g_bufs[i].length = buf.length;
        g_bufs[i].start = mmap(NULL, buf.length, PROT_READ|PROT_WRITE, MAP_SHARED, ctx->fd, buf.m.offset);
        if (g_bufs[i].start == MAP_FAILED) {
            perror("mmap 失败");
            return -1;
        }
        ioctl(ctx->fd, VIDIOC_QBUF, &buf);
    }
    
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(ctx->fd, VIDIOC_STREAMON, &type) < 0) {
        perror("开启视频流失败");
        return -1;
    }
    return 0;
}

int camera_capture(CameraContext* ctx, unsigned char** out) {
    struct v4l2_buffer buf = {0};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    
    // 从队列中取出一帧
    if (ioctl(ctx->fd, VIDIOC_DQBUF, &buf) < 0) {
        // 偶尔失败是正常的，不要立即退出
        return -1;
    }
    
    // 转码: YUYV -> RGB
    yuyv_to_rgb((unsigned char*)g_bufs[buf.index].start, ctx->buffer_rgb, ctx->width, ctx->height);
    
    // 返回 RGB 数据指针
    *out = ctx->buffer_rgb;
    
    // 将缓冲区放回队列
    ioctl(ctx->fd, VIDIOC_QBUF, &buf);
    return 0;
}

void camera_close(CameraContext* ctx) {
    if (ctx->buffer_rgb) free(ctx->buffer_rgb);
    if (ctx->fd >= 0) {
        int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        ioctl(ctx->fd, VIDIOC_STREAMOFF, &type);
        for(int i=0; i<4; i++) {
            if (g_bufs[i].start) munmap(g_bufs[i].start, g_bufs[i].length);
        }
        close(ctx->fd);
    }
}
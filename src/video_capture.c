#include "include/video_capture.h"
#include "include/utils.h"
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>

int camera_init(CameraContext* ctx, const char* device, int width, int height) {
    printf("初始化摄像头: %s\n", device);
    
    ctx->fd = open(device, O_RDWR);
    if (ctx->fd < 0) {
        perror("无法打开摄像头设备");
        return -1;
    }
    
    // 查询摄像头能力
    struct v4l2_capability cap;
    if (ioctl(ctx->fd, VIDIOC_QUERYCAP, &cap) < 0) {
        perror("查询摄像头能力失败");
        close(ctx->fd);
        return -1;
    }
    
    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        fprintf(stderr, "设备不支持视频捕获\n");
        close(ctx->fd);
        return -1;
    }
    
    // 设置视频格式
    memset(&ctx->format, 0, sizeof(ctx->format));
    ctx->format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ctx->format.fmt.pix.width = width;
    ctx->format.fmt.pix.height = height;
    ctx->format.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG; // 或 V4L2_PIX_FMT_YUYV
    ctx->format.fmt.pix.field = V4L2_FIELD_NONE;
    
    if (ioctl(ctx->fd, VIDIOC_S_FMT, &ctx->format) < 0) {
        perror("设置视频格式失败");
        close(ctx->fd);
        return -1;
    }
    
    ctx->width = width;
    ctx->height = height;
    
    // 申请缓冲区
    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    
    if (ioctl(ctx->fd, VIDIOC_REQBUFS, &req) < 0) {
        perror("申请缓冲区失败");
        close(ctx->fd);
        return -1;
    }
    
    ctx->buffer_count = req.count;
    
    // 映射缓冲区
    for (unsigned int i = 0; i < ctx->buffer_count; i++) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        
        if (ioctl(ctx->fd, VIDIOC_QUERYBUF, &buf) < 0) {
            perror("查询缓冲区失败");
            close(ctx->fd);
            return -1;
        }
        
        ctx->buffers[i] = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, 
                              MAP_SHARED, ctx->fd, buf.moffset);
        
        if (ctx->buffers[i] == MAP_FAILED) {
            perror("映射缓冲区失败");
            close(ctx->fd);
            return -1;
        }
        
        // 将缓冲区加入队列
        if (ioctl(ctx->fd, VIDIOC_QBUF, &buf) < 0) {
            perror("缓冲区入队失败");
            close(ctx->fd);
            return -1;
        }
    }
    
    // 开始采集
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(ctx->fd, VIDIOC_STREAMON, &type) < 0) {
        perror("开始采集失败");
        close(ctx->fd);
        return -1;
    }
    
    printf("摄像头初始化完成: %dx%d\n", width, height);
    return 0;
}

int camera_capture_frame(CameraContext* ctx, unsigned char** frame_data) {
    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    
    // 从队列中取出一个已填充的缓冲区
    if (ioctl(ctx->fd, VIDIOC_DQBUF, &buf) < 0) {
        perror("获取帧失败");
        return -1;
    }
    
    *frame_data = ctx->buffers[buf.index];
    
    // 将缓冲区重新加入队列
    if (ioctl(ctx->fd, VIDIOC_QBUF, &buf) < 0) {
        perror("缓冲区重新入队失败");
        return -1;
    }
    
    return buf.bytesused; // 返回帧数据大小
}

void camera_cleanup(CameraContext* ctx) {
    if (ctx->fd >= 0) {
        // 停止采集
        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        ioctl(ctx->fd, VIDIOC_STREAMOFF, &type);
        
        // 取消映射
        for (unsigned int i = 0; i < ctx->buffer_count; i++) {
            if (ctx->buffers[i]) {
                munmap(ctx->buffers[i], ctx->format.fmt.pix.sizeimage);
            }
        }
        
        close(ctx->fd);
    }
}
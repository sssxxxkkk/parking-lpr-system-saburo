#include "include/video_capture.h"
#include "include/utils.h"

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <linux/videodev2.h>

#define USE_REAL_CAMERA   1   // ===== 👈 设置为 0 则进入模拟模式 =====

// 缓存结构
typedef struct {
    void* start;
    size_t length;
} Buffer;

// 全局缓存（最多4个）
static Buffer buffers[4];

// ==========================
// 初始化摄像头
// ==========================
int camera_init(CameraContext* ctx, const char* device, int width, int height) {

#if USE_REAL_CAMERA == 0
    printf("【模拟模式】初始化模拟摄像头: %s (%dx%d)\n", device, width, height);
    ctx->fd = -1;
    ctx->width = width;
    ctx->height = height;
    ctx->buffer_count = 0;
    return 0;
#endif

    printf("【真实摄像头】打开设备: %s\n", device);

    ctx->fd = open(device, O_RDWR);
    if (ctx->fd < 0) {
        perror("摄像头打开失败");
        return -1;
    }

    ctx->width = width;
    ctx->height = height;

    // 配置格式(YUYV)
    struct v4l2_format fmt = {0};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width  = width;
    fmt.fmt.pix.height = height;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;

    if (ioctl(ctx->fd, VIDIOC_S_FMT, &fmt) < 0) {
        perror("设置格式失败");
        return -1;
    }

    // 请求缓冲区
    struct v4l2_requestbuffers req = {0};
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (ioctl(ctx->fd, VIDIOC_REQBUFS, &req) < 0) {
        perror("申请缓冲区失败");
        return -1;
    }

    ctx->buffer_count = req.count;

    // mmap 每个 buffer
    for (int i = 0; i < req.count; i++) {
        struct v4l2_buffer buf = {0};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;

        ioctl(ctx->fd, VIDIOC_QUERYBUF, &buf);

        buffers[i].length = buf.length;
        buffers[i].start = mmap(NULL, buf.length,
                                PROT_READ | PROT_WRITE, MAP_SHARED,
                                ctx->fd, buf.m.offset);

        // 加入队列
        ioctl(ctx->fd, VIDIOC_QBUF, &buf);
    }

    // 开始捕获
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(ctx->fd, VIDIOC_STREAMON, &type);

    printf("【真实摄像头】初始化完成 (%dx%d)\n", width, height);
    return 0;
}

// ==========================
// 读取一帧图像
// ==========================
int camera_capture_frame(CameraContext* ctx, unsigned char** frame_data) {

#if USE_REAL_CAMERA == 0
    // ============================
    //  模拟模式 —— 用你原来的代码
    // ============================
    static int frame_counter = 0;
    static unsigned char dummy_frame[640 * 480 * 3];

    *frame_data = dummy_frame;

    for (int y = 0; y < 480; y++) {
        for (int x = 0; x < 640; x++) {
            int idx = (y * 640 + x) * 3;

            dummy_frame[idx] = (x + frame_counter) % 256;
            dummy_frame[idx + 1] = (y + frame_counter) % 256;
            dummy_frame[idx + 2] = (x + y + frame_counter) % 256;

            // 模拟车辆
            if (x > 200 && x < 440 && y > 150 && y < 330) {
                dummy_frame[idx] = 255;
                dummy_frame[idx + 1] = 0;
                dummy_frame[idx + 2] = 0;
            }

            // 模拟车牌
            if (x > 280 && x < 360 && y > 280 && y < 310) {
                dummy_frame[idx] = 255;
                dummy_frame[idx + 1] = 255;
                dummy_frame[idx + 2] = 255;
            }
        }
    }

    frame_counter++;
    return 640 * 480 * 3;
#endif

    // ========================
    // 真实摄像头读取一帧
    // ========================
    struct v4l2_buffer buf = {0};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    ioctl(ctx->fd, VIDIOC_DQBUF, &buf);

    *frame_data = buffers[buf.index].start;

    ioctl(ctx->fd, VIDIOC_QBUF, &buf);

    return buf.bytesused;  // YUYV 格式数据大小
}


// ==========================
// 清理摄像头
// ==========================
void camera_cleanup(CameraContext* ctx) {
    printf("释放摄像头资源...\n");

#if USE_REAL_CAMERA == 1
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(ctx->fd, VIDIOC_STREAMOFF, &type);

    for (int i = 0; i < ctx->buffer_count; i++) {
        munmap(buffers[i].start, buffers[i].length);
    }
#endif

    if (ctx->fd != -1) close(ctx->fd);
}


// ==========================
// YUYV → RGB888
// ==========================
int convert_yuyv_to_rgb(const unsigned char* yuyv, unsigned char* rgb, int width, int height)
{
    int frameSize = width * height * 2;

    for (int i = 0, j = 0; i < frameSize; i += 4, j += 6) {

        int y0 = yuyv[i];
        int u  = yuyv[i + 1] - 128;
        int y1 = yuyv[i + 2];
        int v  = yuyv[i + 3] - 128;

        // 使用标准 YUV 转换公式
        int r0 = y0 + 1.402 * v;
        int g0 = y0 - 0.344136 * u - 0.714136 * v;
        int b0 = y0 + 1.772 * u;

        int r1 = y1 + 1.402 * v;
        int g1 = y1 - 0.344136 * u - 0.714136 * v;
        int b1 = y1 + 1.772 * u;

        rgb[j]     = r0 < 0 ? 0 : (r0 > 255 ? 255 : r0);
        rgb[j + 1] = g0 < 0 ? 0 : (g0 > 255 ? 255 : g0);
        rgb[j + 2] = b0 < 0 ? 0 : (b0 > 255 ? 255 : b0);

        rgb[j + 3] = r1 < 0 ? 0 : (r1 > 255 ? 255 : r1);
        rgb[j + 4] = g1 < 0 ? 0 : (g1 > 255 ? 255 : g1);
        rgb[j + 5] = b1 < 0 ? 0 : (b1 > 255 ? 255 : b1);
    }

    return 0;
}

int convert_mjpeg_to_rgb(const unsigned char* mjpeg, unsigned char* rgb, int width, int height) {
    // TODO: 你以后需要 MJPEG → RGB 时我再写
    memcpy(rgb, mjpeg, width * height * 3);
    return 0;
}

# 简化的车牌识别系统 Makefile
CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -O2 -D_GNU_SOURCE
INCLUDES = -Isrc/include -Ithird_party/onnxruntime/include
LIBS = -Lthird_party/onnxruntime/lib -lonnxruntime -lpthread -lm

# 源文件
SRCS = src/main.c src/onnx_inference.c src/image_utils.c src/video_capture.c src/anti_fraud.c src/utils.c src/plate_recognition.c
OBJS = $(SRCS:.c=.o)
TARGET = plate_recognition

# 默认目标
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(LIBS)

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# 检查依赖
check_deps:
	@if [ ! -f "third_party/onnxruntime/include/onnxruntime_c_api.h" ]; then \
		echo "错误: ONNX Runtime头文件未找到"; \
		echo "请运行: make download_deps"; \
		exit 1; \
	fi

# 下载依赖
download_deps:
	@echo "下载 ONNX Runtime..."
	@mkdir -p third_party
	@cd third_party && \
	wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.16.1/onnxruntime-linux-x64-1.16.1.tgz -O onnxruntime.tgz && \
	tar -xzf onnxruntime.tgz && \
	mv onnxruntime-linux-x64-1.16.1 onnxruntime && \
	rm onnxruntime.tgz
	@echo "ONNX Runtime下载完成"

# 清理
clean:
	rm -f $(OBJS) $(TARGET)

# 运行
run: $(TARGET)
	./$(TARGET)

.PHONY: all check_deps download_deps clean run
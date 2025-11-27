#!/bin/bash

echo "=== 车牌识别系统构建脚本 ==="

# 检查ONNX Runtime
if [ ! -f "third_party/onnxruntime/include/onnxruntime_c_api.h" ]; then
    echo "ONNX Runtime未找到，正在下载..."
    mkdir -p third_party
    cd third_party
    wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.1/onnxruntime-linux-x64-1.16.1.tgz
    tar -xzf onnxruntime-linux-x64-1.16.1.tgz
    mv onnxruntime-linux-x64-1.16.1 onnxruntime
    rm onnxruntime-linux-x64-1.16.1.tgz
    cd ..
    echo "ONNX Runtime下载完成"
fi

# 检查模型文件
echo "检查模型文件..."
if [ ! -f "models/yolov5s.onnx" ]; then
    echo "警告: yolov5s.onnx 未找到"
fi
if [ ! -f "models/ppocr_det_v4.onnx" ]; then
    echo "警告: ppocr_det.onnx 未找到"
fi
if [ ! -f "models/ppocr_rec_v4.onnx" ]; then
    echo "警告: ppocr_rec.onnx 未找到"
fi

# 编译
echo "编译项目..."
make clean
make

if [ $? -eq 0 ]; then
    echo "编译成功!"
    echo "运行: ./plate_recognition"
else
    echo "编译失败!"
    exit 1
fi
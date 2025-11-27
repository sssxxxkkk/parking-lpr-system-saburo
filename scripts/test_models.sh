#!/bin/bash

echo "测试模型集成..."

# 设置环境
source scripts/setup_env.sh

# 检查模型文件
echo "检查模型文件..."
if [ ! -f "models/yolov5s.onnx" ]; then
    echo "错误: yolov5s.onnx 不存在"
    exit 1
fi

if [ ! -f "models/ppocr_det_v4.onnx" ]; then
    echo "错误: ppocr_det_v4.onnx 不存在"
    exit 1
fi

if [ ! -f "models/ppocr_rec_v4.onnx" ]; then
    echo "错误: ppocr_rec_v4.onnx 不存在"
    exit 1
fi

echo "✓ 所有模型文件存在"

# 编译项目
echo "编译项目..."
make clean
make

if [ $? -eq 0 ]; then
    echo "✓ 编译成功"
else
    echo "✗ 编译失败"
    exit 1
fi

# 运行简单测试
echo "运行简单测试..."
./plate_recognition --test-mode

echo "测试完成！"
#!/bin/bash

# 快速开始脚本
echo "=== 车牌识别系统快速开始 ==="

# 设置环境变量
export LD_LIBRARY_PATH=third_party/onnxruntime/lib:$LD_LIBRARY_PATH

# 构建项目
./build.sh

# 运行程序
echo "启动车牌识别系统..."
./plate_recognition
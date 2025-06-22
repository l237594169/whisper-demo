#!/bin/bash

# WhisperX 视频转字幕 Web 工具启动脚本

echo "🎬 启动 WhisperX 视频转字幕 Web 工具..."

# 检查是否在conda环境中
if [[ "$CONDA_DEFAULT_ENV" != "whisperx" ]]; then
    echo "⚠️  当前不在 whisperx 环境中，尝试激活..."
    
    # 尝试激活whisperx环境
    if conda activate whisperx 2>/dev/null; then
        echo "✅ 成功激活 whisperx 环境"
    else
        echo "❌ 无法激活 whisperx 环境，请确保已创建该环境"
        echo "请运行: conda create -n whisperx python=3.9"
        echo "然后运行: conda activate whisperx"
        exit 1
    fi
else
    echo "✅ 已在 whisperx 环境中"
fi

# 检查依赖是否安装
echo "📦 检查依赖..."
if ! python -c "import flask, whisperx, torch" 2>/dev/null; then
    echo "⚠️  缺少依赖，正在安装..."
    pip install -r requirements.txt
fi

# 创建必要的目录
echo "📁 创建必要的目录..."
mkdir -p uploads outputs

# 启动应用
echo "🚀 启动 Web 服务器..."
echo "📍 应用将在 http://localhost:5000 启动"
echo "🛑 按 Ctrl+C 停止服务器"
echo ""

python app.py 
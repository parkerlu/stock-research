#!/bin/bash
# 阿里云 GPU 机器一键准备。选带 CUDA 驱动的镜像, 这个脚本装剩下的。
set -e
echo "=== 1. 检查显卡 ==="
nvidia-smi || { echo "没有 GPU 驱动, 换个预装 CUDA 的镜像"; exit 1; }
echo "=== 2. 依赖 ==="
pip install -q torch --index-url https://download.pytorch.org/whl/cu121 2>/dev/null || pip install -q torch
pip install -q numpy pandas huggingface_hub safetensors einops
echo "=== 3. Kronos 代码 ==="
[ -d Kronos ] || git clone --depth 1 https://github.com/shiyu-coder/Kronos.git
echo "=== 4. 检查数据包 ==="
ls -lh kaggle_pack.npz || echo "⚠️ 还没上传 kaggle_pack.npz"
echo "=== 就绪。下一步 ==="
echo "  python extract.py --model NeoQuasar/Kronos-small   # 提特征, 断点可续"
echo "  python head.py                                     # 训头 + 出指标"

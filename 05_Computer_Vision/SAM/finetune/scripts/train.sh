#!/bin/bash
# ============================================================================
# SAM3 微调训练启动脚本
#
# 使用方法:
#   cd /home/ljz/Jingze/MylearningNote/05_Computer_Vision/SAM/finetune
#   bash scripts/train.sh
#
# 前置条件:
#   1. 已运行 yolo_to_coco.py 生成 COCO JSON
#   2. 已运行 verify_dataset.py 验证数据集
#   3. 已修改 config/table_objects.yaml 中的路径
# ============================================================================

set -euo pipefail

# ==================================================
# 配置区（根据实际情况修改）
# ==================================================

# SAM3 仓库根目录
SAM3_ROOT="/home/ljz/Jingze/MylearningNote/05_Computer_Vision/SAM/sam3"

# 本项目根目录
FINETUNE_ROOT="/home/ljz/Jingze/MylearningNote/05_Computer_Vision/SAM/finetune"

# 配置文件路径
CONFIG_FILE="${FINETUNE_ROOT}/config/table_objects.yaml"

# 实验日志目录（会自动创建）
EXPERIMENT_DIR="${FINETUNE_ROOT}/experiment_logs/$(date +%Y%m%d_%H%M%S)"

# GPU 数量
NUM_GPUS=1

# ==================================================
# 检查前置条件
# ==================================================

echo "=========================================="
echo "SAM3 微调训练"
echo "=========================================="
echo ""

# 检查 SAM3 仓库
if [ ! -d "${SAM3_ROOT}/sam3/train" ]; then
    echo "❌ 未找到 SAM3 训练代码: ${SAM3_ROOT}/sam3/train"
    echo "   请确认 SAM3 仓库路径正确"
    exit 1
fi

# 检查配置文件
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "❌ 未找到配置文件: ${CONFIG_FILE}"
    exit 1
fi

# 检查配置中的路径是否已修改
if grep -q "<YOUR_DATASET_ROOT>" "${CONFIG_FILE}"; then
    echo "❌ 请先修改 config/table_objects.yaml 中的路径："
    echo "   dataset_root: <YOUR_DATASET_ROOT>  →  实际数据集路径"
    echo "   experiment_log_dir: <YOUR_EXPERIMENT_LOG_DIR>  →  ${EXPERIMENT_DIR}"
    echo "   bpe_path: <BPE_PATH>  →  ${SAM3_ROOT}/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    exit 1
fi

# 检查 GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  未检测到 nvidia-smi，训练可能无法运行"
fi

echo "SAM3 仓库:     ${SAM3_ROOT}"
echo "配置文件:       ${CONFIG_FILE}"
echo "实验日志:       ${EXPERIMENT_DIR}"
echo "GPU 数量:       ${NUM_GPUS}"
echo ""

mkdir -p "${EXPERIMENT_DIR}"

# ==================================================
# 启动训练
# ==================================================

echo "开始训练..."
echo ""

cd "${SAM3_ROOT}"

python sam3/train/train.py \
    -c "${CONFIG_FILE}" \
    --use-cluster 0 \
    --num-gpus "${NUM_GPUS}" \
    2>&1 | tee "${EXPERIMENT_DIR}/train.log"

echo ""
echo "训练完成！"
echo "Checkpoint 保存在: ${EXPERIMENT_DIR}/checkpoints/"
echo "Tensorboard 日志:  ${EXPERIMENT_DIR}/tensorboard/"
echo ""
echo "下一步：运行 verify_checkpoint.py 验证 checkpoint"

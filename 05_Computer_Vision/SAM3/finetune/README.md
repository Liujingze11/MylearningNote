# SAM3 微调工具包

基于 SAM3 官方仓库的 image fine-tuning 流程，针对 **固定文本 prompt + 单类别 + 每张图多个实例** 的场景定制。

---

## 目录结构

```
finetune/
├── README.md                           ← 本文件
├── config/
│   └── table_objects.yaml              ← 训练配置（需要修改路径）
├── data/
│   ├── yolo_to_coco.py                 ← Step 1: YOLO-seg → COCO JSON 转换
│   └── verify_dataset.py               ← Step 2: 数据集验证
├── scripts/
│   ├── train.sh                        ← Step 3: 训练启动脚本
│   ├── verify_checkpoint.py            ← Step 4: checkpoint 验证
│   └── inference_test.py               ← Step 5: 推理测试
└── experiment_logs/                    ← 训练日志（自动创建，已 gitignore）
```

---

## 完整流程

### Step 0: 准备数据

你的 YOLO-seg 标注格式：
```
images/
    1006.jpg
    1007.jpg
    ...
labels/
    1006.txt    ← 每行: class_id x1 y1 x2 y2 x3 y3 ...
    1007.txt
    ...
```

---

### Step 1: 转换为 COCO JSON

```bash
cd /home/ljz/Jingze/MylearningNote/05_Computer_Vision/SAM/finetune

# 转换训练集
python data/yolo_to_coco.py \
    --images-dir /path/to/train/images \
    --labels-dir /path/to/train/labels \
    --output /path/to/dataset/train/_annotations.coco.json \
    --prompt "objects on a white table"

# 转换验证集
python data/yolo_to_coco.py \
    --images-dir /path/to/val/images \
    --labels-dir /path/to/val/labels \
    --output /path/to/dataset/val/_annotations.coco.json \
    --prompt "objects on a white table"
```

生成的数据集结构：
```
dataset/
├── train/
│   ├── 1006.jpg
│   ├── 1007.jpg
│   └── _annotations.coco.json
└── val/
    ├── ...
    └── _annotations.coco.json
```

---

### Step 2: 验证数据集

```bash
python data/verify_dataset.py \
    --ann-file /path/to/dataset/train/_annotations.coco.json \
    --visualize 3
```

输出示例：
```
验证数据集: /path/to/_annotations.coco.json
============================================================
  图片数量:   50
  标注数量:   213
  类别数量:   1
  类别 [1]: "objects on a white table"

✅ 数据集验证通过！

错误: 0, 警告: 0
```

---

### Step 3: 修改配置 & 开始训练

**必须修改 `config/table_objects.yaml` 中的三个路径：**

```yaml
paths:
  dataset_root: /path/to/dataset          # ← 改成你的数据集根目录
  experiment_log_dir: /path/to/logs       # ← 改成日志输出目录
  bpe_path: /path/to/sam3/assets/bpe_simple_vocab_16e6.txt.gz
```

然后启动训练：

```bash
bash scripts/train.sh
```

或直接运行（不使用 train.sh）：

```bash
cd /home/ljz/Jingze/MylearningNote/05_Computer_Vision/SAM/sam3

python sam3/train/train.py \
    -c /home/ljz/Jingze/MylearningNote/05_Computer_Vision/SAM/finetune/config/table_objects.yaml \
    --use-cluster 0 \
    --num-gpus 1
```

训练日志会输出到 `experiment_logs/` 目录，包含：
- `checkpoints/checkpoint.pt` — 最终模型权重
- `tensorboard/` — TensorBoard 日志（用 `tensorboard --logdir` 查看）
- `train.log` — 训练日志

---

### Step 4: 验证 Checkpoint

```bash
python scripts/verify_checkpoint.py \
    --checkpoint /path/to/experiment_logs/checkpoints/checkpoint.pt
```

输出示例：
```
验证 checkpoint: .../checkpoint.pt
============================================================
  文件大小: 1832.5 MB
  model 键数量: 847
  segmentation 相关键: 42 个
  backbone: 312 个参数
  transformer: 489 个参数

尝试加载到 SAM3 模型...
  ✅ 完美加载，所有键匹配！
  模型总参数: 1831.2M

✅ checkpoint 验证通过！
```

---

### Step 5: 推理测试

```bash
python scripts/inference_test.py \
    --checkpoint /path/to/checkpoint.pt \
    --image /path/to/test_image.jpg \
    --prompt "objects on a white table" \
    --output-dir ./inference_output
```

输出：
- `original.jpg` — 原图
- `masks_overlay.jpg` — 彩色 Mask 叠加图

---

## 关键配置说明

| 配置项 | 当前值 | 说明 |
|--------|--------|------|
| `scratch.enable_segmentation` | `True` | ★ 必须为 True 才训练 Mask |
| `trainer.skip_saving_ckpts` | `false` | ★ 必须为 False 才保存 checkpoint |
| `scratch.max_data_epochs` | `40` | 训练轮数，可根据数据量调整 |
| `scratch.train_batch_size` | `1` | batch size，显存够可增大 |
| `scratch.resolution` | `1008` | 输入图片分辨率 |

---

## 与你现有 SAM3 测试脚本的关系

你现有的推理脚本（`test/sam3_photo_yolo_mask_loop.py` 等）使用的是官方预训练权重。
微调后，只需在加载模型时替换 checkpoint 路径即可：

```python
# 旧：使用官方预训练权重
# model = build_sam3_image_model(load_from_HF=True)

# 新：使用微调后的权重（手动加载，避免 key 不匹配）
model = build_sam3_image_model(
    checkpoint_path=None,
    load_from_HF=False,
    enable_segmentation=True,
)
ckpt = torch.load("checkpoint.pt", map_location="cpu")
model.load_state_dict(ckpt["model"], strict=False)
```

---

## 常见问题

**Q: 训练时显存不够怎么办？**
A: 将 `scratch.resolution` 从 1008 降到 640，或减小 `scratch.train_batch_size`。

**Q: 训练多少轮合适？**
A: 数据量 < 100 张：20-30 轮；100-500 张：10-20 轮；> 500 张：5-10 轮。观察 val loss 不再下降即可停止。

**Q: 如何修改 prompt？**
A: 两处需要同步修改：`yolo_to_coco.py` 的 `--prompt` 参数，和推理时的 `query_text`。

**Q: checkpoint 加载时报 key 不匹配？**
A: 使用 `scripts/verify_checkpoint.py` 检查，或参考 Step 5 中的手动加载方式。

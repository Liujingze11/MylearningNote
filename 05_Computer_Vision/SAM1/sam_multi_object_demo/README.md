# SAM Multi-Object Interactive Segmentation

一个基于 **Meta SAM 1（Segment Anything Model）** 的多目标交互式分割 Demo。

这个项目的目标不是自动检测所有物体，而是通过人工点击告诉 SAM：

> “当前这些点属于同一个 Object，请把这个 Object 分割出来。”

每个 Object 都拥有自己独立的：

- Object ID
- 显示颜色
- Prompt 点
- Mask
- Score
- SAM refinement logit

因此可以在同一张图片中交互式分割多个独立物体。

---

## 1. 项目结构

```text
sam_multi_object_demo/
│
├── main.py
├── app.py
├── config.py
├── sam_engine.py
├── object_manager.py
├── renderer.py
├── saver.py
│
├── test.png
├── sam_vit_b_01ec64.pth
│
├── output/
│
└── README.md
```

各文件作用：

### `main.py`

程序入口。

只负责创建并启动：

```python
SAMInteractiveApp()
```

### `config.py`

统一保存可修改参数，例如：

- 图片路径
- SAM 权重路径
- 模型类型
- 输出目录
- 显示尺寸
- Mask 透明度
- Object 颜色

以后通常优先改这个文件。

### `sam_engine.py`

SAM 推理模块。

负责：

1. 加载 SAM
2. 加载图片
3. 将图片传入 Image Encoder
4. 根据 Point Prompt 生成 Mask
5. 使用上一轮 logit 做 iterative refinement

其中：

```python
predictor.set_image(...)
```

只执行一次。

后续鼠标点击不会重新执行 Image Encoder，只调用 SAM 的交互式分割部分。

### `object_manager.py`

多 Object 状态管理模块。

负责：

- 新建 Object
- 切换 Object
- 添加点
- 撤销点
- 清空 Object
- 删除 Object

一个 Object 的基本结构：

```python
{
    "id": 1,
    "points": [],
    "mask": None,
    "score": None,
    "logit": None,
    "color": ...
}
```

### `renderer.py`

界面和可视化模块。

负责：

- Mask 半透明显示
- 多 Object 不同颜色
- 当前 Object 白色轮廓
- Prompt 点显示
- 图片缩放
- 显示坐标和原图坐标转换
- 顶部快捷键提示

### `saver.py`

结果保存模块。

按 `S` 后保存：

```text
output/
├── object_001_mask.png
├── object_002_mask.png
├── object_003_mask.png
├── instance_mask.png
├── result_all.jpg
└── objects.json
```

### `app.py`

整个程序的交互控制器。

负责把：

```text
鼠标
键盘
ObjectManager
SAMEngine
Renderer
ResultSaver
```

串起来。

---

## 2. 环境

先进入你的 Conda 环境：

```bash
conda activate sam
```

确认至少可以正常 import：

```python
import cv2
import torch
import numpy
from segment_anything import SamPredictor
```

本项目不会主动安装新的 Python 包。

---

## 3. SAM 权重

当前默认使用：

```text
sam_vit_b_01ec64.pth
```

对应配置：

```python
MODEL_TYPE = "vit_b"
```

如果未来改用：

```text
sam_vit_l_0b3195.pth
```

需要同时修改：

```python
MODEL_TYPE = "vit_l"
```

如果使用：

```text
sam_vit_h_4b8939.pth
```

则：

```python
MODEL_TYPE = "vit_h"
```

---

## 4. 准备图片

把图片放到项目目录，例如：

```text
test.png
```

然后在 `config.py` 中设置：

```python
IMAGE_PATH = "test.png"
```

---

## 5. 运行

```bash
python main.py
```

启动后，SAM 会先：

```text
加载模型
↓
读取图片
↓
执行 Image Encoder
↓
打开交互窗口
```

图片 Embedding 只计算一次。

---

# 6. 基本操作

## 左键：添加 Prompt 点

当前默认：

```text
Object 1
```

例如在杯子上左键点击：

```text
●
```

SAM 会立即分割杯子。

如果结果不够准确，可以继续在杯子的其他区域点击：

```text
●
●
●
```

这些点都会属于：

```text
Object 1
```

SAM 会结合所有点不断修正当前 Mask。


---

## Shift + 左键：删除附近的点

如果某个 Prompt 点点错了，不需要依赖右键。

按住：

```text
Shift
```

然后在错误点附近左键点击：

```text
Shift + 左键
```

程序会：

```text
查找当前 Object 中最近的 Prompt 点
↓
如果距离在 REMOVE_RADIUS 范围内
↓
删除该点
↓
清除旧 SAM logit
↓
使用剩余点立即重新计算当前 Mask
```

默认删除半径在 `config.py` 中设置：

```python
REMOVE_RADIUS = 25
```

这个值表示**显示窗口中的像素距离**，程序会自动换算到原始图片坐标。

这个设计用于避免 Linux / OpenCV GUI 下右键可能与窗口自身行为冲突。

---

## N：新建 Object

当 Object 1 已经满意：

```text
N
```

程序创建：

```text
Object 2
```

并自动换一种颜色。

随后左键点击另一个物体。

例如：

```text
Object 1 = 杯子 = 绿色
Object 2 = 苹果 = 蓝色
Object 3 = 键盘 = 橙色
```

---

## A / D：切换 Object

```text
A = 上一个 Object
D = 下一个 Object
```

例如当前：

```text
Object 3
```

按：

```text
A
```

切回：

```text
Object 2
```

然后仍然可以继续添加 Prompt 点修正 Object 2。

---

## Z：撤销最后一个点

如果刚刚点错：

```text
Z
```

会删除当前 Object 最后添加的一个点。

删除后旧 Mask 的 refinement 信息会被清除，然后根据剩余点重新计算。

---

## C：清空当前 Object

```text
C
```

只清空当前 Object：

```text
points
mask
score
logit
```

其他 Object 不受影响。

---

## X：删除当前 Object

```text
X
```

删除整个当前 Object。

如果当前只剩最后一个 Object，则不会真正删除，而是清空它。

---

## S：保存

程序平时不会自动保存。

只有按：

```text
S
```

才保存当前所有有效 Object。

---

## Q / ESC：退出

```text
Q
```

或者：

```text
ESC
```

退出程序。

---

# 7. 输出文件说明

## 单 Object Mask

例如：

```text
object_001_mask.png
```

是一张二值图：

```text
0   = 背景
255 = Object 1
```

---

## `instance_mask.png`

记录所有 Object ID：

```text
0 = 背景
1 = Object 1
2 = Object 2
3 = Object 3
...
```

这是一个 `uint16` PNG。

后续如果做多目标跟踪、数据标注或 Instance Segmentation，这个文件会比较有价值。

---

## `result_all.jpg`

所有 Object 的可视化结果。

每个 Object 使用不同颜色。

---

## `objects.json`

保存每个 Object 的：

```json
[
    {
        "id": 1,
        "points": [
            [532, 421],
            [560, 445]
        ],
        "score": 0.971
    }
]
```

主要用于保留：

- Object ID
- Prompt 点
- SAM Score

---

# 8. 当前交互逻辑

目前所有鼠标左键都是：

```text
Positive Point
label = 1
```

所以：

```text
同一个 Object 内的多个点
```

含义是：

> 这些点都属于同一个目标。

而不是：

> 每一个点分别代表一个物体。

不同物体通过：

```text
N
```

显式创建新的 Object。

---

# 9. SAM 的 Iterative Refinement

第一次点击 Object 时：

```text
Point
↓
SAM
↓
Mask
```

程序会保存 SAM 输出的：

```text
logit
```

第二次继续点击同一个 Object：

```text
已有 Mask logit
+
新的 Point
+
之前的所有 Point
↓
SAM
↓
修正后的 Mask
```

这样比每次完全从零开始分割更符合 SAM 的交互式使用方式。

---

# 10. 当前项目边界

当前版本主要用于学习和测试 SAM，因此暂时没有加入：

- Negative Point
- Bounding Box Prompt
- 自动分割
- SAM 2 视频跟踪
- 自动保存
- Label 名称
- Object 类别
- 数据集导出
- COCO / YOLO 格式转换

这些功能建议在现有模块基础上逐步添加，而不是重新堆进 `main.py`。

---

# 11. 推荐后续学习顺序

当前版本跑通以后，可以依次增加：

```text
1. Negative Point
      ↓
2. Box Prompt
      ↓
3. Object 命名
      ↓
4. 标注格式导出
      ↓
5. SAM 2 视频传播 / Tracking
```

这样可以比较完整地理解 SAM 从单图交互分割到多目标视频分割的工作方式。

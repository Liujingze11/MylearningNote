import os
import cv2
import torch
import numpy as np

from segment_anything import sam_model_registry, SamPredictor


# ============================================================
# 1. 参数设置
# ============================================================

# 输入图片
IMAGE_PATH = "test.png"

# SAM 权重
CHECKPOINT_PATH = "sam_vit_b_01ec64.pth"

# 模型类型
MODEL_TYPE = "vit_b"

# 输出目录
OUTPUT_DIR = "output"

# 显示窗口最大尺寸
# 图片过大时只缩放显示，不影响 SAM 使用原图推理
MAX_DISPLAY_WIDTH = 1400
MAX_DISPLAY_HEIGHT = 850

# 右键删除点时的判定半径（屏幕像素）
REMOVE_RADIUS = 25

# Mask 透明度
MASK_ALPHA = 0.40

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. 运行状态
# ============================================================

# 当前所有正样本点
# 坐标全部使用【原始图片坐标】
points = []

# 用于 U 撤销
history = []

# 当前 SAM Mask
current_mask = None

# 当前 Mask 分数
current_score = None

# SAM 上一次输出的 logits
# 用于下一次点击时进一步 refinement
previous_logit = None

# 是否需要重新推理
needs_update = False

# 是否强制重新计算
# 删除点 / Undo 后需要重新计算，而不是沿用旧 Mask
force_full_recompute = False

# 状态提示
status_message = "Left click to add point"

# 显示缩放比例
display_scale = 1.0


# ============================================================
# 3. 加载 SAM
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("SAM Interactive Segmentation")
print("=" * 60)
print(f"当前设备：{device}")
print("正在加载 SAM...")


sam = sam_model_registry[MODEL_TYPE](
    checkpoint=CHECKPOINT_PATH
)

sam.to(device=device)

predictor = SamPredictor(sam)

print("SAM 加载完成")


# ============================================================
# 4. 读取图片
# ============================================================

image_bgr = cv2.imread(IMAGE_PATH)

if image_bgr is None:
    raise FileNotFoundError(
        f"找不到图片：{IMAGE_PATH}"
    )

height, width = image_bgr.shape[:2]

print(f"图片尺寸：{width} x {height}")


# SAM 使用 RGB
image_rgb = cv2.cvtColor(
    image_bgr,
    cv2.COLOR_BGR2RGB
)


# ============================================================
# 5. 提取图片特征
# ============================================================

# 这一步最耗时
#
# 但只运行一次。
#
# 后面无论点击多少次，
# 都不会重新跑 Image Encoder。

print("正在提取图片特征...")

predictor.set_image(image_rgb)

print("图片特征提取完成")


# ============================================================
# 6. 计算显示尺寸
# ============================================================

scale_w = MAX_DISPLAY_WIDTH / width
scale_h = MAX_DISPLAY_HEIGHT / height

# 最大只能为 1
# 小图片不会被强行放大
display_scale = min(
    scale_w,
    scale_h,
    1.0
)

display_width = int(
    width * display_scale
)

display_height = int(
    height * display_scale
)

print(
    f"显示尺寸："
    f"{display_width} x {display_height}"
)

print(
    f"显示缩放比例："
    f"{display_scale:.3f}"
)


# ============================================================
# 7. 坐标转换
# ============================================================

def display_to_original(x, y):
    """
    显示窗口坐标
        ↓
    原始图片坐标

    SAM 必须使用原图坐标。
    """

    original_x = int(
        x / display_scale
    )

    original_y = int(
        y / display_scale
    )

    # 防止越界
    original_x = np.clip(
        original_x,
        0,
        width - 1
    )

    original_y = np.clip(
        original_y,
        0,
        height - 1
    )

    return (
        int(original_x),
        int(original_y)
    )


def original_to_display(x, y):
    """
    原始图片坐标
        ↓
    显示窗口坐标
    """

    display_x = int(
        x * display_scale
    )

    display_y = int(
        y * display_scale
    )

    return (
        display_x,
        display_y
    )


# ============================================================
# 8. SAM 推理
# ============================================================

def update_segmentation():
    """
    根据当前 points 重新执行 SAM。

    如果是继续增加点：
        使用 previous_logit 继续 refinement

    如果删除 / Undo：
        清除 previous_logit
        重新根据当前所有点计算
    """

    global current_mask
    global current_score
    global previous_logit
    global force_full_recompute
    global status_message

    # --------------------------------------------------------
    # 没有点
    # --------------------------------------------------------

    if len(points) == 0:

        current_mask = None
        current_score = None
        previous_logit = None

        status_message = "No points"

        return


    # --------------------------------------------------------
    # 构造 SAM Point Prompt
    # --------------------------------------------------------

    input_points = np.array(
        points,
        dtype=np.float32
    )


    # 当前全部是正样本点
    #
    # 1 = foreground
    #
    # 后面如果想增加负点，
    # 这里可以加入 label=0
    input_labels = np.ones(
        len(points),
        dtype=np.int32
    )


    # --------------------------------------------------------
    # 第一次点击
    # --------------------------------------------------------

    if (
        previous_logit is None
        or force_full_recompute
    ):

        # 如果只有一个点
        # SAM 会输出 3 个候选结果
        if len(points) == 1:

            masks, scores, logits = predictor.predict(

                point_coords=input_points,

                point_labels=input_labels,

                multimask_output=True
            )

            best_index = np.argmax(scores)

            current_mask = masks[
                best_index
            ]

            current_score = float(
                scores[best_index]
            )

            previous_logit = logits[
                best_index
            ]


        # 多个点时
        # SAM 意图已经更加明确
        else:

            masks, scores, logits = predictor.predict(

                point_coords=input_points,

                point_labels=input_labels,

                multimask_output=False
            )

            current_mask = masks[0]

            current_score = float(
                scores[0]
            )

            previous_logit = logits[0]


    # --------------------------------------------------------
    # 继续左键添加点
    #
    # 使用上一轮 Mask logits 继续 Refinement
    # --------------------------------------------------------

    else:

        masks, scores, logits = predictor.predict(

            point_coords=input_points,

            point_labels=input_labels,

            # SAM 官方支持的 iterative refinement
            mask_input=previous_logit[
                None,
                :, :
            ],

            multimask_output=False
        )

        current_mask = masks[0]

        current_score = float(
            scores[0]
        )

        previous_logit = logits[0]


    force_full_recompute = False

    status_message = (
        f"Updated | "
        f"Points: {len(points)} | "
        f"Score: {current_score:.3f}"
    )

    print(
        f"SAM 更新完成 | "
        f"Points={len(points)} | "
        f"Score={current_score:.4f}"
    )


# ============================================================
# 9. 创建完整分割可视化
# ============================================================

def create_result_image(
    draw_points=True
):
    """
    生成：
        原图
        +
        半透明 Mask
        +
        Prompt 点
    """

    result = image_bgr.copy()


    # --------------------------------------------------------
    # Mask
    # --------------------------------------------------------

    if current_mask is not None:

        overlay = image_bgr.copy()

        # Mask 区域绿色
        overlay[
            current_mask
        ] = (
            0,
            255,
            0
        )

        result = cv2.addWeighted(

            image_bgr,
            1.0 - MASK_ALPHA,

            overlay,
            MASK_ALPHA,

            0
        )


    # --------------------------------------------------------
    # Prompt Points
    # --------------------------------------------------------

    if draw_points:

        for x, y in points:

            # 外圈白色
            cv2.circle(
                result,
                (x, y),
                radius=8,
                color=(255, 255, 255),
                thickness=-1
            )

            # 内圈红色
            cv2.circle(
                result,
                (x, y),
                radius=5,
                color=(0, 0, 255),
                thickness=-1
            )


    return result


# ============================================================
# 10. 创建显示画面
# ============================================================

def create_display_image():

    # 先生成原尺寸结果
    result = create_result_image(
        draw_points=True
    )


    # 缩放到屏幕尺寸
    if display_scale != 1.0:

        result = cv2.resize(
            result,
            (
                display_width,
                display_height
            ),
            interpolation=cv2.INTER_AREA
        )


    # ========================================================
    # UI 提示信息
    # ========================================================

    # 黑色背景
    cv2.rectangle(
        result,
        (0, 0),
        (display_width, 75),
        (0, 0, 0),
        thickness=-1
    )


    # 第一行
    info1 = (
        f"Points: {len(points)}"
    )

    if current_score is not None:

        info1 += (
            f"   Score: "
            f"{current_score:.3f}"
        )


    cv2.putText(
        result,
        info1,
        (15, 28),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2,

        cv2.LINE_AA
    )


    # 第二行快捷键
    info2 = (
        "Left:Add  Right:Remove  "
        "U:Undo  C:Clear  "
        "S:Save  Q/ESC:Quit"
    )


    cv2.putText(
        result,
        info2,
        (15, 57),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.52,

        (200, 200, 200),

        1,

        cv2.LINE_AA
    )


    return result


# ============================================================
# 11. 鼠标事件
# ============================================================

def mouse_callback(
    event,
    x,
    y,
    flags,
    param
):

    global points
    global previous_logit
    global needs_update
    global force_full_recompute
    global status_message


    # ========================================================
    # 左键
    #
    # 添加正样本点
    # ========================================================

    if event == cv2.EVENT_LBUTTONDOWN:

        original_x, original_y = (
            display_to_original(
                x,
                y
            )
        )


        # 保存当前状态
        # 用于 Undo
        history.append(
            points.copy()
        )


        # 添加新点
        points.append(
            (
                original_x,
                original_y
            )
        )


        print()
        print(
            f"添加点："
            f"({original_x}, {original_y})"
        )


        # 告诉主循环：
        # 需要重新跑 SAM
        needs_update = True

        status_message = "Updating..."


    # ========================================================
    # 右键
    #
    # 删除附近最近的点
    # ========================================================

    elif event == cv2.EVENT_RBUTTONDOWN:

        if len(points) == 0:
            return


        # ----------------------------------------------------
        # 在显示坐标中计算距离
        # ----------------------------------------------------

        distances = []

        for px, py in points:

            dx, dy = original_to_display(
                px,
                py
            )

            distance = np.sqrt(
                (dx - x) ** 2
                +
                (dy - y) ** 2
            )

            distances.append(
                distance
            )


        nearest_index = int(
            np.argmin(distances)
        )

        nearest_distance = distances[
            nearest_index
        ]


        # ----------------------------------------------------
        # 只有足够靠近才删除
        # ----------------------------------------------------

        if nearest_distance <= REMOVE_RADIUS:

            history.append(
                points.copy()
            )


            removed_point = points.pop(
                nearest_index
            )


            print()
            print(
                f"删除点："
                f"{removed_point}"
            )


            # 删除点后旧 Mask 已经包含被删除点的信息
            #
            # 因此不能继续使用 previous_logit
            previous_logit = None

            force_full_recompute = True

            needs_update = True

            status_message = "Updating..."


# ============================================================
# 12. 保存结果
# ============================================================

def save_result():

    if current_mask is None:

        print()
        print("当前没有 Mask，无法保存。")

        return


    # ========================================================
    # 保存二值 Mask
    # ========================================================

    mask_image = (
        current_mask.astype(
            np.uint8
        )
        * 255
    )


    mask_path = os.path.join(
        OUTPUT_DIR,
        "mask.png"
    )


    cv2.imwrite(
        mask_path,
        mask_image
    )


    # ========================================================
    # 保存可视化结果
    # ========================================================

    # 保存原始分辨率
    result = create_result_image(
        draw_points=True
    )


    result_path = os.path.join(
        OUTPUT_DIR,
        "result.jpg"
    )


    cv2.imwrite(
        result_path,
        result
    )


    print()
    print("=" * 50)
    print("保存完成")
    print(f"Mask ：{mask_path}")
    print(f"Result：{result_path}")
    print("=" * 50)


# ============================================================
# 13. OpenCV 窗口
# ============================================================

WINDOW_NAME = "SAM Interactive"

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_AUTOSIZE
)

cv2.setMouseCallback(
    WINDOW_NAME,
    mouse_callback
)


print()
print("=" * 60)
print("操作说明")
print("=" * 60)
print("左键      ：添加分割点")
print("右键      ：删除附近的点")
print("U         ：撤销")
print("C         ：清空")
print("S         ：保存")
print("Q / ESC   ：退出")
print("=" * 60)


# ============================================================
# 14. 主循环
# ============================================================

while True:

    # ========================================================
    # 有新的鼠标操作
    #
    # 执行 SAM
    # ========================================================

    if needs_update:

        # 先显示 Updating
        display = create_display_image()

        cv2.imshow(
            WINDOW_NAME,
            display
        )

        cv2.waitKey(1)


        # 真正执行推理
        update_segmentation()

        needs_update = False


    # ========================================================
    # 显示当前画面
    # ========================================================

    display = create_display_image()

    cv2.imshow(
        WINDOW_NAME,
        display
    )


    key = cv2.waitKey(10) & 0xFF


    # ========================================================
    # S：保存
    # ========================================================

    if key in (
        ord("s"),
        ord("S")
    ):

        save_result()


    # ========================================================
    # C：清空
    # ========================================================

    elif key in (
        ord("c"),
        ord("C")
    ):

        if len(points) > 0:

            history.append(
                points.copy()
            )


        points = []

        current_mask = None

        current_score = None

        previous_logit = None

        force_full_recompute = False

        needs_update = False

        status_message = "Cleared"


        print()
        print("已清空所有点")


    # ========================================================
    # U：撤销
    # ========================================================

    elif key in (
        ord("u"),
        ord("U")
    ):

        if len(history) > 0:

            points = history.pop()


            # Undo 后不能继续使用旧 logits
            previous_logit = None

            force_full_recompute = True


            if len(points) > 0:

                needs_update = True

            else:

                current_mask = None

                current_score = None

                needs_update = False


            print()
            print(
                f"撤销完成，"
                f"当前点数量：{len(points)}"
            )


    # ========================================================
    # Q / ESC：退出
    # ========================================================

    elif (
        key == 27
        or
        key in (
            ord("q"),
            ord("Q")
        )
    ):

        print()
        print("退出 SAM")

        break


# ============================================================
# 15. 关闭窗口
# ============================================================

cv2.destroyAllWindows()
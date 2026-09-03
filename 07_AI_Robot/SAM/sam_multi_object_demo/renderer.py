import cv2
import numpy as np

from config import (
    MASK_ALPHA,
    MAX_DISPLAY_WIDTH,
    MAX_DISPLAY_HEIGHT,
)


class Renderer:
    """
    负责：
    - 显示缩放
    - 坐标转换
    - Mask 可视化
    - 当前 Object 高亮
    - UI 提示
    """

    def __init__(self, image_bgr):
        self.image_bgr = image_bgr

        self.height, self.width = (
            image_bgr.shape[:2]
        )

        scale_w = (
            MAX_DISPLAY_WIDTH / self.width
        )

        scale_h = (
            MAX_DISPLAY_HEIGHT / self.height
        )

        self.display_scale = min(
            scale_w,
            scale_h,
            1.0
        )

        self.display_width = int(
            self.width * self.display_scale
        )

        self.display_height = int(
            self.height * self.display_scale
        )

        print(
            f"显示尺寸："
            f"{self.display_width} x "
            f"{self.display_height}"
        )

    def display_to_original(self, x, y):
        """
        窗口坐标 -> 原图坐标
        """

        ox = int(
            x / self.display_scale
        )

        oy = int(
            y / self.display_scale
        )

        ox = np.clip(
            ox,
            0,
            self.width - 1
        )

        oy = np.clip(
            oy,
            0,
            self.height - 1
        )

        return int(ox), int(oy)

    def create_result_image(
        self,
        objects,
        current_index,
        draw_ui=True
    ):
        """
        绘制所有 Object 的 Mask 和点击点。
        """

        result = self.image_bgr.copy()

        # 所有 Object Mask
        for obj in objects:
            mask = obj["mask"]

            if mask is None:
                continue

            overlay = result.copy()
            overlay[mask] = obj["color"]

            result = cv2.addWeighted(
                result,
                1.0 - MASK_ALPHA,
                overlay,
                MASK_ALPHA,
                0
            )

        # 当前 Object 白色轮廓
        current_obj = objects[
            current_index
        ]

        if current_obj["mask"] is not None:
            mask_uint8 = (
                current_obj["mask"].astype(
                    np.uint8
                )
                * 255
            )

            contours, _ = cv2.findContours(
                mask_uint8,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            cv2.drawContours(
                result,
                contours,
                -1,
                (255, 255, 255),
                2
            )

        # 所有点击点
        for index, obj in enumerate(objects):
            for x, y in obj["points"]:

                if index == current_index:
                    # 当前 Object：白色外圈
                    cv2.circle(
                        result,
                        (x, y),
                        9,
                        (255, 255, 255),
                        -1
                    )

                    cv2.circle(
                        result,
                        (x, y),
                        6,
                        obj["color"],
                        -1
                    )

                else:
                    cv2.circle(
                        result,
                        (x, y),
                        5,
                        obj["color"],
                        -1
                    )

        if draw_ui:
            self._draw_ui(
                result,
                objects,
                current_index
            )

        return result

    def _draw_ui(
        self,
        image,
        objects,
        current_index
    ):
        obj = objects[
            current_index
        ]

        cv2.rectangle(
            image,
            (0, 0),
            (self.width, 95),
            (0, 0, 0),
            -1
        )

        info = (
            f"Current Object: {obj['id']}"
            f"   Points: {len(obj['points'])}"
            f"   Total Objects: {len(objects)}"
        )

        if obj["score"] is not None:
            info += (
                f"   Score: {obj['score']:.3f}"
            )

        cv2.putText(
            image,
            info,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # 当前 Object 颜色
        cv2.circle(
            image,
            (20, 60),
            8,
            obj["color"],
            -1
        )

        cv2.putText(
            image,
            "Current Object Color",
            (38, 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (220, 220, 220),
            1,
            cv2.LINE_AA
        )

        shortcuts = (
            "Left:Add | Shift+Left:Remove | N:New | "
            "A/D:Switch | Z:Undo | C:Clear | "
            "X:Delete | S:Save | Q:Quit"
        )

        cv2.putText(
            image,
            shortcuts,
            (260, 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )

    def create_display_image(
        self,
        objects,
        current_index
    ):
        result = self.create_result_image(
            objects,
            current_index,
            draw_ui=True
        )

        if self.display_scale != 1.0:
            result = cv2.resize(
                result,
                (
                    self.display_width,
                    self.display_height
                ),
                interpolation=cv2.INTER_AREA
            )

        return result

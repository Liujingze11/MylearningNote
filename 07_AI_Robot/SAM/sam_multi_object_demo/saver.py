import os
import json
import cv2
import numpy as np


class ResultSaver:
    """
    负责保存：
    1. 每个 Object 的单独二值 Mask
    2. 总 Instance Mask
    3. 所有 Object 的可视化结果
    4. 点击点和 score 的 JSON
    """

    def __init__(
        self,
        output_dir,
        image_shape,
        renderer
    ):
        self.output_dir = output_dir
        self.height, self.width = (
            image_shape[:2]
        )
        self.renderer = renderer

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

    def save_all(
        self,
        objects,
        current_index
    ):
        valid_objects = [
            obj
            for obj in objects
            if obj["mask"] is not None
        ]

        if len(valid_objects) == 0:
            print(
                "当前没有任何可保存的 Object。"
            )
            return

        print()
        print("=" * 60)
        print("正在保存全部 Object...")

        # 清理旧的单 Object Mask
        for filename in os.listdir(
            self.output_dir
        ):
            if (
                filename.startswith(
                    "object_"
                )
                and filename.endswith(
                    "_mask.png"
                )
            ):
                os.remove(
                    os.path.join(
                        self.output_dir,
                        filename
                    )
                )

        # uint16 足够保存较多 Object ID
        instance_mask = np.zeros(
            (
                self.height,
                self.width
            ),
            dtype=np.uint16
        )

        metadata = []

        for obj in valid_objects:
            object_id = obj["id"]
            mask = obj["mask"]

            # 单 Object 二值 Mask
            binary_mask = (
                mask.astype(np.uint8)
                * 255
            )

            mask_path = os.path.join(
                self.output_dir,
                f"object_{object_id:03d}_mask.png"
            )

            cv2.imwrite(
                mask_path,
                binary_mask
            )

            # Instance Mask
            instance_mask[
                mask
            ] = object_id

            metadata.append(
                {
                    "id": object_id,
                    "points": [
                        [int(x), int(y)]
                        for x, y
                        in obj["points"]
                    ],
                    "score": obj["score"],
                }
            )

            print(
                f"Object {object_id}"
                f" -> {mask_path}"
            )

        # 总 Instance Mask
        instance_path = os.path.join(
            self.output_dir,
            "instance_mask.png"
        )

        cv2.imwrite(
            instance_path,
            instance_mask
        )

        # 总可视化结果
        result = (
            self.renderer.create_result_image(
                objects,
                current_index,
                draw_ui=False
            )
        )

        result_path = os.path.join(
            self.output_dir,
            "result_all.jpg"
        )

        cv2.imwrite(
            result_path,
            result
        )

        # Object 信息
        json_path = os.path.join(
            self.output_dir,
            "objects.json"
        )

        with open(
            json_path,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                metadata,
                f,
                ensure_ascii=False,
                indent=4
            )

        print()
        print("保存完成：")
        print(
            f"总结果：{result_path}"
        )
        print(
            f"Instance Mask：{instance_path}"
        )
        print(
            f"Object 信息：{json_path}"
        )
        print("=" * 60)

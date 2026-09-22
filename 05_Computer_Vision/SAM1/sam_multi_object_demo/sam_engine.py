import cv2
import torch
import numpy as np

from segment_anything import sam_model_registry, SamPredictor


class SAMEngine:
    """
    负责：
    1. 加载 SAM
    2. 加载图片
    3. 提取 Image Embedding
    4. 根据点 Prompt 更新指定 Object 的 Mask
    """

    def __init__(self, image_path, checkpoint_path, model_type):
        self.image_path = image_path
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("=" * 60)
        print("SAM Multi-Object Interactive Segmentation")
        print("=" * 60)
        print(f"当前设备：{self.device}")
        print("正在加载 SAM...")

        sam = sam_model_registry[self.model_type](
            checkpoint=self.checkpoint_path
        )
        sam.to(device=self.device)

        self.predictor = SamPredictor(sam)

        print("SAM 加载完成")

        self.image_bgr = cv2.imread(self.image_path)

        if self.image_bgr is None:
            raise FileNotFoundError(
                f"找不到图片：{self.image_path}"
            )

        self.height, self.width = self.image_bgr.shape[:2]

        print(
            f"图片尺寸：{self.width} x {self.height}"
        )

        self.image_rgb = cv2.cvtColor(
            self.image_bgr,
            cv2.COLOR_BGR2RGB
        )

        print("正在提取图片特征...")

        # 整张图片只做一次 Image Encoder
        with torch.inference_mode():
            self.predictor.set_image(self.image_rgb)

        print("图片特征提取完成")

    def update_object(self, obj, full_recompute=False):
        """
        根据当前 Object 的全部正样本点更新 Mask。

        obj 格式：
        {
            "points": [(x, y), ...],
            "mask": ...,
            "score": ...,
            "logit": ...
        }

        full_recompute=True:
            不使用旧 logit，完全根据当前点重新计算。
            常用于撤销点之后。
        """

        points = obj["points"]

        if len(points) == 0:
            obj["mask"] = None
            obj["score"] = None
            obj["logit"] = None
            return

        input_points = np.array(
            points,
            dtype=np.float32
        )

        # 当前所有点均为正样本点
        input_labels = np.ones(
            len(points),
            dtype=np.int32
        )

        # 首次推理 / 强制重新计算
        if full_recompute or obj["logit"] is None:

            # 单点时让 SAM 返回 3 个候选 Mask
            if len(points) == 1:
                with torch.inference_mode():
                    masks, scores, logits = self.predictor.predict(
                        point_coords=input_points,
                        point_labels=input_labels,
                        multimask_output=True
                    )

                best_index = int(np.argmax(scores))

                obj["mask"] = masks[best_index]
                obj["score"] = float(scores[best_index])
                obj["logit"] = logits[best_index]

            # 多点时意图相对明确
            else:
                with torch.inference_mode():
                    masks, scores, logits = self.predictor.predict(
                        point_coords=input_points,
                        point_labels=input_labels,
                        multimask_output=False
                    )

                obj["mask"] = masks[0]
                obj["score"] = float(scores[0])
                obj["logit"] = logits[0]

        # 在上一轮 Mask 上继续 refinement
        else:
            with torch.inference_mode():
                masks, scores, logits = self.predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    mask_input=obj["logit"][None, :, :],
                    multimask_output=False
                )

            obj["mask"] = masks[0]
            obj["score"] = float(scores[0])
            obj["logit"] = logits[0]

        print(
            f"Object {obj['id']} 更新"
            f" | Points={len(points)}"
            f" | Score={obj['score']:.4f}"
        )

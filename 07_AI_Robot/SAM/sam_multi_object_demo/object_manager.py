import numpy as np

from config import COLORS


def create_object(object_id):
    """
    创建一个新的 Object。
    一个 Object = 一种颜色 + 一组点 + 一个 Mask。
    """

    color = COLORS[
        (object_id - 1) % len(COLORS)
    ]

    return {
        "id": object_id,
        "points": [],
        "mask": None,
        "score": None,
        "logit": None,
        "color": color,
    }


class ObjectManager:
    """
    负责管理所有 Object：
    - 新建
    - 切换
    - 添加点
    - 撤销点
    - 清空
    - 删除
    """

    def __init__(self):
        self.objects = [
            create_object(1)
        ]

        self.current_index = 0

    def current(self):
        return self.objects[
            self.current_index
        ]

    def add_point(self, x, y):
        obj = self.current()

        obj["points"].append(
            (int(x), int(y))
        )

        return obj

    def new_object(self):
        current_obj = self.current()

        # 当前 Object 还是空的，不重复创建
        if len(current_obj["points"]) == 0:
            print(
                "当前 Object 还没有任何点，"
                "无需创建新的 Object。"
            )
            return None

        new_id = max(
            obj["id"]
            for obj in self.objects
        ) + 1

        new_obj = create_object(new_id)

        self.objects.append(new_obj)

        self.current_index = (
            len(self.objects) - 1
        )

        print()
        print("=" * 40)
        print(f"进入 Object {new_id}")
        print("=" * 40)

        return new_obj

    def switch(self, direction):
        if len(self.objects) <= 1:
            return self.current()

        self.current_index += direction
        self.current_index %= len(
            self.objects
        )

        obj = self.current()

        print()
        print(
            f"当前切换到 Object {obj['id']}"
        )

        return obj

    def undo_point(self):
        obj = self.current()

        if len(obj["points"]) == 0:
            print(
                "当前 Object 没有点可以撤销。"
            )
            return None

        removed = obj["points"].pop()

        # 旧 logit 已经包含被撤销点的信息
        obj["logit"] = None

        if len(obj["points"]) == 0:
            obj["mask"] = None
            obj["score"] = None

        print(
            f"Object {obj['id']} "
            f"撤销点：{removed}"
        )

        return removed


    def remove_nearest_point(
        self,
        x,
        y,
        radius
    ):
        """
        删除当前 Object 中距离 (x, y) 最近的点。

        只有最近点与点击位置的距离 <= radius 时才删除。

        返回：
            被删除的点 (x, y)
            如果范围内没有点，则返回 None
        """

        obj = self.current()

        if len(obj["points"]) == 0:
            print(
                "当前 Object 没有点可以删除。"
            )
            return None

        points_array = np.array(
            obj["points"],
            dtype=np.float32
        )

        target = np.array(
            [x, y],
            dtype=np.float32
        )

        distances = np.linalg.norm(
            points_array - target,
            axis=1
        )

        nearest_index = int(
            np.argmin(distances)
        )

        nearest_distance = float(
            distances[nearest_index]
        )

        if nearest_distance > radius:
            print(
                f"附近 {radius:.1f}px 范围内"
                "没有可删除的点。"
            )
            return None

        removed = obj["points"].pop(
            nearest_index
        )

        # 旧 logit 中已经包含被删除点的信息，
        # 删除点后必须重新计算。
        obj["logit"] = None

        if len(obj["points"]) == 0:
            obj["mask"] = None
            obj["score"] = None

        print(
            f"Object {obj['id']} "
            f"删除点：{removed}"
        )

        return removed

    def clear_current(self):
        obj = self.current()

        obj["points"] = []
        obj["mask"] = None
        obj["score"] = None
        obj["logit"] = None

        print(
            f"Object {obj['id']} 已清空"
        )

    def delete_current(self):
        # 至少保留一个 Object
        if len(self.objects) == 1:
            self.clear_current()
            return

        deleted = self.objects.pop(
            self.current_index
        )

        print(
            f"已删除 Object {deleted['id']}"
        )

        if self.current_index >= len(
            self.objects
        ):
            self.current_index = (
                len(self.objects) - 1
            )

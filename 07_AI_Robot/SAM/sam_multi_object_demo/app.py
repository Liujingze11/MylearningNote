import cv2

from config import (
    IMAGE_PATH,
    CHECKPOINT_PATH,
    MODEL_TYPE,
    OUTPUT_DIR,
    REMOVE_RADIUS,
)
from sam_engine import SAMEngine
from object_manager import ObjectManager
from renderer import Renderer
from saver import ResultSaver


class SAMInteractiveApp:
    """
    SAM 多 Object 交互应用主控制器。

    负责：
    - 初始化各模块
    - 处理鼠标事件
    - 处理键盘事件
    - 控制实时推理
    """

    WINDOW_NAME = "SAM Multi Object"

    def __init__(self):
        # SAM / 图片
        self.sam_engine = SAMEngine(
            image_path=IMAGE_PATH,
            checkpoint_path=CHECKPOINT_PATH,
            model_type=MODEL_TYPE
        )

        # Object 管理
        self.object_manager = (
            ObjectManager()
        )

        # 显示
        self.renderer = Renderer(
            self.sam_engine.image_bgr
        )

        # 保存
        self.saver = ResultSaver(
            output_dir=OUTPUT_DIR,
            image_shape=(
                self.sam_engine.image_bgr.shape
            ),
            renderer=self.renderer
        )

        # 交互状态
        self.needs_update = False
        self.force_full_recompute = False

    def mouse_callback(
        self,
        event,
        x,
        y,
        flags,
        param
    ):
        """
        鼠标交互：

        左键：
            给当前 Object 添加正样本点，
            并立即触发 SAM 更新。

        Shift + 左键：
            删除当前 Object 中距离点击位置最近的点，
            删除后立即使用剩余点重新计算 Mask。
        """

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        ox, oy = (
            self.renderer.display_to_original(
                x,
                y
            )
        )

        # ====================================================
        # Shift + 左键：删除附近最近点
        # ====================================================

        if flags & cv2.EVENT_FLAG_SHIFTKEY:

            # REMOVE_RADIUS 配置的是显示窗口像素，
            # 这里换算成原图像素。
            original_radius = (
                REMOVE_RADIUS
                / self.renderer.display_scale
            )

            removed = (
                self.object_manager.remove_nearest_point(
                    ox,
                    oy,
                    original_radius
                )
            )

            if removed is None:
                return

            obj = (
                self.object_manager.current()
            )

            # 删除后旧 logit 不能继续使用。
            self.force_full_recompute = True

            # 如果仍有点，立即重新推理；
            # 如果已经没有点，ObjectManager 已清空 Mask。
            self.needs_update = (
                len(obj["points"]) > 0
            )

            return

        # ====================================================
        # 普通左键：添加正样本点
        # ====================================================

        obj = (
            self.object_manager.add_point(
                ox,
                oy
            )
        )

        print()
        print(
            f"Object {obj['id']} "
            f"添加点：({ox}, {oy})"
        )

        self.force_full_recompute = False
        self.needs_update = True

    def update_segmentation(self):
        obj = (
            self.object_manager.current()
        )

        self.sam_engine.update_object(
            obj,
            full_recompute=(
                self.force_full_recompute
            )
        )

        self.needs_update = False
        self.force_full_recompute = False

    def print_help(self):
        print()
        print("=" * 60)
        print("操作说明")
        print("=" * 60)
        print("左键        ：给当前 Object 添加点")
        print("Shift+左键  ：删除当前 Object 附近最近的点")
        print("N      ：新建下一个 Object")
        print("A      ：切换到上一个 Object")
        print("D      ：切换到下一个 Object")
        print("Z      ：撤销当前 Object 最后一个点")
        print("C      ：清空当前 Object")
        print("X      ：删除当前 Object")
        print("S      ：保存全部 Object")
        print("Q/ESC  ：退出")
        print("=" * 60)

    def handle_key(self, key):
        """
        返回 False 表示退出程序。
        """

        # N：新建 Object
        if key in (
            ord("n"),
            ord("N")
        ):
            self.object_manager.new_object()

        # A：上一个 Object
        elif key in (
            ord("a"),
            ord("A")
        ):
            self.object_manager.switch(-1)

        # D：下一个 Object
        elif key in (
            ord("d"),
            ord("D")
        ):
            self.object_manager.switch(1)

        # Z：撤销当前 Object 最后一个点
        elif key in (
            ord("z"),
            ord("Z")
        ):
            removed = (
                self.object_manager.undo_point()
            )

            obj = (
                self.object_manager.current()
            )

            if (
                removed is not None
                and len(obj["points"]) > 0
            ):
                self.force_full_recompute = True
                self.needs_update = True

        # C：清空当前 Object
        elif key in (
            ord("c"),
            ord("C")
        ):
            self.object_manager.clear_current()

        # X：删除当前 Object
        elif key in (
            ord("x"),
            ord("X")
        ):
            self.object_manager.delete_current()

        # S：保存全部
        elif key in (
            ord("s"),
            ord("S")
        ):
            self.saver.save_all(
                self.object_manager.objects,
                self.object_manager.current_index
            )

        # Q / ESC：退出
        elif (
            key == 27
            or key in (
                ord("q"),
                ord("Q")
            )
        ):
            print()
            print("退出 SAM")
            return False

        return True

    def run(self):
        cv2.namedWindow(
            self.WINDOW_NAME,
            cv2.WINDOW_AUTOSIZE
        )

        cv2.setMouseCallback(
            self.WINDOW_NAME,
            self.mouse_callback
        )

        self.print_help()

        running = True

        while running:

            # 有新点 / 撤销操作时更新 SAM
            if self.needs_update:
                # 先刷新一次，让用户看到点已加入
                display = (
                    self.renderer.create_display_image(
                        self.object_manager.objects,
                        self.object_manager.current_index
                    )
                )

                cv2.imshow(
                    self.WINDOW_NAME,
                    display
                )

                cv2.waitKey(1)

                self.update_segmentation()

            # 正常刷新显示
            display = (
                self.renderer.create_display_image(
                    self.object_manager.objects,
                    self.object_manager.current_index
                )
            )

            cv2.imshow(
                self.WINDOW_NAME,
                display
            )

            key = cv2.waitKey(10) & 0xFF

            if key != 255:
                running = self.handle_key(key)

        cv2.destroyAllWindows()

#!/usr/bin/env python3
"""
让一只海龟追另一只海龟

演示了 ROS2 中一个节点同时做 Subscriber + Publisher 的经典模式：
  1. 订阅 /turtle1/pose（获取主海龟的位置）
  2. 发布 /turtle2/cmd_vel（控制追捕海龟的速度）

运行方式：
    # 终端1: 启动小海龟
    ros2 run turtlesim turtlesim_node

    # 终端2: 生成第二只海龟
    ros2 service call /spawn turtlesim/srv/Spawn "{x: 8.0, y: 8.0, theta: 0.0, name: 'turtle2'}"

    # 终端3: 运行追捕程序
    python3 06_turtle_follower.py

    # 终端4: 手动控制第一只海龟跑
    ros2 run turtlesim turtle_teleop_key

看看 turtle2 会不会追上来！

--- 核心概念 ---
闭环控制：读取传感器数据 → 计算 → 发出控制指令 → 读取传感器数据 → ...
这里是最简单的 P 控制器：速度 = 比例系数 × 距离误差
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist      # 速度指令
from turtlesim.msg import Pose           # 海龟的位置（x, y, theta）
import math


class TurtleFollower(Node):
    """
    同时扮演两个角色：
      Subscriber  →  订阅 /turtle1/pose 获取目标位置
      Publisher   →  发布 /turtle2/cmd_vel 控制追捕海龟

    控制算法：简单的 P 控制器（比例控制）
      - 距离越远，追得越快
      - 方向偏差越大，转得越快
    """

    def __init__(self):
        super().__init__('turtle_follower')

        # ---- 订阅者：获取 turtle1 的位置 ----
        self.target_pose = None
        self.create_subscription(
            Pose,
            '/turtle1/pose',
            self.target_pose_callback,
            10
        )

        # ---- 订阅者：获取 turtle2 自己的位置 ----
        self.self_pose = None
        self.create_subscription(
            Pose,
            '/turtle2/pose',
            self.self_pose_callback,
            10
        )

        # ---- 发布者：控制 turtle2 的速度 ----
        self.cmd_publisher = self.create_publisher(
            Twist,
            '/turtle2/cmd_vel',
            10
        )

        # ---- 定时器：每 0.1 秒计算一次追捕指令 ----
        self.timer = self.create_timer(0.1, self.control_loop)

        # ---- P 控制器的比例系数 ----
        self.K_linear = 1.5   # 线速度系数：距离越远越快
        self.K_angular = 4.0  # 角速度系数：角度偏差越大转得越快

        self.get_logger().info('海龟追捕者已就绪！用键盘控制 turtle1 跑起来吧~')

    def target_pose_callback(self, msg):
        """收到 turtle1 的新位置"""
        self.target_pose = msg

    def self_pose_callback(self, msg):
        """收到 turtle2 自己的新位置"""
        self.self_pose = msg

    def control_loop(self):
        """
        核心控制循环：每 0.1 秒执行一次

        步骤：
          1. 计算 turtle2 到 turtle1 的 距离 和 方向角
          2. 用 P 控制器把误差转成速度指令
          3. 发布速度指令
        """
        if self.target_pose is None or self.self_pose is None:
            return  # 还没收到位置数据，等待

        # ---- 1. 计算距离和角度 ----
        dx = self.target_pose.x - self.self_pose.x
        dy = self.target_pose.y - self.self_pose.y
        distance = math.sqrt(dx**2 + dy**2)

        # 目标方向角（从 turtle2 指向 turtle1）
        target_angle = math.atan2(dy, dx)

        # 当前 turtle2 的朝向
        current_angle = self.self_pose.theta

        # 角度误差（归一化到 [-π, π]）
        angle_error = target_angle - current_angle
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))

        # ---- 2. P 控制器 ----
        # 线速度 = 比例系数 × 距离（但要设上限）
        linear_speed = self.K_linear * distance
        linear_speed = min(linear_speed, 5.0)  # 最快不超过 5 m/s

        # 角速度 = 比例系数 × 角度误差
        angular_speed = self.K_angular * angle_error

        # ---- 3. 发布速度指令 ----
        twist = Twist()
        twist.linear.x = linear_speed
        twist.angular.z = angular_speed
        self.cmd_publisher.publish(twist)

        # 每 1 秒打印一次状态（避免刷屏）
        if int(self.get_clock().now().nanoseconds / 1e9) % 1 == 0:
            self.get_logger().info(
                f'距离={distance:.2f}m | '
                f'角度偏差={angle_error:.2f}rad | '
                f'速度={linear_speed:.2f}m/s'
            )


def main(args=None):
    rclpy.init(args=args)
    node = TurtleFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

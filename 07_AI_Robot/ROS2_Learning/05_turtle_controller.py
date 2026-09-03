#!/usr/bin/env python3
"""
控制小海龟：画圆、画方形、画螺旋线

运行方式：
    # 终端1: 先启动小海龟
    ros2 run turtlesim turtlesim_node

    # 终端2: 运行控制器
    python3 05_turtle_controller.py

--- 核心概念 ---
这个脚本展示了"用代码代替键盘"来控制海龟：
  - 键盘遥控：你在终端按 ↑，teleop_key 节点发布 Twist 消息
  - 代码遥控：这个脚本创建一个 Publisher，发布 Twist 消息到 /turtle1/cmd_vel

这就是 ROS2 的核心价值：
  你可以写一个 Python 节点来"假装"是键盘，实现自动化控制。
  海龟本身（turtlesim_node）完全不需要改动！
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist   # 速度消息：包含线速度和角速度
import math
import time


class TurtleController(Node):
    """
    小海龟控制器 — 发布速度指令让海龟按规律运动

    发布的消息类型：geometry_msgs/msg/Twist
      Twist.linear.x   → 前进/后退速度 (m/s)
      Twist.angular.z  → 旋转速度 (rad/s)，正=逆时针，负=顺时针
    """

    def __init__(self):
        super().__init__('turtle_controller')

        # 创建 Publisher：往 /turtle1/cmd_vel 发 Twist 消息
        self.cmd_publisher = self.create_publisher(
            Twist,
            '/turtle1/cmd_vel',   # 小海龟的速度指令 Topic
            10
        )
        self.get_logger().info('海龟控制器已就绪！')

    def publish_velocity(self, linear_x, angular_z):
        """工具函数：发布一个速度指令"""
        twist = Twist()
        twist.linear.x = linear_x      # 线速度
        twist.angular.z = angular_z    # 角速度
        self.cmd_publisher.publish(twist)

    def stop(self):
        """停止海龟"""
        self.publish_velocity(0.0, 0.0)
        self.get_logger().info('🛑 停止')

    def draw_circle(self, duration=8.0):
        """
        画圆：恒定线速度 + 恒定角速度
        线速度让海龟往前走，角速度让它同时旋转 → 走出一个圆弧
        """
        self.get_logger().info('⭕ 开始画圆...')
        self.publish_velocity(linear_x=2.0, angular_z=1.5)
        time.sleep(duration)
        self.stop()

    def draw_square(self, side_duration=2.0):
        """
        画方形：走直线 → 转90° → 走直线 → 转90° → ...
        """
        self.get_logger().info('🟦 开始画方形...')
        for i in range(4):
            self.get_logger().info(f'  第{i+1}边: 前进')
            self.publish_velocity(linear_x=2.0, angular_z=0.0)
            time.sleep(side_duration)

            self.get_logger().info(f'  第{i+1}边: 转弯')
            # 角速度 π/2 ≈ 1.57 rad/s, 转 0.5 秒 ≈ 转 45°
            # 精确转 90° (π/2 rad)：角速度 * 时间 = π/2
            # 所以时间 = (π/2) / 角速度
            turn_time = (math.pi / 2) / 1.57
            self.publish_velocity(linear_x=0.0, angular_z=1.57)
            time.sleep(turn_time)

        self.stop()

    def draw_spiral(self):
        """
        画螺旋线：逐渐增大线速度或角速度
        向外螺旋：线速度不变，角速度逐渐减小
        """
        self.get_logger().info('🌀 开始画螺旋线...')
        for r in range(20):
            # 角速度逐渐减小 → 转弯半径越来越大 → 向外扩散
            self.publish_velocity(linear_x=2.0, angular_z=3.0 - r * 0.15)
            time.sleep(0.3)
        self.stop()


def main(args=None):
    rclpy.init(args=args)
    controller = TurtleController()

    try:
        # 依次执行三种图案
        controller.draw_circle(duration=6.0)
        time.sleep(1.0)

        controller.draw_square(side_duration=2.0)
        time.sleep(1.0)

        controller.draw_spiral()
        time.sleep(1.0)

    except KeyboardInterrupt:
        controller.stop()

    controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

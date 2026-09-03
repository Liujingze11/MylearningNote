#!/usr/bin/env python3
"""
第一个 ROS2 节点：发布者 (Publisher / Talker)

功能：每 0.5 秒发布一次消息，模拟一个"计数器"在说话。

运行方式（先确保 source 了 ROS2 环境）：
    python3 03_first_node_talker.py

同时在另一个终端运行 04 来接收消息。

--- 核心概念 ---
Node（节点）       =  ROS2 中最小的执行单元。一个 Python 脚本就是一个 Node。
Publisher（发布者） =  向某个 Topic（话题）发送消息的实体。
Topic（话题）      =  消息流通的"频道名"，发布者和订阅者通过 Topic 名字匹配。
Message（消息）    =  在 Topic 上传输的数据，有固定的类型（如 String, Twist）。
"""

import rclpy                          # ROS2 的 Python 客户端库
from rclpy.node import Node           # 所有节点的基类
from std_msgs.msg import String       # 使用标准消息类型 String


# ============================================================
# 1. 定义一个 Node 类
# ============================================================
class MinimalPublisher(Node):
    """
    自定义节点：继承自 rclpy.node.Node

    构造函数中做了三件事：
      1. 给节点起个名字
      2. 创建一个 Publisher（告诉 ROS：我要往某某 Topic 发某某类型的消息）
      3. 创建一个 Timer（定时器，每隔一段时间调用一次回调函数）
    """

    def __init__(self):
        # 调用父类构造函数，给节点起名 "minimal_publisher"
        super().__init__('minimal_publisher')

        # -------------------------------------------------------
        # 创建 Publisher
        # 参数: (消息类型, Topic名称, 队列大小)
        #       队列大小=10 表示最多缓存 10 条消息
        # -------------------------------------------------------
        self.publisher_ = self.create_publisher(String, 'chatter_topic', 10)

        # -------------------------------------------------------
        # 创建 Timer
        # 参数: (间隔秒数, 回调函数)
        #       每 0.5 秒执行一次 timer_callback
        # -------------------------------------------------------
        self.timer = self.create_timer(0.5, self.timer_callback)

        # 计数器，用于生成递增的消息内容
        self.count = 0

        self.get_logger().info('Publisher 节点已启动！')

    def timer_callback(self):
        """定时器回调：每 0.5 秒执行一次"""
        msg = String()                           # 创建一条 String 消息
        msg.data = f'Hello ROS2! 第 {self.count} 次打招呼'
        self.publisher_.publish(msg)              # 发布消息！
        self.get_logger().info(f'发布: "{msg.data}"')
        self.count += 1


# ============================================================
# 2. 主函数：初始化 → 创建节点 → 进入事件循环
# ============================================================
def main(args=None):
    rclpy.init(args=args)                    # 初始化 ROS2 客户端
    node = MinimalPublisher()                # 创建节点实例
    rclpy.spin(node)                         # 让节点保持运行（"自旋"）
    # spin() 会阻塞在这里，直到 Ctrl+C 被按下
    node.destroy_node()                      # 清理
    rclpy.shutdown()                         # 关闭 ROS2


if __name__ == '__main__':
    main()

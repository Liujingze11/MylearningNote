#!/usr/bin/env python3
"""
第一个 ROS2 节点：订阅者 (Subscriber / Listener)

功能：订阅 chatter_topic，每当有新消息就打印出来。

运行方式：
    python3 04_first_node_listener.py

先启动 03_talker，再启动本文件，你应该能看到消息一行行打印出来。

--- 核心概念 ---
Subscriber（订阅者）=  从某个 Topic（话题）接收消息的实体。
回调函数（Callback） =  每当有新消息到达时，自动被调用的函数。
spin()               =  ROS2 的事件循环，让节点一直 "活着" 等待消息。
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


# ============================================================
# 1. 订阅者节点
# ============================================================
class MinimalSubscriber(Node):
    """
    订阅者节点：监听 chatter_topic 并打印收到的消息

    create_subscription 的三个参数：
        消息类型    — 这个 Topic 上传输什么类型的消息
        Topic 名称  — 字符串，必须和发布者一致
        回调函数    — 收到消息时调用，参数就是消息对象
        队列大小    — 缓存多少条消息（默认 10）

    ⚠️ 注意：回调函数中不要做耗时操作（会阻塞其他回调）
    """

    def __init__(self):
        super().__init__('minimal_subscriber')

        self.subscription = self.create_subscription(
            String,
            'chatter_topic',
            self.listener_callback,
            10
        )
        # 防止未使用变量被 IDE 警告（可选）
        self.subscription

        self.get_logger().info('Subscriber 节点已启动，等待消息...')

    def listener_callback(self, msg):
        """
        消息回调函数：每当 chatter_topic 上有新消息，
        ROS2 的 spin 机制会自动调用这个函数。

        参数 msg 就是发布者发送的 String 消息对象。
        """
        self.get_logger().info(f'收到: "{msg.data}"')


# ============================================================
# 2. 主函数
# ============================================================
def main(args=None):
    rclpy.init(args=args)
    node = MinimalSubscriber()
    rclpy.spin(node)        # 让节点保持运行，等待消息
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
RabbitMQ 教程 1: Hello World
最简单的消息发送和接收示例
"""

import pika
import sys

def send_message():
    """发送消息到队列"""
    # 建立到 RabbitMQ 服务器的连接
    connection = pika.BlockingConnection(
        pika.ConnectionParameters('localhost')
    )
    channel = connection.channel()

    # 声明队列（如果不存在则创建）
    # 队列名: hello
    # durable=False: 队列不持久化（重启后丢失）
    channel.queue_declare(queue='hello')

    # 发送消息
    # exchange='': 使用默认交换机
    # routing_key='hello': 路由到 hello 队列
    # body: 消息内容（必须是字节串）
    message = 'Hello World!'
    channel.basic_publish(
        exchange='',
        routing_key='hello',
        body=message.encode()
    )
    print(f" [x] Sent '{message}'")

    # 关闭连接
    connection.close()

def receive_message():
    """从队列接收消息"""
    # 建立连接
    connection = pika.BlockingConnection(
        pika.ConnectionParameters('localhost')
    )
    channel = connection.channel()

    # 声明队列（确保队列存在）
    channel.queue_declare(queue='hello')

    # 定义回调函数（收到消息时调用）
    def callback(ch, method, properties, body):
        """
        参数说明:
        - ch: channel 对象
        - method: 包含 delivery_tag 等信息
        - properties: 消息属性
        - body: 消息内容（字节串）
        """
        print(f" [x] Received {body.decode()}")

    # 设置消费者
    # queue: 从哪个队列消费
    # on_message_callback: 收到消息的回调函数
    # auto_ack=True: 自动确认消息（收到即确认）
    channel.basic_consume(
        queue='hello',
        on_message_callback=callback,
        auto_ack=True
    )

    print(' [*] Waiting for messages. To exit press CTRL+C')
    # 开始消费（阻塞式）
    channel.start_consuming()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法:")
        print("  发送消息: python 01_hello_world.py send")
        print("  接收消息: python 01_hello_world.py receive")
        sys.exit(1)

    if sys.argv[1] == 'send':
        send_message()
    elif sys.argv[1] == 'receive':
        receive_message()
    else:
        print(f"未知命令: {sys.argv[1]}")
        sys.exit(1)

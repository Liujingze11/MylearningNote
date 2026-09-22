#!/usr/bin/env python3
"""
RabbitMQ 教程 3: Publish/Subscribe (发布/订阅)
广播消息到多个消费者

特点:
- 一条消息可以被多个消费者接收
- 使用 Fanout 交换机
- 临时队列（消费者断开后自动删除）
"""

import pika
import sys

def emit_log(message):
    """发布日志消息"""
    connection = pika.BlockingConnection(
        pika.ConnectionParameters('localhost')
    )
    channel = connection.channel()

    # 声明交换机
    # exchange='logs': 交换机名称
    # exchange_type='fanout': 广播类型
    channel.exchange_declare(
        exchange='logs',
        exchange_type='fanout'
    )

    # 发布消息到交换机（不指定 routing_key）
    channel.basic_publish(
        exchange='logs',
        routing_key='',
        body=message.encode()
    )
    print(f" [x] Sent '{message}'")
    connection.close()

def receive_logs():
    """接收日志消息"""
    connection = pika.BlockingConnection(
        pika.ConnectionParameters('localhost')
    )
    channel = connection.channel()

    # 声明交换机
    channel.exchange_declare(
        exchange='logs',
        exchange_type='fanout'
    )

    # 声明临时队列
    # exclusive=True: 断开连接后自动删除
    result = channel.queue_declare(queue='', exclusive=True)
    queue_name = result.method.queue

    # 绑定队列到交换机
    channel.queue_bind(
        exchange='logs',
        queue=queue_name
    )

    print(f' [*] Waiting for logs on queue: {queue_name}')

    def callback(ch, method, properties, body):
        print(f" [x] {body.decode()}")

    channel.basic_consume(
        queue=queue_name,
        on_message_callback=callback,
        auto_ack=True
    )

    print(' [*] Waiting for logs. To exit press CTRL+C')
    channel.start_consuming()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法:")
        print("  发布日志: python 03_pub_sub.py publish '日志消息'")
        print("  订阅日志: python 03_pub_sub.py subscribe")
        sys.exit(1)

    if sys.argv[1] == 'publish':
        if len(sys.argv) < 3:
            print("错误: 请提供日志消息")
            sys.exit(1)
        emit_log(sys.argv[2])
    elif sys.argv[1] == 'subscribe':
        receive_logs()
    else:
        print(f"未知命令: {sys.argv[1]}")
        sys.exit(1)

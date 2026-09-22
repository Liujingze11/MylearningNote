#!/usr/bin/env python3
"""
RabbitMQ 教程 2: Work Queues (工作队列)
任务分发和负载均衡示例

特点:
- 多个消费者可以同时处理任务
- 消息会轮询分配给不同的消费者
- 支持消息持久化
- 支持手动确认（防止消息丢失）
"""

import pika
import sys
import time

def send_task(message):
    """发送任务到队列"""
    connection = pika.BlockingConnection(
        pika.ConnectionParameters('localhost')
    )
    channel = connection.channel()

    # 声明持久化队列
    # durable=True: 队列持久化（重启后保留）
    channel.queue_declare(queue='task_queue', durable=True)

    # 发送持久化消息
    channel.basic_publish(
        exchange='',
        routing_key='task_queue',
        body=message.encode(),
        properties=pika.BasicProperties(
            delivery_mode=2,  # 使消息持久化
        )
    )
    print(f" [x] Sent '{message}'")
    connection.close()

def process_tasks():
    """处理任务（消费者）"""
    connection = pika.BlockingConnection(
        pika.ConnectionParameters('localhost')
    )
    channel = connection.channel()

    # 声明持久化队列
    channel.queue_declare(queue='task_queue', durable=True)

    # 设置预取计数为 1
    # 这样 RabbitMQ 不会在消费者处理完上一条消息前发送新消息
    channel.basic_qos(prefetch_count=1)

    def callback(ch, method, properties, body):
        """处理任务的回调函数"""
        message = body.decode()
        print(f" [x] Received '{message}'")

        # 模拟任务处理时间（每个点代表 1 秒）
        processing_time = message.count('.')
        time.sleep(processing_time)

        print(f" [x] Done")

        # 手动确认消息
        # delivery_tag: 消息的唯一标识
        ch.basic_ack(delivery_tag=method.delivery_tag)

    # 设置消费者（auto_ack=False 手动确认）
    channel.basic_consume(
        queue='task_queue',
        on_message_callback=callback
    )

    print(' [*] Waiting for tasks. To exit press CTRL+C')
    channel.start_consuming()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法:")
        print("  发送任务: python 02_work_queues.py send '任务描述...'")
        print("  处理任务: python 02_work_queues.py worker")
        print("")
        print("示例:")
        print("  python 02_work_queues.py send 'First task.'")
        print("  python 02_work_queues.py send 'Second task..'")
        print("  python 02_work_queues.py send 'Third task...'")
        sys.exit(1)

    if sys.argv[1] == 'send':
        if len(sys.argv) < 3:
            print("错误: 请提供任务描述")
            sys.exit(1)
        send_task(sys.argv[2])
    elif sys.argv[1] == 'worker':
        process_tasks()
    else:
        print(f"未知命令: {sys.argv[1]}")
        sys.exit(1)

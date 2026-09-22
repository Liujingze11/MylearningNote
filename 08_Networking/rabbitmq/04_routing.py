#!/usr/bin/env python3
"""
RabbitMQ 教程 4: Routing (路由)
选择性接收消息

特点:
- 使用 Direct 交换机
- 根据 routing_key 精确匹配路由
- 消费者可以订阅多个 routing_key
"""

import pika
import sys

def emit_log(routing_key, message):
    """发送带路由键的日志"""
    connection = pika.BlockingConnection(
        pika.ConnectionParameters('localhost')
    )
    channel = connection.channel()

    # 声明 Direct 交换机
    channel.exchange_declare(
        exchange='direct_logs',
        exchange_type='direct'
    )

    # 发送消息（指定 routing_key）
    channel.basic_publish(
        exchange='direct_logs',
        routing_key=routing_key,
        body=message.encode()
    )
    print(f" [x] Sent '{routing_key}':'{message}'")
    connection.close()

def receive_logs(severity_levels):
    """接收指定级别的日志"""
    connection = pika.BlockingConnection(
        pika.ConnectionParameters('localhost')
    )
    channel = connection.channel()

    # 声明 Direct 交换机
    channel.exchange_declare(
        exchange='direct_logs',
        exchange_type='direct'
    )

    # 声明临时队列
    result = channel.queue_declare(queue='', exclusive=True)
    queue_name = result.method.queue

    # 绑定队列到交换机（多个 routing_key）
    for severity in severity_levels:
        channel.queue_bind(
            exchange='direct_logs',
            queue=queue_name,
            routing_key=severity
        )

    print(f' [*] Waiting for logs: {severity_levels}')

    def callback(ch, method, properties, body):
        print(f" [x] {method.routing_key}:{body.decode()}")

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
        print("  发送日志: python 04_routing.py send <级别> '消息'")
        print("  接收日志: python 04_routing.py receive <级别1> [级别2] ...")
        print("")
        print("日志级别: info, warning, error")
        print("")
        print("示例:")
        print("  python 04_routing.py send info 'Info message'")
        print("  python 04_routing.py send warning 'Warning message'")
        print("  python 04_routing.py send error 'Error message'")
        print("  python 04_routing.py receive info warning error")
        sys.exit(1)

    if sys.argv[1] == 'send':
        if len(sys.argv) < 4:
            print("错误: 请提供路由键和消息")
            sys.exit(1)
        emit_log(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'receive':
        if len(sys.argv) < 3:
            print("错误: 请提供至少一个日志级别")
            sys.exit(1)
        receive_logs(sys.argv[2:])
    else:
        print(f"未知命令: {sys.argv[1]}")
        sys.exit(1)

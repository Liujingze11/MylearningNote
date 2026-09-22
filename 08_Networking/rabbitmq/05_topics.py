#!/usr/bin/env python3
"""
RabbitMQ 教程 5: Topics (主题)
模式匹配路由

特点:
- 使用 Topic 交换机
- routing_key 支持通配符匹配
- *: 匹配一个单词
- #: 匹配零个或多个单词
"""

import pika
import sys

def emit_topic(routing_key, message):
    """发送主题消息"""
    connection = pika.BlockingConnection(
        pika.ConnectionParameters('localhost')
    )
    channel = connection.channel()

    # 声明 Topic 交换机
    channel.exchange_declare(
        exchange='topic_logs',
        exchange_type='topic'
    )

    # 发送消息
    channel.basic_publish(
        exchange='topic_logs',
        routing_key=routing_key,
        body=message.encode()
    )
    print(f" [x] Sent '{routing_key}':'{message}'")
    connection.close()

def receive_topics(patterns):
    """接收匹配主题的消息"""
    connection = pika.BlockingConnection(
        pika.ConnectionParameters('localhost')
    )
    channel = connection.channel()

    # 声明 Topic 交换机
    channel.exchange_declare(
        exchange='topic_logs',
        exchange_type='topic'
    )

    # 声明临时队列
    result = channel.queue_declare(queue='', exclusive=True)
    queue_name = result.method.queue

    # 绑定队列到交换机（多个模式）
    for pattern in patterns:
        channel.queue_bind(
            exchange='topic_logs',
            queue=queue_name,
            routing_key=pattern
        )

    print(f' [*] Waiting for topics: {patterns}')

    def callback(ch, method, properties, body):
        print(f" [x] {method.routing_key}:{body.decode()}")

    channel.basic_consume(
        queue=queue_name,
        on_message_callback=callback,
        auto_ack=True
    )

    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法:")
        print("  发送消息: python 05_topics.py send <routing_key> '消息'")
        print("  接收消息: python 05_topics.py receive <模式1> [模式2] ...")
        print("")
        print("Routing Key 格式: <设施>.<级别>")
        print("  例如: kern.info, auth.warning, *.error")
        print("")
        print("通配符:")
        print("  *: 匹配一个单词")
        print("  #: 匹配零个或多个单词")
        print("")
        print("示例:")
        print("  python 05_topics.py send kern.info 'Kernel info'")
        print("  python 05_topics.py send auth.error 'Auth error'")
        print("  python 05_topics.py receive '*.error'")
        print("  python 05_topics.py receive 'kern.*'")
        print("  python 05_topics.py receive '#.error'")
        sys.exit(1)

    if sys.argv[1] == 'send':
        if len(sys.argv) < 4:
            print("错误: 请提供 routing_key 和消息")
            sys.exit(1)
        emit_topic(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'receive':
        if len(sys.argv) < 3:
            print("错误: 请提供至少一个匹配模式")
            sys.exit(1)
        receive_topics(sys.argv[2:])
    else:
        print(f"未知命令: {sys.argv[1]}")
        sys.exit(1)

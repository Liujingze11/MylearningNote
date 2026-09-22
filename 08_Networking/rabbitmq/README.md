# RabbitMQ 学习指南

## 什么是 RabbitMQ？

RabbitMQ 是一个开源的消息代理（Message Broker），实现了高级消息队列协议（AMQP）。它允许应用程序通过消息进行通信，实现解耦、异步处理和负载均衡。

## 核心概念

### 1. 消息模型
```
Producer → Exchange → Queue → Consumer
```

### 2. 核心组件

| 组件 | 说明 |
|------|------|
| **Producer** | 消息生产者，发送消息到 Exchange |
| **Exchange** | 接收消息并根据规则路由到 Queue |
| **Queue** | 存储消息的缓冲区 |
| **Consumer** | 消息消费者，从 Queue 获取消息 |
| **Binding** | 连接 Exchange 和 Queue 的规则 |
| **Channel** | 虚拟连接，减少 TCP 连接开销 |
| **Connection** | TCP 连接 |

### 3. Exchange 类型

| 类型 | 路由规则 | 典型场景 |
|------|----------|----------|
| **Direct** | 精确匹配 routing_key | 点对点通信 |
| **Fanout** | 广播到所有绑定的 Queue | 发布/订阅 |
| **Topic** | 模式匹配 routing_key | 灵活路由 |
| **Headers** | 根据消息头匹配 | 复杂路由 |

## 环境准备

### 1. 安装 RabbitMQ

**Ubuntu/Debian:**
```bash
# 安装 Erlang
sudo apt-get install erlang

# 安装 RabbitMQ
sudo apt-get install rabbitmq-server

# 启动服务
sudo systemctl start rabbitmq-server

# 启用管理插件（Web UI）
sudo rabbitmq-plugins enable rabbitmq_management
```

**Docker:**
```bash
docker run -d --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:management
```

### 2. 安装 Python 客户端
```bash
pip install pika
```

### 3. 访问管理界面
- URL: http://localhost:15672
- 默认用户名: guest
- 默认密码: guest

## 目录结构

```
rabbitmq/
├── README.md              # 本文件
├── 01_hello_world.py      # 入门：发送/接收消息
├── 02_work_queues.py      # 工作队列：任务分发
├── 03_pub_sub.py          # 发布/订阅：广播消息
├── 04_routing.py          # 路由：选择性接收
├── 05_topics.py           # 主题：模式匹配路由
├── 06_rpc.py              # RPC：远程过程调用
├── 07_confirms.py         # 消息确认：可靠性保证
└── rabbitmq_tutorial.ipynb # 交互式学习笔记本
```

## 快速开始

### 发送消息
```python
import pika

# 建立连接
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

# 声明队列
channel.queue_declare(queue='hello')

# 发送消息
channel.basic_publish(exchange='', routing_key='hello', body='Hello World!')
print(" [x] Sent 'Hello World!'")

connection.close()
```

### 接收消息
```python
import pika

# 建立连接
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

# 声明队列
channel.queue_declare(queue='hello')

# 定义回调函数
def callback(ch, method, properties, body):
    print(f" [x] Received {body.decode()}")

# 消费消息
channel.basic_consume(queue='hello', on_message_callback=callback, auto_ack=True)

print(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()
```

## 学习路径

1. **Hello World** - 理解基本的消息发送和接收
2. **Work Queues** - 学习任务分发和负载均衡
3. **Pub/Sub** - 掌握发布/订阅模式
4. **Routing** - 理解消息路由机制
5. **Topics** - 学习灵活的主题匹配
6. **RPC** - 实现远程过程调用
7. **Confirms** - 掌握消息确认机制

## 最佳实践

### 1. 连接管理
```python
# 使用连接上下文管理器
with pika.BlockingConnection(parameters) as connection:
    channel = connection.channel()
    # 使用 channel
```

### 2. 消息持久化
```python
# 声明持久化队列
channel.queue_declare(queue='task_queue', durable=True)

# 发送持久化消息
channel.basic_publish(
    exchange='',
    routing_key='task_queue',
    body=message,
    properties=pika.BasicProperties(
        delivery_mode=2,  # 使消息持久化
    ))
```

### 3. 消息确认
```python
# 手动确认消息
def callback(ch, method, properties, body):
    # 处理消息
    print(f"Received: {body.decode()}")
    # 手动确认
    ch.basic_ack(delivery_tag=method.delivery_tag)

channel.basic_consume(queue='hello', on_message_callback=callback)
```

### 4. 预取计数
```python
# 一次只处理一条消息
channel.basic_qos(prefetch_count=1)
```

## 常见问题

### Q: 消息丢失怎么办？
A: 使用消息持久化 + 发布者确认 + 消费者手动确认

### Q: 如何实现消息优先级？
A: 声明队列时设置 `x-max-priority` 参数

### Q: 如何处理消息积压？
A: 增加消费者数量，或使用惰性队列

### Q: 如何实现延迟消息？
A: 使用 TTL + 死信队列，或安装延迟插件

## 参考资源

- [RabbitMQ 官方文档](https://www.rabbitmq.com/documentation.html)
- [RabbitMQ Tutorials](https://www.rabbitmq.com/tutorials)
- [Pika 文档](https://pika.readthedocs.io/)

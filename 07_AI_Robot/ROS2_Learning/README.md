# ROS2 学习笔记 (基于 Python)

## 环境准备

```bash
# 1. 激活 Python 3.10 环境（ROS2 Humble 要求）
conda activate ros2_humble

# 2. 加载 ROS2 环境变量（每次新终端都要执行）
source /opt/ros/humble/setup.bash

# 3. 验证
ros2 --help
python3 -c "import rclpy; print('rclpy OK')"
```

> 提示：如果尚未创建 ros2_humble 环境：
> ```bash
> conda create -n ros2_humble python=3.10 -y
> ```

## 学习路线

| 序号 | 文件 | 内容 |
|------|------|------|
| 01 | `01_start_turtlesim.md` | 启动小海龟模拟器 + 键盘遥控 |
| 02 | `02_ros2_cli_basics.py` | ROS2 命令行工具速查（topic/service/node） |
| 03 | `03_first_node_talker.py` | 第一个 Python 节点：发布者 (Publisher) |
| 04 | `04_first_node_listener.py` | 第一个 Python 节点：订阅者 (Subscriber) |
| 05 | `05_turtle_controller.py` | 控制小海龟画圆/画方形 |
| 06 | `06_turtle_follower.py` | 让一只海龟追另一只海龟 |

## Ros2 核心概念速览

```
Node（节点）     →  最小的执行单元，一个 Python 脚本就是一个 Node
Topic（话题）    →  节点间通信的通道，发布/订阅 模式
Message（消息）  →  Topic 上传输的数据结构（如坐标、速度）
Service（服务）  →  请求/响应 模式，类似函数调用
Action（动作）   →  可取消、可追踪进度的长任务
```

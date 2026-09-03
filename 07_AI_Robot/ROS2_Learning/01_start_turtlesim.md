# 01 - 启动小海龟 + ROS2 核心概念详解

---

## 零、先理解：ROS2 到底解决什么问题？

想象你要做一个机器人，它需要同时做很多事情：

```
机器人 = 激光雷达 + 摄像头 + 电机驱动 + 路径规划 + SLAM建图 + 避障 + 遥控接收 ...
```

如果把这些全部写在一个程序里：
- 一个模块崩溃 → 整个系统挂掉
- 一个模块改代码 → 全部要重新编译
- 不同模块可能由不同团队用不同语言写（C++ 的视觉算法 + Python 的决策逻辑）

**ROS2 的解决思路：拆成独立的小程序（节点），它们通过网络互相通信。**

```
  ┌──────────────┐     发送"前方30cm有障碍物"
  │  激光雷达节点 │ ──────────────────────────────→  ┌──────────────┐
  │  (C++ 驱动)   │                                   │  避障决策节点  │
  └──────────────┘                                   │  (Python)     │
                                                     │               │
  ┌──────────────┐     发送"速度=0, 左转"             │ 收到障碍物    │
  │  电机控制节点 │ ←──────────────────────────────  │ 决策：停车    │
  │  (C++ 驱动)   │                                   │      左转    │
  └──────────────┘                                   └──────────────┘
```

**核心思想就一句话：玫瑰色的节点，彼此通过网络消息通信。**

---

## 一、核心概念详解

### 1.1 节点 (Node)

**节点 = ROS2 中最小的执行单元。一个节点 = 一个独立的程序。**

```
ros2 run turtlesim turtlesim_node   →  启动了一个叫做 /turtlesim 的节点
ros2 run turtlesim turtle_teleop_key → 启动了一个叫做 /teleop_turtle 的节点
```

类比：你可以把每个节点理解为公司里的一个员工：
- 有人专门看雷达（传感器驱动节点）
- 有人专门规划路线（路径规划节点）
- 有人专门发指令给电机（控制节点）
- 每个人只做自己分内的事，需要协作时通过"邮件"（消息）沟通

**节点的特点：**
| 特性 | 说明 |
|------|------|
| 独立进程 | 每个节点是独立的 Linux 进程，崩溃互不影响 |
| 松耦合 | 节点之间不直接调用对方代码，只通过消息通信 |
| 多语言 | Python / C++ 可以混合使用，节点间通信不关心对方用什么语言 |
| 分布式 | 节点可以跑在同一台机器上，也可以分布在多台机器上 |

**节点能做什么？**
- 发布 (Publish)：往某个 Topic 发送消息
- 订阅 (Subscribe)：从某个 Topic 接收消息
- 提供服务 (Service)：响应别人的请求
- 调用服务 (Client)：请求别人做某事

一个节点通常同时做好几件事。比如小海龟的 `turtlesim_node`：
- 订阅 `/turtle1/cmd_vel`（接收速度指令）← 这是 Subscribe
- 发布 `/turtle1/pose`（报告当前位置）← 这是 Publish
- 提供 `/spawn` 服务（允许外部创建新海龟）← 这是 Service

---

### 1.2 发布/订阅 模型 (Publish/Subscribe)

这是 ROS2 最核心的通信模式。你需要理解三个概念以及它们之间的关系：

```
┌──────────────────┐                          ┌──────────────────┐
│   Publisher      │  ────  Message ────→     │   Subscriber     │
│   (发布者)        │       (消息)              │   (订阅者)        │
│                  │      通过 Topic           │                  │
│  例: teleop_key  │     /turtle1/cmd_vel     │  例: turtlesim   │
└──────────────────┘                          └──────────────────┘
```

#### (a) Publisher（发布者）

发布者是**往某个 Topic 发送消息**的实体。

类比：广播电台。电台只管往外发送信号，不关心有没有人在听、有多少人在听。

```python
# 创建一个 Publisher 只需要一句话：
self.publisher = self.create_publisher(消息类型, Topic名称, 队列大小)
#                                       ↑          ↑         ↑
#                                      Twist   "/cmd_vel"   10
```

#### (b) Subscriber（订阅者）

订阅者是**从某个 Topic 接收消息**的实体。

类比：收音机。你调到某个频率（Topic），就能收到该电台的内容。你可以同时听多个频率，也可以只听一个。

```python
# 创建一个 Subscriber 只需要一句话：
self.subscriber = self.create_subscription(消息类型, Topic名称, 回调函数, 队列大小)
#                                           ↑          ↑         ↑          ↑
#                                          Twist   "/cmd_vel"  on_msg()    10
```

**关键：回调函数 (Callback)**
- 每当有新消息到达，ROS2 会自动调用你指定的函数
- 你不需要写 `while True: check_new_message()` — ROS2 帮你处理了
- 回调函数里写"收到消息后做什么"

#### (c) Topic（话题）

Topic 是 Publisher 和 Subscriber 之间的**命名通道**。

类比：Topic 就是电台的频率号（比如 FM 101.7）。

| Topic 的特点 | 说明 |
|-------------|------|
| 名字唯一 | `/turtle1/cmd_vel` 和 `/turtle1/pose` 是不同的 Topic |
| 多对多 | 一个 Topic 可以有多个 Publisher，也可以有多个 Subscriber |
| 类型绑定 | 每个 Topic 只能传输一种类型的消息 |
| 匿名 | Publisher 不知道谁在订阅，Subscriber 不知道谁在发布 |

**类比总结：**
```
        YouTube 频道               ≈    ROS2 Topic
        上传视频的人                ≈    Publisher
        订阅频道的观众              ≈    Subscriber
        上传的视频内容              ≈    Message
```

- 上传者不知道有谁在看（匿名）
- 订阅者只管接收新视频（被动等待）
- 一个频道可以有多个上传者（不常见），也可以有无数订阅者（常见）
- 视频格式是固定的（消息类型固定）

#### (d) Message（消息）

消息是在 Topic 上传输的**数据结构**，有固定的格式。

类比：消息就是信件的格式 — 信封上必须有"收件人"和"地址"，缺一不可。

```bash
# 查看 Twist 消息的定义
$ ros2 interface show geometry_msgs/msg/Twist

Vector3 linear        # 线速度 (x=前进, y=侧移, z=上升)
        float64 x
        float64 y
        float64 z
Vector3 angular       # 角速度 (绕 x/y/z 轴旋转)
        float64 x
        float64 y
        float64 z
```

对于小海龟运动，我们只用两个字段：
```
linear.x   = 前进速度 (正=前进, 负=后退)
angular.z  = 转向速度 (正=逆时针, 负=顺时针)
```

ROS2 内置了大量标准消息类型，也可以自定义：
| 消息类型 | 用途 |
|---------|------|
| `std_msgs/msg/String` | 传输字符串 |
| `std_msgs/msg/Int32` | 传输整数 |
| `geometry_msgs/msg/Twist` | 传输速度指令（线速度+角速度） |
| `sensor_msgs/msg/Image` | 传输图像（摄像头数据） |
| `sensor_msgs/msg/LaserScan` | 传输激光雷达数据 |
| `nav_msgs/msg/Odometry` | 传输里程计数据（位置+速度估计） |

---

### 1.3 三种通信模式全景对比

ROS2 有三种通信方式，这里先建立印象，后面会逐个深入：

```
1. Topic（发布/订阅） — 最常用
   Publisher ────→ Topic ────→ Subscriber
   特点：单向、持续、多对多
   场景：传感器数据流、速度指令、状态更新

2. Service（服务） — 请求/响应
   Client ────→ Request ────→ Server ────→ Response ────→ Client
   特点：一问一答、有去有回
   场景：重置位置、查询参数、触发一次性动作

3. Action（动作） — 可追踪的长任务
   Client ────→ Goal ────→ Server ────→ Feedback (过程中)
                                     ────→ Result (完成时)
   特点：可取消、可追踪进度
   场景：导航到目标点、机械臂抓取、长时间移动
```

---

### 1.4 小海龟案例中的实际数据流

让我们把上面所有的概念用小海龟场景串起来：

```
终端1: ros2 run turtlesim turtlesim_node
终端2: ros2 run turtlesim turtle_teleop_key
```

```
  ┌──────────────────────┐                              ┌───────────────────────┐
  │  Node: /teleop_turtle │                              │  Node: /turtlesim      │
  │                      │                              │                       │
  │  你按键盘 ↑           │     Topic: /turtle1/cmd_vel   │                       │
  │      ↓               │     消息类型: Twist            │                       │
  │  Publisher           │ ═══════════════════════════→  │  Subscriber           │
  │  发布 {linear.x:2.0, │                              │  收到速度指令后        │
  │        angular.z:0.0}│                              │  更新海龟位置并渲染     │
  │                      │                              │                       │
  │                      │    Topic: /turtle1/pose       │                       │
  │                      │    消息类型: Pose              │                       │
  │                      │ ←═══════════════════════════  │  Publisher            │
  │                      │                              │  发布 {x:5.5, y:5.5,  │
  │  (teleop 不订阅这个， │                              │        theta:0.0, ...} │
  │   但其他节点可以订阅)  │                              │                       │
  └──────────────────────┘                              └───────────────────────┘
```

注意：`turtlesim_node` 同时是：
- `/turtle1/cmd_vel` 的 **Subscriber**（接收外部发来的速度指令）
- `/turtle1/pose` 的 **Publisher**（对外发布海龟当前位置）

**这就是 ROS2 的核心模式：节点之间通过 Topic 发布和订阅消息，互不直接调用。**

---

### 1.5 全局图景：ROS2 的层级结构

```
┌─────────────────────────────────────────────────────┐
│  ROS2 生态系统                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │   Node A    │  │   Node B    │  │   Node C    │  │  ← 你的代码
│  │  (Python)   │  │  (C++)      │  │  (Python)   │  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  │
│         │                │                │          │
│  ┌──────┴────────────────┴────────────────┴──────┐  │
│  │              ROS2 Middleware (DDS)              │  │  ← 网络通信层
│  │    负责：发现节点、传递消息、序列化/反序列化      │  │
│  └────────────────────────────────────────────────┘  │
│                       │                              │
│  ┌────────────────────┴──────────────────────────┐  │
│  │              操作系统 (Linux)                   │  │  ← 底层
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

**你作为开发者，只需要关心最上层：写 Python 节点，决定它发布什么、订阅什么。**

中间的网络通信、消息序列化、节点发现等等，ROS2 自动帮你处理了。

---

## 二、动手：启动小海龟 + 键盘遥控

### 第一步：启动小海龟模拟器

打开 **终端1**，执行：

```bash
conda activate ros2_humble
source /opt/ros/humble/setup.bash
ros2 run turtlesim turtlesim_node
```

你会看到一个蓝底窗口，中间有一只小海龟 🐢。

这时，ROS2 系统中已经有**一个节点**在运行了：`/turtlesim`。

验证一下（开另一个终端）：
```bash
ros2 node list
# 输出: /turtlesim
```

### 第二步：键盘遥控海龟

打开 **终端2**（保持终端1运行），执行：

```bash
conda activate ros2_humble
source /opt/ros/humble/setup.bash
ros2 run turtlesim turtle_teleop_key
```

现在 ROS2 系统中有了**两个节点**：
```bash
ros2 node list
# 输出:
#   /turtlesim
#   /teleop_turtle
```

现在用键盘的 **↑ ↓ ← →** 方向键控制海龟移动。

### 第三步：窥探背后的通信

再开 **终端3**，用命令行工具查看这两个节点之间的通信：

```bash
conda activate ros2_humble
source /opt/ros/humble/setup.bash

# 1. 看看 /teleop_turtle 节点到底在做什么
ros2 node info /teleop_turtle
# 你会发现它只有一个 Publisher，往 /turtle1/cmd_vel 发消息

# 2. 看看 /turtlesim 节点在做什么
ros2 node info /turtlesim
# 你会发现它：
#   - 订阅了 /turtle1/cmd_vel（接收速度指令）
#   - 发布了 /turtle1/pose（报告自己的位置）
#   - 提供了 /spawn, /reset 等服务

# 3. 实时监听速度指令 Topic，然后你在终端2按方向键，观察数据变化
ros2 topic echo /turtle1/cmd_vel
```

你会亲眼看到：按 ↑ 时 `linear.x` 变成 2.0，松开时变回 0.0。

### 第四步：不用键盘，手动发消息控制海龟

```bash
# 发一次消息，让海龟走一个弧线
ros2 topic pub --once /turtle1/cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 2.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 1.5}}"
```

**这揭示了一个关键事实：海龟根本不在乎速度指令是谁发的 — 键盘也好，命令行也好，你写的 Python 脚本也好 — 只要有人往 `/turtle1/cmd_vel` 发 Twist 消息，它就乖乖执行。**

这就是 ROS2 的魅力：**解耦。** 你可以独立开发、独立测试每个节点。

---

## 三、总结

| 概念 | 一句话解释 | 小海龟中的例子 |
|------|-----------|---------------|
| **Node** | 独立运行的程序 | `turtlesim_node`, `teleop_key` |
| **Topic** | 消息的频道名 | `/turtle1/cmd_vel`, `/turtle1/pose` |
| **Publisher** | 往 Topic 发消息 | `teleop_key` 往 `/turtle1/cmd_vel` 发速度指令 |
| **Subscriber** | 从 Topic 收消息 | `turtlesim_node` 从 `/turtle1/cmd_vel` 收指令 |
| **Message** | 传输的数据结构 | `Twist {linear.x, angular.z}`, `Pose {x, y, theta}` |

**记住这个公式：**
```
Publisher ──(Message)──→ Topic ──(Message)──→ Subscriber
```

下一步 → [02_ros2_cli_basics.md](02_ros2_cli_basics.md)

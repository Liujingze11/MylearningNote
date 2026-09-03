# 02 - ROS2 命令行工具速查

> 在启动 turtlesim_node 后，另开终端执行以下命令来探索 ROS2 系统。

## 节点相关

```bash
# 查看所有正在运行的节点
ros2 node list
# 输出示例:  /turtlesim

# 查看某个节点的详细信息
ros2 node info /turtlesim
# 会列出该节点发布/订阅/提供的所有 Topic、Service、Action
```

## Topic 相关

```bash
# 查看所有 Topic
ros2 topic list

# 查看某个 Topic 的消息类型
ros2 topic type /turtle1/cmd_vel
# 输出: geometry_msgs/msg/Twist

# 实时监听 Topic 上的消息（相当于 "printf 调试"，非常重要！）
ros2 topic echo /turtle1/cmd_vel
# 然后用键盘遥控海龟，你会看到实时的速度数据

# 查看 Topic 发送频率
ros2 topic hz /turtle1/pose

# 手动发布一次消息（不启动 Python 节点也能发消息！）
ros2 topic pub --once /turtle1/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 2.0}, angular: {z: 1.0}}"
# 海龟会走一个弧线
```

## Service 相关

```bash
# 查看所有 Service
ros2 service list

# 调用 Service（示例：重置海龟位置）
ros2 service call /reset std_srvs/srv/Empty "{}"

# 生成新海龟
ros2 service call /spawn turtlesim/srv/Spawn "{x: 5.0, y: 5.0, theta: 0.0, name: 'turtle2'}"
```

## 消息结构速查

```bash
# 查看消息的定义（有哪些字段、什么类型）
ros2 interface show geometry_msgs/msg/Twist
# 输出:
#   Vector3 linear    ← 线速度 (x, y, z)
#   Vector3 angular   ← 角速度 (x, y, z)

ros2 interface show turtlesim/msg/Pose
# 输出:
#   float32 x         ← 海龟的 x 坐标
#   float32 y         ← 海龟的 y 坐标
#   float32 theta     ← 海龟的朝向角
#   ...
```

## 最常用的调试三板斧

1. `ros2 topic list` — 有哪些 Topic？
2. `ros2 topic echo /xxx` — Topic 上在传什么数据？
3. `ros2 topic pub /xxx ...` — 手动发消息测试

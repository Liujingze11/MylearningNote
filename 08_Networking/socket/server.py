import socket

IP = "127.0.0.1"
PORT = 9999

# 1. 创建 socket 对象
# 实例化一个socket对象
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# AF_INET: IPv4（表示网络层使用IP协议）, SOCK_STREAM: TCP（表示传输层使用tcp协议）

# 2. 绑定 IP 和端口
server_socket.bind((IP,PORT))

# 3. 开始监听 参数5表示 最多接受多少个等待连接的客户端
server_socket.listen(5)
print("服务器启动，等待连接...")

# 4. 等待客户端连接（阻塞状态）
conn, addr = server_socket.accept()
print(f"客户端已连接: {addr}")

# 5. 循环接收和发送数据
while True:
    data = conn.recv(1024)  # 一次最多接收1024字节
    if not data:
        print("客户端断开连接")
        break
    print("收到消息:", data.decode('utf-8'))

    # Echo 回传
    conn.sendall(data)

# 6. 关闭连接
conn.close()
server_socket.close()

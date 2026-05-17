import socket

IP = "127.0.0.1"
PORT = 9999

# 1. 创建 socket 对象
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 2. 连接服务器
client_socket.connect((IP,PORT))

# 3. 发送数据
while True:
    message = input("请输入消息(输入exit退出): ")
    if message.lower() == "exit":
        break
    client_socket.sendall(message.encode('utf-8'))

    # 接收服务器返回的消息
    data = client_socket.recv(1024)
    print("服务器返回:", data.decode('utf-8'))

# 4. 关闭连接
client_socket.close()

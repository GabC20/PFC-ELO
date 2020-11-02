import socket
import threading

# Parâmetros definem IPv4 e conexão TCP, respectivamente
sock=socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Endereço e porta
sock.bind(('0.0.0.0',10000))

# Numero de conexoes permitidas
sock.listen(1)

while True:
    c, a = sock.accept()
    while True:
        message = input('Digite a mensagem: ')
        c.send(message.encode('utf-8'))

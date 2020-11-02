
import socket
import threading


# Parametros definem IPv4 e conexao TCP respectivamente
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


# Endereco e porta
sock.bind(('192.168.1.85', 6000))

# Numero de conexoes permitidas
sock.listen(1)

connections = []


def handler(c, a):
    global connections
    while True:
        data = c.recv(1024)
        for connection in connections:
            print(data)
            connection.send(bytes(data))
        if not data:
            connections.remove(c)
            c.close()
            break

while True:
    c, a = sock.accept()
    print('Conexao estabelecida')
    cThread = threading.Thread(target=handler, args=(c,a))
    cThread.daemon = True
    cThread.start()
    connections.append(c)
    print(connections)

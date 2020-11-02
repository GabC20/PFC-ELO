import cv2, socket, threading, serial
import RPi.GPIO as GPIO

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD) # Utilizar posicao fisica dos pinos

#Configura a serial e a velocidade de transmissao
ser = serial.Serial("/dev/ttyAMA0", 115200)

HOST='172.20.10.2' # Endereco IP do computador que esta rodando o servidor
PORT=10000 # Porta para acessar o servidor
s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

print('Loading video capture')
video_cap = cv2.VideoCapture(0)

cv2.namedWindow('webcam')

def handler_client(s, any1):
    while True:
        data = s.recv(1024)
        if data.decode('utf-8')=='Esquerda':
                print('foi para a esquerda')
                ser.write("E")
        if data.decode('utf-8')=='Direita':
                print('foi para a direita')
                ser.write("D")
        if data.decode('utf-8')=='Frente':
                print('foi para a frente')
                ser.write("F")
        if data.decode('utf-8')=='Tras':
                print('foi para a tras')
                ser.write("T")
        if data.decode('utf-8')=='Parado':
                print('ficou parado')
                ser.write("P")
        if data.decode('utf-8')=='q':
                print('Fechar tudo')
                #cv2.destroyAllWindows()
                s.close()
                break

# Criacao de um thread para realizar a comunicacao TCP
# Note que sao necessarios 2 argumentos iteraveis em um thread
# E o segundo nao realiza nenhuma tarefa na funcao handler_client
clientThread = threading.Thread(target = handler_client, args = (s,'any'))
clientThread.daemon = True
clientThread.start()

while True:
    # print('Reading frame')
    grabbed, frame = video_cap.read()
    if not grabbed:
        print('Frame not grabbed')
        continue

    cv2.imshow('webcam', frame)

    # Digite q para fechar a janela da camera e interromper o programa
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()

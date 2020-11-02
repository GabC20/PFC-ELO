import time

import cv2
import numpy as np
import torch
# import torch2trt
import torchvision

from PIL import Image
from torch.nn import Conv2d, Linear, MaxPool2d, ReLU, Sequential
from torch.nn.functional import softmax
from torchvision import transforms
import gest_model
import socket

# Parâmetros definem IPv4 e conexão TCP, respectivamente
sock=socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Endereço e porta
sock.bind(('0.0.0.0',10000))

# Numero de conexoes permitidas
sock.listen(1)

def main():


    num_outputs = 5

    # model = torchvision.models.mobilenet_v2()
    # model = torchvision.models.mnasnet1_0()
    # model = torchvision.models.mnasnet1_0()

    # model.classifier = torch.nn.Sequential(
    #     # torch.nn.AvgPool2d((7, 7)),
    #     # torch.nn.Flatten(),
    #     torch.nn.Linear(1280, 128),
    #     torch.nn.Sigmoid(),
    #     torch.nn.Dropout(),
    #     torch.nn.Linear(128, num_outputs),
    #     torch.nn.Sigmoid() if num_outputs==1 else torch.nn.Softmax(1)
    # )

    model = gest_model.GestModel()
    # model = torchvision.models.mnasnet1_0(pretrained=True)


    model.load_state_dict(torch.load('model/f25/epoch40.pth')['state_dict'])
    model.cuda().eval()


    print('Loading video capture')
    video_cap = cv2.VideoCapture(0)

    tfs = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        # transforms.Normalize((0.5), (0.5))

    ])

    c, a = sock.accept()

    cv2.namedWindow('webcam')

    num_frame = 0
    total_frames_time = 0
    # for frame in video_cap:
    while True:
        # predictions = []
        # length = 0
        # while(length <= 5):

        # print('Reading frame')
        grabbed, frame = video_cap.read()
        print(frame.shape)
        if not grabbed:
            print('Frame not grabbed')
            continue

        num_frame += 1

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = tfs(frame).unsqueeze(0).cuda()
        # print(tfs(frame).shape)
        # print(tfs(frame).unsqueeze(0).shape)
        # print(tensor.shape)





        t0 = time.time()
        output = model(tensor) #.cpu().detach().numpy()
        output = softmax(output, dim=1)
        latency = time.time() - t0

        total_frames_time += latency

        cv2.putText(frame, 'FPS: {:.1f}'.format(num_frame/total_frames_time), (450,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)


        # print('output dim: ', output.shape)

        # print(output.data)
        # return

        THRESHOLD = 0.95





        print('Output data', output.data)
        print('Output', output)


        _, result = torch.max(output.data, 1)
        print(f'output: [{output[0, 0]:.4f} | {output[0, 1]:.4f} | {output[0, 2]:.4f} | {output[0, 3]:.4f} | {output[0, 4]:.4f}], result: {result[0]}, latency: {latency*1000:.2f} ms')

        # predictions.append(result)
        # print('Drawing results on frame')
        # color = (0, int(255.0*(1-result)), int(255.0*(result)))

        count = 0

        for i in range(0,5):
            if (output.data[0, i] < THRESHOLD):
                count += 1

        if count == 5:
            result = 5




        if (result == 0):
            print('Direita')
            cv2.putText(frame, 'Direita', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
            c.send('Direita'.encode('utf-8'))
        if (result == 1):
            print('Esquerda')
            cv2.putText(frame, 'Esquerda', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
            c.send('Esquerda'.encode('utf-8'))
        if (result == 2):
            print('Frente')
            cv2.putText(frame, 'Frente', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
            c.send('Frente'.encode('utf-8'))
        if (result == 3):
            print('Parado')
            cv2.putText(frame, 'Parado', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
            c.send('Parado'.encode('utf-8'))
        if (result == 4):
            print('Tras')
            cv2.putText(frame, 'Tras', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
            c.send('Tras'.encode('utf-8'))
        # if (result == 5):
        #     print('Sem comando')
        #     cv2.putText(frame, 'Sem comando', (200,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        #     c.send('Sem Comando'.encode('utf-8'))

        # if result == 1:
        #     color = (0, 0, 255)
        # else:
        #     color = (0, 255, 0)

        # length += 1
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.imshow('webcam', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            c.send('Parado'.encode('utf-8'))
            break

cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

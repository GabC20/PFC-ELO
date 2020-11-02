import os
import cv2
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from matplotlib import style

REBUILD_DATA = False # set to true to one once, then back to false unless you want to change something in your training data.

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('Running on the GPU')
else:
    device = torch.device('cpu')
    print('Running on the CPU')

class Gest():
    IMG_SIZE = 56
    Direita = 'dataset/Train_Gest_Dataset_Resized/train/Direita'
    Esquerda = 'dataset/Train_Gest_Dataset_Resized/train/Esquerda'
    Frente = 'dataset/Train_Gest_Dataset_Resized/train/Frente'
    Parado = 'dataset/Train_Gest_Dataset_Resized/train/Parado'
    Tras = 'dataset/Train_Gest_Dataset_Resized/train/Tras'


    LABELS = {Direita: 0, Esquerda: 1, Frente: 2, Parado: 3, Tras: 4}

    training_data = []
    cont_direita = 0
    cont_esquerda = 0
    cont_frente = 0
    cont_parado = 0
    cont_tras = 0


    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            for f in os.listdir(label):
                if "jpg" in f:
                    try:
                        path = os.path.join(label, f)
                        img = cv2.imread(path)
                        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                        self.training_data.append([np.array(img), np.eye(5)[self.LABELS[label]]])  # do something like print(np.eye(2)[1]), just makes one_hot
                        #print(np.eye(2)[self.LABELS[label]])

                        if label == self.Direita:
                            self.cont_direita += 1
                        if label == self.Esquerda:
                            self.cont_esquerda += 1
                        if label == self.Frente:
                            self.cont_frente += 1
                        if label == self.Parado:
                            self.cont_parado += 1
                        if label == self.Tras:
                            self.cont_tras += 1

                    except Exception as e:
                        pass
                        #print(label, f, str(e))

        np.random.shuffle(self.training_data)
        np.save('training_data.npy', self.training_data)
        print('Direita: ', self.cont_direita)
        print('Esquerda: ', self.cont_esquerda)
        print('Frente: ', self.cont_frente)
        print('Parado: ', self.cont_parado)
        print('Tras: ', self.cont_tras)


if REBUILD_DATA:
    gest = Gest()
    gest.make_training_data()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.features = 3*3*64

        # self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=0, bias=True)
        # self.maxpool= torch.nn.MaxPool2d(kernel_size=2)
        # self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=2, bias=True)
        # self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True)

        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0, bias=True)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=0, bias=True)
        self.maxpool= torch.nn.MaxPool2d(kernel_size=2)

        self.dropout1 = torch.nn.Dropout(0.5)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(self.features, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 5)
        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax()

    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)

        x = x.view(-1, self.features)
        # x = self.dropout1(x)
        x = self.elu(self.fc1(x))
        x = self.dropout1(x)
        x = self.elu(self.fc2(x))
        # x = F.sigmoid(self.fc3(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return F.softmax(x, dim=1)

net = Net().to(device)

print(net)

training_data = np.load("training_data.npy", allow_pickle=True, encoding='bytes')
print(len(training_data))

optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()

X = torch.Tensor([i[0] for i in training_data]).view(-1,3, 56, 56)
# print(len(X))
X = X/255.0
# print(len(X))

y = torch.Tensor([i[1] for i in training_data])

VAL_PCT = 0.2
val_size = int(len(X)*VAL_PCT)
# print(val_size)

train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]

# print(len(train_X))
# print(len(test_X))

# BATCH_SIZE = 128
# EPOCHS = 10

# def train(net):
#     for epoch in range(EPOCHS):
#         for i in range(0,len(train_X), BATCH_SIZE):
#             batch_X = train_X[i:i+BATCH_SIZE].view(-1,3,56,56)
#             batch_y = train_y[i:i+BATCH_SIZE]
#             # print(batch_X.shape)
#             # print(batch_y.shape)
#
#             batch_X, batch_y = batch_X.to(device), batch_y.to(device)
#
#             net.zero_grad()
#             outputs = net(batch_X)
#             loss = loss_function(outputs, batch_y)
#             loss.backward()
#             optimizer.step()
#         print(f"Epoch: {epoch}. Loss: {loss}")
#
# train(net)

# def test(net):
#     correct = 0
#     total = 0
#
#     with torch.no_grad():
#         for i in range(len(test_X)):
#             real_class = torch.argmax(test_y[i]).to(device)
#             net_out = net(test_X[i].view(-1,3,56,56).to(device))[0]
#             predicted_class = torch.argmax(net_out)
#             if predicted_class == real_class:
#                 correct += 1
#             total += 1
#
#     print('Accuracy: ', round(correct/total, 3))
#
# test(net)

def fwd_pass(X,y, train=False):
    if train:
        net.zero_grad()
    outputs = net(X)
    matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs,y)]
    acc = matches.count(True)/len(matches)
    loss = loss_function(outputs, y)

    if train:
        loss.backward()
        optimizer.step()
    return acc, loss

def test(size=32):
    random_start = np.random.randint(len(test_X)-size)
    X, y = test_X[random_start:random_start+size], test_y[random_start:random_start+size]
    with torch.no_grad():
        val_acc, val_loss = fwd_pass(X.view(-1,3,56,56).to(device), y.to(device))
    return val_acc, val_loss

val_acc, val_loss = test(size=32)
print(val_acc, val_loss)

MODEL_NAME = f"model-{int(time.time())}"  # gives a dynamic model name, to just help with things getting messy over time.
net = Net().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()

def train(net):
    BATCH_SIZE = 100
    EPOCHS = 2

    with open("model.log", "a") as f:
        for epoch in range(EPOCHS):
            for i in range(0, len(train_X), BATCH_SIZE):
                batch_X = train_X[i:i+BATCH_SIZE].view(-1,3,56,56)
                batch_y = train_y[i:i+BATCH_SIZE]

                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                acc, loss = fwd_pass(batch_X, batch_y, train=True)

                #print(f"Acc: {round(float(acc),2)}  Loss: {round(float(loss),4)}")
                #f.write(f"{MODEL_NAME},{round(time.time(),3)},train,{round(float(acc),2)},{round(float(loss),4)}\n")
                # just to show the above working, and then get out:
                if i % 10 == 0:
                    val_acc, val_loss = test(size=100)
                    f.write(f"{MODEL_NAME},{round(time.time(),3)},{round(float(acc),2)},{round(float(loss), 4)},{round(float(val_acc),2)},{round(float(val_loss),4)}\n")
train(net)

style.use('ggplot')

model_name = 'model-1602443062'

def create_acc_loss_graph(model_name):
    contents = open("model.log", "r").read().split("\n")

    times = []
    accuracies = []
    losses = []

    val_accs = []
    val_losses = []

    for c in contents:
        if model_name in c:
            name, timestamp, acc, loss, val_acc, val_loss = c.split(",")

            times.append(float(timestamp))
            accuracies.append(float(acc))
            losses.append(float(loss))

            val_accs.append(float(val_acc))
            val_losses.append(float(val_loss))


    fig = plt.figure()

    ax1 = plt.subplot2grid((2,1), (0,0))
    ax2 = plt.subplot2grid((2,1), (1,0), sharex=ax1)


    ax1.plot(times, accuracies, label="acc")
    ax1.plot(times, val_accs, label="val_acc")
    ax1.legend(loc=2)
    ax2.plot(times,losses, label="loss")
    ax2.plot(times,val_losses, label="val_loss")
    ax2.legend(loc=2)
    plt.show()

create_acc_loss_graph(model_name)

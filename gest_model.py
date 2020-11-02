import torch.nn as nn
import torch
import torch.nn.functional as F


# MODELO 1

# class GestModel(torch.nn.Module):
#
#     def __init__(self):
#         super(GestModel, self).__init__()
#
#         self.features = 3*3*64
#
#         # self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=0, bias=True)
#         # self.maxpool= torch.nn.MaxPool2d(kernel_size=2)
#         # self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=2, bias=True)
#         # self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True)
#
#         self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2, bias=True)
#         self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0, bias=True)
#         self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=0, bias=True)
#         self.maxpool= torch.nn.MaxPool2d(kernel_size=2)
#
#         self.dropout1 = torch.nn.Dropout(0.5)
#         self.dropout2 = torch.nn.Dropout(0.5)
#         self.fc1 = torch.nn.Linear(self.features, 120)
#         self.fc2 = torch.nn.Linear(120, 84)
#         self.fc3 = torch.nn.Linear(84, 5)
#         self.elu = nn.ELU()
#         self.relu = nn.ReLU()
#         # self.softmax = nn.Softmax()
#
#     def forward(self, x):
#
#         x = self.elu(self.conv1(x))
#         x = self.maxpool(x)
#         x = self.elu(self.conv2(x))
#         x = self.maxpool(x)
#         x = self.elu(self.conv3(x))
#         x = self.maxpool(x)
#
#         x = x.view(-1, self.features)
#         # x = self.dropout1(x)
#         x = self.elu(self.fc1(x))
#         x = self.dropout1(x)
#         x = self.elu(self.fc2(x))
#         # x = F.sigmoid(self.fc3(x))
#         x = self.dropout2(x)
#         x = self.fc3(x)
#
#         return x


# MODELO 2

# class GestModel(torch.nn.Module):
#
#     def __init__(self):
#         super(GestModel, self).__init__()
#
#         self.features = 128*26*26
#
#         # self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=0, bias=True)
#         # self.maxpool= torch.nn.MaxPool2d(kernel_size=2)
#         # self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=2, bias=True)
#         # self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True)
#
#         self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2, bias=True)
#         self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5, stride=1, padding=0, bias=True)
#         self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0, bias=True)
#         self.maxpool= torch.nn.MaxPool2d(kernel_size=2)
#
#         self.dropout1 = torch.nn.Dropout(0.5)
#         self.dropout2 = torch.nn.Dropout(0.5)
#         self.fc1 = torch.nn.Linear(self.features, 256)
#         self.fc2 = torch.nn.Linear(256, 64)
#         self.fc3 = torch.nn.Linear(64, 5)
#         self.elu = nn.ELU()
#         self.relu = nn.ReLU()
#         # self.softmax = nn.Softmax()
#
#     def forward(self, x):
#
#         x = self.elu(self.conv1(x))
#         x = self.maxpool(x)
#         x = self.elu(self.conv2(x))
#         x = self.maxpool(x)
#         x = self.elu(self.conv3(x))
#         x = self.maxpool(x)
#
#         x = x.view(-1, self.features)
#         # x = self.dropout1(x)
#         x = self.elu(self.fc1(x))
#         x = self.dropout1(x)
#         x = self.elu(self.fc2(x))
#         # x = F.sigmoid(self.fc3(x))
#         x = self.dropout2(x)
#         x = self.fc3(x)
#
#         return x


# MODELO 3

# class GestModel(torch.nn.Module):
#
#     def __init__(self):
#         super(GestModel, self).__init__()
#
#         self.features = 7*7*1024
#
#         # self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=0, bias=True)
#         # self.maxpool= torch.nn.MaxPool2d(kernel_size=2)
#         # self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=2, bias=True)
#         # self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True)
#
#         self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
#         self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
#         self.conv3 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
#         self.conv4 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True)
#         self.conv5 = torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=True)
#         self.maxpool= torch.nn.MaxPool2d(kernel_size=2)
#
#         self.dropout1 = torch.nn.Dropout(0.5)
#         self.dropout2 = torch.nn.Dropout(0.5)
#         self.fc1 = torch.nn.Linear(self.features, 1024)
#         self.fc2 = torch.nn.Linear(1024, 256)
#         self.fc3 = torch.nn.Linear(256, 5)
#         self.elu = nn.ELU()
#         self.relu = nn.ReLU()
#         # self.softmax = nn.Softmax()
#
#     def forward(self, x):
#
#         x = self.elu(self.conv1(x))
#         x = self.elu(self.conv2(x))
#         x = self.maxpool(x)
#         x = self.elu(self.conv3(x))
#         x = self.elu(self.conv4(x))
#         x = self.maxpool(x)
#         x = self.elu(self.conv5(x))
#         x = self.maxpool(x)
#
#         x = x.view(-1, self.features)
#         # x = self.dropout1(x)
#         x = self.elu(self.fc1(x))
#         x = self.dropout1(x)
#         x = self.elu(self.fc2(x))
#         # x = F.sigmoid(self.fc3(x))
#         x = self.dropout2(x)
#         x = self.fc3(x)
#
#         return x


# MODELO 4

class GestModel(torch.nn.Module):

    def __init__(self):
        super(GestModel, self).__init__()

        self.features = 5*5*128

        # self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=0, bias=True)
        # self.maxpool= torch.nn.MaxPool2d(kernel_size=2)
        # self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=2, bias=True)
        # self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True)

        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=11, stride=1, padding=0, bias=True)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, stride=1, padding=0, bias=True)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0, bias=True)
        self.conv4 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True)
        self.conv5 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0, bias=True)
        self.maxpool= torch.nn.MaxPool2d(kernel_size=2)

        self.dropout1 = torch.nn.Dropout(0.5)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(self.features, 1024)
        self.fc2 = torch.nn.Linear(1024, 256)
        self.fc3 = torch.nn.Linear(256, 5)
        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax()

    def forward(self, x):

        x = self.elu(self.conv1(x))
        x = self.elu(self.conv2(x))
        x = self.maxpool(x)
        x = self.elu(self.conv3(x))
        x = self.elu(self.conv4(x))
        x = self.maxpool(x)
        x = self.elu(self.conv5(x))
        x = self.maxpool(x)

        x = x.view(-1, self.features)
        # x = self.dropout1(x)
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        # x = F.sigmoid(self.fc3(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x

#
# # MODELO 5
#
# class GestModel(torch.nn.Module):
#
#     def __init__(self):
#         super(GestModel, self).__init__()
#
#         self.features = 4*4*256
#
#         self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=11, stride=1, padding=0, bias=True)
#         self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=11, stride=1, padding=0, bias=True)
#         self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=0, bias=True)
#         self.conv4 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=0, bias=True)
#         self.conv5 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=0, bias=True)
#         self.conv6 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0, bias=True)
#         self.conv7 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0, bias=True)
#         self.conv8 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=0, bias=True)
#         self.maxpool= torch.nn.MaxPool2d(kernel_size=2)
#
#         self.dropout1 = torch.nn.Dropout(0.5)
#         self.dropout2 = torch.nn.Dropout(0.5)
#         self.fc1 = torch.nn.Linear(self.features, 1024)
#         self.fc2 = torch.nn.Linear(1024, 256)
#         self.fc3 = torch.nn.Linear(256, 5)
#         self.elu = nn.ELU()
#         self.relu = nn.ReLU()
#         # self.softmax = nn.Softmax()
#
#     def forward(self, x):
#
#         x = self.elu(self.conv1(x))
#         x = self.elu(self.conv2(x))
#         x = self.maxpool(x)
#         x = self.elu(self.conv3(x))
#         x = self.elu(self.conv4(x))
#         x = self.maxpool(x)
#         x = self.elu(self.conv5(x))
#         x = self.elu(self.conv6(x))
#         x = self.maxpool(x)
#         x = self.elu(self.conv7(x))
#         x = self.maxpool(x)
#         x = self.elu(self.conv8(x))
#
#         x = x.view(-1, self.features)
#         # x = self.dropout1(x)
#         x = self.relu(self.fc1(x))
#         x = self.dropout1(x)
#         x = self.relu(self.fc2(x))
#         # x = F.sigmoid(self.fc3(x))
#         x = self.dropout2(x)
#         x = self.fc3(x)
#
#         return x



# Modelo a24 (Melhor ate agora)

# class GestModel(torch.nn.Module):
#
#     def __init__(self):
#         super(GestModel, self).__init__()
#
#         self.features = 3*3*64
#
#         # self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=0, bias=True)
#         # self.maxpool= torch.nn.MaxPool2d(kernel_size=2)
#         # self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=2, bias=True)
#         # self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True)
#
#         self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2, bias=True)
#         self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0, bias=True)
#         self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=0, bias=True)
#         self.maxpool= torch.nn.MaxPool2d(kernel_size=2)
#
#         self.dropout1 = torch.nn.Dropout(0.5)
#         self.dropout2 = torch.nn.Dropout(0.5)
#         self.fc1 = torch.nn.Linear(self.features, 120)
#         self.fc2 = torch.nn.Linear(120, 84)
#         self.fc3 = torch.nn.Linear(84, 5)
#         self.elu = nn.ELU()
#         self.relu = nn.ReLU()
#         # self.softmax = nn.Softmax()
#
#     def forward(self, x):
#
#         x = self.elu(self.conv1(x))
#         x = self.maxpool(x)
#         x = self.elu(self.conv2(x))
#         x = self.maxpool(x)
#         x = self.elu(self.conv3(x))
#         x = self.maxpool(x)
#
#         x = x.view(-1, self.features)
#         # x = self.dropout1(x)
#         x = self.elu(self.fc1(x))
#         x = self.dropout1(x)
#         x = self.elu(self.fc2(x))
#         # x = F.sigmoid(self.fc3(x))
#         x = self.dropout2(x)
#         x = self.fc3(x)
#
#         return x

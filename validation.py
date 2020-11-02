import cv2
import os
import gest_model
import time
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.nn import Conv2d, Linear, MaxPool2d, ReLU, Sequential
from torch.nn.functional import softmax
from torchvision import transforms
import gest_model
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.metrics import roc_curve, auc


num_outputs = 5

model = gest_model.GestModel()

# model = torchvision.models.mnasnet1_0(pretrained=True)

model.load_state_dict(torch.load('model/f25/epoch40.pth')['state_dict'])
model.cuda().eval()

tfs = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    # transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    # transforms.Normalize(mean=[0.5, 0.5, 0.5],
    #                      std=[0.5, 0.5, 0.5])
])

PATH = '/home/gabriel_carvalho/teste/val6'

# categories = ['Direita', 'Esquerda', 'Frente', 'Parado', 'Sem_Comando', 'Tras']
categories = ['Direita', 'Esquerda', 'Frente', 'Parado', 'Tras']

roc_class_pred = []
roc_class_actual = []

idx2class = {0: 'Direita',
             1: 'Esquerda',
             2: 'Frente',
             3: 'Parado',
             4: 'Tras'
}

# Actual class
#   Predicted

stats = {
    'Direita': {
        'Direita': 0,
        'Esquerda': 0,
        'Frente': 0,
        'Parado': 0,
        'Tras': 0
    },
    'Esquerda': {
        'Direita': 0,
        'Esquerda': 0,
        'Frente': 0,
        'Parado': 0,
        'Tras': 0
    },
    'Frente': {
        'Direita': 0,
        'Esquerda': 0,
        'Frente': 0,
        'Parado': 0,
        'Tras': 0
    },
    'Parado': {
        'Direita': 0,
        'Esquerda': 0,
        'Frente': 0,
        'Parado': 0,
        'Tras': 0
    },
    'Tras': {
        'Direita': 0,
        'Esquerda': 0,
        'Frente': 0,
        'Parado': 0,
        'Tras': 0
    }
}

count = {
        'Direita': 0,
        'Esquerda': 0,
        'Frente': 0,
        'Parado': 0,
        'Tras': 0
}

actual_class_count = {
        'Direita': 0,
        'Esquerda': 0,
        'Frente': 0,
        'Parado': 0,
        'Tras': 0
}

y_actual = []
y_pred = []

total_inf_time = 0.0
num_inferences = 0

for idx, category in enumerate(categories):
    actual_category = category
    print('================================')
    print(actual_category)
    print('================================\n')
    LOCAL_PATH = os.path.join(PATH, category)
    for img in os.listdir(LOCAL_PATH):
        try:
            # img = cv2.imread(os.path.join(LOCAL_PATH, img), cv2.IMREAD_GRAYSCALE)
            img = cv2.imread(os.path.join(LOCAL_PATH, img))
            # img = img[ :, :, ::-1]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # print(img.shape)
            # img = img[100:550, 100:550]
            tensor = tfs(img).unsqueeze(0).cuda()
            # print(tensor)
            # print(tfs(img).shape)
            # print(tfs(img).unsqueeze(0).shape)
            # print(tensor.shape)
            t0 = time.time()
            output = model(tensor) #.cpu().detach().numpy()
            # print('model: ', output.shape)
            output = softmax(output, dim=1)
            # print(output)
            # print('softmax: ',output.shape)
            latency = time.time() - t0
            print(f'latency: {latency*1000:.2f} ms')

            total_inf_time += latency
            num_inferences += 1

            actual_class_count[category] += 1
            _, result = torch.max(output.data, 1)
            # print(result)
            stats[category][idx2class[result.item()]] += 1
            print(f'Resultado: {result.item()} | Label: {idx}')

            # print(torch.max(output).item())


            roc_class_pred.append(torch.max(output).item())



            y_actual.append(idx)
            y_pred.append(result.item())

            if (idx == result.item()):
                count[category] += 1
                roc_class_actual.append(1)
            else:
                roc_class_actual.append(0)
                pass

        except:
            pass




print('Mean Inference time: ', round(total_inf_time/num_inferences, 6))

print(classification_report(y_actual, y_pred, digits=4))
print(confusion_matrix(y_actual, y_pred))


confusion_matrix_df = pd.DataFrame(confusion_matrix(y_actual, y_pred)).rename(columns=idx2class, index=idx2class)

acertos = 0
total = 0

for i in range(0, len(categories)):
    acertos += stats[idx2class[i]][idx2class[i]]
    total += actual_class_count[idx2class[i]]

mean_accuracy = round(100*acertos/total, 4)
print('Mean Accuracy: ', mean_accuracy)


fig, ax = plt.subplots(figsize=(7,5))
sns.heatmap(confusion_matrix_df, annot=True, ax=ax, fmt='d')
plt.savefig('Confusion_Matrix_model_f36')
plt.show()

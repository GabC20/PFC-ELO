import os
import cv2
import random


PATH = '/home/gabriel_carvalho/teste/dataset/webcam_Resized'

folders = ['train', 'val']

categories = ['Direita', 'Esquerda', 'Frente', 'Parado', 'Tras']


files=os.listdir(PATH)
# print(len(files))

# d=random.choice(files)
# print(d)
# # os.startfile(d)
#
# # N: numero de imagens que se quer extrair
# #
# N =
#

# files = os.listdir('/home/gabriel_carvalho/teste/dataset/webcam_Resized/train/Direita/')
# print(len(files))

split_porcentage = 0.10

# N = NUM_IMG_TO_BE_EXTRACTED

for folder in folders:
    for category in categories:
        LOCAL_PATH = os.path.join(PATH, folder)
        LOCALEST_PATH = os.path.join(LOCAL_PATH, category)
        files = os.listdir(LOCALEST_PATH)
        N = round(split_porcentage * len(files))
        for n in range(N):
            try:
                d=random.choice(files)
                img = cv2.imread(os.path.join(LOCALEST_PATH, d))
                cv2.imwrite('/home/gabriel_carvalho/teste/dataset/webcam_Resized_10%/'+folder+'/'+category+'/IME_img'+str(n)+'.jpg', img)
                os.remove('/home/gabriel_carvalho/teste/dataset/webcam_Resized/'+folder+'/'+category+'/'+d)
            except:
                pass

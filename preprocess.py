import cv2
import os


PATH = '/home/gabriel_carvalho/teste/dataset/celular_IME/'

IMG_SIZE = 224

folders = ['train', 'val']

categories = ['Direita', 'Esquerda', 'Frente', 'Parado', 'Tras']


for folder in folders:
    for category in categories:
        LOCAL_PATH = os.path.join(PATH, folder)
        LOCALEST_PATH = os.path.join(LOCAL_PATH, category)
        for i, img in enumerate(os.listdir(LOCALEST_PATH)):
            img = cv2.imread(os.path.join(LOCALEST_PATH, img))
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            cv2.imwrite('/home/gabriel_carvalho/teste/dataset/celular_IME_Resized/'+folder+'/'+category+'/img_IME'+str(i)+'.jpg', img)
            print('/home/gabriel_carvalho/teste/dataset/celular_IME_Resized/'+folder+'/'+category+'/img_IME'+str(i)+'.jpg')

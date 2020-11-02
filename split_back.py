import cv2
import os


PATH = '/home/gabriel_carvalho/teste/dataset/Train_Gest_Dataset_Resized/'

folders = ['val']

categories = ['Direita', 'Esquerda', 'Frente', 'Parado', 'Tras']

# for i, img in enumerate(os.listdir(PATH)):
#     img = cv2.imread(os.path.join(PATH, img))
#     img = cv2.resize(img, (64,64))
#     cv2.imwrite('/home/gabriel_carvalho/Gesture_Recognition/dataset/Train_Gest_Dataset_Resized/train/Direita/img'+str(i)+'.jpg', img)

imgs = []

for i in range(0,469):
    img = 'img'+str(i)+'.jpg'
    imgs.append(img)





for folder in folders:
    for category in categories:
        LOCAL_PATH = os.path.join(PATH, folder)
        LOCALEST_PATH = os.path.join(LOCAL_PATH, category)
        for img in os.listdir(LOCALEST_PATH):
            try:
                if img  in imgs:
                    # cv2.imwrite('/home/gabriel_carvalho/teste/dataset/aux/'+folder+'/'+category+'/'+img)
                    os.remove('/home/gabriel_carvalho/teste/dataset/Train_Gest_Dataset_Resized/'+folder+'/'+category+'/'+img)
                    print('/home/gabriel_carvalho/teste/dataset/aux/'+folder+'/'+category+'/'+img)
            except:
                pass

import cv2
import os


PATH = '/home/gabriel_carvalho/Gesture_Recognition/dataset/Train_Gest_Dataset_Resized'



folders = ['train', 'val']

categories = ['Direita', 'Esquerda', 'Frente', 'Parado', 'Tras']

for folder in folders:
    for category in categories:
        LOCAL_PATH = os.path.join(PATH, folder)
        LOCALEST_PATH = os.path.join(LOCAL_PATH, category)
        for i, img in enumerate(os.listdir(LOCALEST_PATH)):
            # try:
            img = cv2.imread(os.path.join(LOCALEST_PATH, img))
            img = cv2.flip(img, 1)
            cv2.imwrite('/home/gabriel_carvalho/Gesture_Recognition/dataset/Train_Gest_Dataset_Resized_flipped/'+folder+'/'+category+'/img'+str(i)+'.jpg', img)

            print('/home/gabriel_carvalho/Gesture_Recognition/dataset/Train_Gest_Dataset_Resized_flipped/'+folder+'/'+category+'/img'+str(i)+'.jpg')
            # except:
            #     pass

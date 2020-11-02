import cv2


# for vid in PATH:
#     if(vid.split('.')[-1] == 'mp4')
# Opens the Video file
cap = cv2.VideoCapture('/home/gabriel_carvalho/teste/val6/Tras/tras.webm')
i=0
while(cap.isOpened()):
    try:
        ret, frame = cap.read()
        # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if ret == False:
            break
        cv2.imwrite('/home/gabriel_carvalho/teste/val6/Tras/val_tras'+str(i)+'.jpg',frame)
    except:
        pass
    i+=1

cap.release()
cv2.destroyAllWindows()

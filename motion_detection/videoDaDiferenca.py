import cv2
import numpy as np

video   = cv2.VideoCapture("my_video-1.mkv")
largura = video.get(cv2.CAP_PROP_FRAME_WIDTH)
altura  = video.get(cv2.CAP_PROP_FRAME_HEIGHT)


while(video.isOpened()):
    
    _  , frame1 = video.read()
    ret, frame2 = video.read()
    if ret == True:
        cv2.imshow('Frame',frame2)
        print(video.get(cv2.CAP_PROP_POS_FRAMES))
        if cv2.waitKey(130) & 0xFF == ord('q'):
          break
    else: 
        break
    diferenca = cv2.absdiff(frame1, frame2)
    media = np.mean(diferenca)

video.release()
cv2.destroyAllWindows()
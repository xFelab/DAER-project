import cv2
import numpy as np

video   = cv2.VideoCapture("my_video-1.mp4")
frames  = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
media   = np.zeros(frames)
mediaMovel = np.zeros(5)
ret, frameNovo = video.read()
frameVelho = frameNovo

for i in range(frames):
    diferenca = cv2.absdiff(frameNovo, frameVelho)
    mediaMovel[i%5] = np.mean(diferenca)
    frameVelho = frameNovo
    ret, frameNovo = video.read()
    if i > 4:
        media[i-4] = np.mean(mediaMovel)
        if media[i-4] > 2:
            cv2.imwrite('frames/'+str(i)+'.jpg', frameNovo)
video.release()
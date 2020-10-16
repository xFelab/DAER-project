import cv2
import numpy as np

video   = cv2.VideoCapture("my_video-2.mp4")
largura = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
altura  = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames  = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
media   = np.zeros(frames)
i = 0
ret, frameNovo = video.read()
frameVelho = frameNovo
oi=np.zeros(frames)
for i in range(frames):
    diferenca = cv2.absdiff(frameNovo, frameVelho)
    media[i] = np.mean(diferenca)
    i = i + 1
    frameVelho = frameNovo
    ret, frameNovo = video.read()
video.release()
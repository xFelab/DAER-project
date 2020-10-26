import cv2
import numpy as np
from time import sleep
import string
import random

def random_name():
    name = ""
    for x in range(15):
        name = name+random.choice(string.ascii_letters + string.digits)
    #print (name)
    return name

media_num = 5

video   = cv2.VideoCapture("0HDJPJO1.mp4")
frames  = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
media   = np.zeros(frames)
mediaMovel = np.zeros(media_num)
ret, frameNovo = video.read()
frameVelho = frameNovo

font = cv2.FONT_HERSHEY_SIMPLEX 

random_name()
  
for i in range(frames):
    diferenca = cv2.absdiff(frameNovo, frameVelho)
    mediaMovel[i%(media_num)] = np.mean(diferenca)
    frameVelho = frameNovo
    ret, frameNovo = video.read()
    
    if i > (media_num-1):
        #sleep(0.02)
        media[i-(media_num - 1)] = np.mean(mediaMovel)
        print ("frame: "+'{:d}'.format(i)+"  diff: "+'{:f}'.format(media[i-4])) 

        cv2.putText(frameNovo, "{:4.2f}".format(media[i-4]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Frame", frameNovo)
        #cv2.waitKey(1)
	
        if media[i-4] > 2.5:
            #cv2.imwrite(str(i)+".jpg", frameNovo)
            cv2.imwrite("frames/"+random_name()+".jpg", frameNovo)
            
video.release()

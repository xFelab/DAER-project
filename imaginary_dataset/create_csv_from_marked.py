from __future__ import print_function
import argparse
import numpy as np
from matplotlib import pyplot as plt
import cv2
import csv

image = cv2.imread('image_marked.png')
image_mod = cv2.imread('image_marked.png')
""" show the image with open cv
"""
#cv2.imshow("original", image)

""" show the image with matplot lib
plt.imshow(image, interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
"""

width = len(image[0])
height = len(image)

entrances = np.array([[1,1]])
exits = np.array([[1,1]])
#print (entrances)

f = open('image_coordinates.csv', 'w')
writer = csv.writer(f)
writer.writerow(['width','height','red','green','blue'])

print ('the image is '+str(width)+' by '+str(height)+' pixels ')

first_entrance_flag = 0
first_exit_flag = 0

for x in range(width): #print (x)
    for y in range(height): #print (y)
        (b, g, r) = image[y, x]
        
        #writer.writerow(pixels[index])
        np.savetxt(f, np.array([x, y, r, g, b]), newline=", ")
        f.write("\n")

        #look for entrances
        if(b == 255 and g == 0 and r == 0):
            print("found blue at x="+str(x)+" and y="+str(y))
            if(first_entrance_flag == 0):
                first_entrance_flag = 1
                entrances[0] = x, y
            else:
                entrances = np.append(entrances, [[x, y]], axis=0)
            print (entrances)

        #look for exits
        if(b == 0 and g == 0 and r == 255):
            print("found red at x="+str(x)+" and y="+str(y))
            if(first_exit_flag == 0):
                first_exit_flag = 1
                exits[0] = x, y
            else:
                exits = np.append(exits, [[x, y]], axis=0)
            print (exits)
            
f.close()

# write entrances to csv
f = open('entrances.csv', 'w')
for x in range(len(entrances)):
    np.savetxt(f, entrances[x], newline=", ")
    f.write("\n")
f.close()

# write exits to csv
f = open('exits.csv', 'w')
for x in range(len(exits)):
    np.savetxt(f, exits[x], newline=", ")
    f.write("\n")
f.close()

#cv2.imshow("image_mod", image_mod)

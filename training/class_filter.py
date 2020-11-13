# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 2020

@author: Felipe Fitarelli
"""
import os, re, copy, csv, shutil
import glob, numpy as np


dir_yolo_raw = '1/' #Aqui pode-se trocar o nome da pasta das imagens
dir_filtered = '/1_filtered' #Aqui pode-se trocar o nome do arquivo CSV de saída

len_yolo_raw = len(glob.glob(dir_yolo_raw+"/*.txt"))
files_yolo_raw = glob.glob(dir_yolo_raw+"/*.txt")

#print(len_yolo_raw)
#print(files_yolo_raw[5])
#files_yolo_raw[5] = files_yolo_raw[5,5:10]

yolo_names = [[] for i in range(len_yolo_raw)]

yolo_selected = []

#CLASSES
# 0 -> CAR
# 1 -> TRUCK
# 2 -> BUS
# 3 -> MOTORCYCLE

goal_car = 50
goal_truck = 50
goal_bus = 50
goal_motorcycle = 10

goals = [goal_car, goal_truck, goal_bus, goal_motorcycle]

for x in range(len_yolo_raw):
    temp_string = files_yolo_raw[x]
    yolo_names[x] = temp_string[2:]
    #print (yolo_names[x])
    print (goals)

    matrix = np.loadtxt(dir_yolo_raw+yolo_names[x], usecols=range(5))
    #print(matrix.ndim)
    if (matrix.ndim == 1):
        matrix = matrix.reshape(1, 5)

    for y in range (int(matrix.size/5)):
        #print(matrix[y,0])
        goals[int(matrix[y,0])] -= 1

    if ( all ( i == 0 for i in goals ) ):
        print("************************************")
        print("TODOS OBJETOS RECOLHIDOS COM SUCESSO")
        print("************************************")

        break
    
    elif ( all ( i >= 0 for i in goals ) ):    # verifica se todos são maiores que zero
        yolo_selected.append(yolo_names[x])
                  
    else:
        for y in range (int(matrix.size/5)):
            goals[int(matrix[y,0])] += 1

for x in range (len(yolo_selected)):
    yolo_jpg = yolo_selected[x]
    yolo_jpg = yolo_jpg[:-4] 
    #print(yolo_jpg)
    print (yolo_selected[x])    
    # target filename is /dst/dir/file.ext
    shutil.copy('1/'+yolo_selected[x], '1_filtered/')
    shutil.copy('1/'+yolo_jpg+'.jpg', '1_filtered/') 

        









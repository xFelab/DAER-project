#example obtained from:
# - https://docs.w3cub.com/scikit_learn/auto_examples/mixture/plot_gmm_pdf/#sphx-glr-auto-examples-mixture-plot-gmm-pdf-py

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture
import time

width = 352
height = 250

n_samples = 10
splash = 15

entrances = 3
exits = 3

coord_sorted = pd.read_csv("coordinates_sorted.csv", sep=',', header = 0)
#print(coord_sorted.values)

directions = np.zeros([exits, exits])
# lines are entrances
# columns are exits

for todos in range(len(coord_sorted.values)):
   for entradas in range(entrances):
       for saidas in range(exits):
           if(coord_sorted.values[todos, 1] == entradas and
              coord_sorted.values[todos, 2] == saidas):
               directions[entradas, saidas] = directions[entradas, saidas]+1
                          
print(directions)

for entradas in range(entrances):
    for saidas in range(exits):
        print(str(entradas)+" -> "+str(saidas)+" : "+
              str(directions[entradas,saidas]))



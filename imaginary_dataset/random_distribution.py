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

n_samples = 300
splash = 35

entrances = pd.read_csv("entrances.csv", sep=',', header = 0) 
print(entrances.values)

# generate random sample, two components
np.random.seed(int(time.time()))
#np.random.seed(0)

entrance_array = np.zeros([len(entrances.values), n_samples, 2])

for entr in range (len(entrances.values)):
    # generate spherical data centered on (20, 30)
    if(entr == 0):
        splash = 5
    if(entr == 1):
        splash = 25
    if(entr == 2):
        splash = 25
    
    entrance_array[entr] = np.array([entrances.values[entr, 0], entrances.values[entr, 1]])
    for i in range (n_samples):
        while(1):
            random_linha = np.random.randn(1, 1)*splash
            random_coluna = np.random.randn(1, 1)*splash
            entrance_array[entr, i, 0] += random_linha
            entrance_array[entr, i, 1] += random_coluna
            if(entrance_array[entr, i, 0] < width and entrance_array[entr, i, 1] < height and
               entrance_array[entr, i, 0] >= 0 and entrance_array[entr, i, 1] >= 0 ):
                break
            else:
                entrance_array[entr, i, 0] -= random_linha
                entrance_array[entr, i, 1] -= random_coluna
                
# concatenate the two datasets into the final training set
X_train = np.vstack([entrance_array[0], entrance_array[1], entrance_array[2]])

# fit a Gaussian Mixture Model with two components
clf = mixture.GaussianMixture(n_components=3, covariance_type='full')
clf.fit(X_train)

labels = clf.predict(X_train)
print(X_train)
print(labels)

X_train0 = np.zeros([1, 2])
flag_first_X_train0 = 0
y0 = 0
for x in range (len(labels)):
    if(labels[x] == 0):
        y0 += 1
        #print(X_train[x])
        if(flag_first_X_train0 == 0):
            flag_first_X_train0 = 1
            X_train0 = X_train[x]
        else:
            X_train0 = np.vstack([X_train0, X_train[x]])    
            #print(X_train0)
                        
#display predicted scores by the model as a contour plot
x = np.linspace(0, width-1, width)
y = np.linspace(0, height-1, height)
X, Y = np.meshgrid(x, y)

XX = np.array([X.ravel(), Y.ravel()]).T
Z = -clf.score_samples(XX)
Z = Z.reshape(X.shape)

print(Z[50, 50])

CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=30.0),
                 levels=np.logspace(0, 5, 50))
CB = plt.colorbar(CS, shrink=0.8, extend='both')

scatter = plt.scatter(X_train[:, 0], X_train[:, 1], 1, c='b')
scatter = plt.scatter(X_train0[:, 0], X_train0[:, 1], s=2**2, c ='r') 
#plt.scatter(X_train1[0], X_train1[1], c ='yellow') 
#plt.scatter(X_train2[0], X_train2[1], c ='g') 

ax = scatter.axes
#ax.invert_xaxis()
ax.invert_yaxis()

plt.title('probabilidade Entradas')
#plt.axis('tight')
plt.axis('equal')
plt.show()

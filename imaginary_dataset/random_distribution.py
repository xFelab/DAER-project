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
splash = 25

entrances = pd.read_csv("entrances.csv", sep=',', header = 0)
#print(entrances.values)
exits = pd.read_csv("exits.csv", sep=',', header = 0)
print(exits.values)

# generate random sample, two components
#np.random.seed(int(time.time()))
np.random.seed(0)

entrance_array = np.zeros([len(entrances.values), n_samples, 2])
exit_array = np.zeros([len(exits.values), n_samples, 2])

for entr in range (len(entrances.values)):
    # generate spherical data centered on (20, 30)
    """
    if(entr == 0):
        splash = 5
    if(entr == 1):
        splash = 25
    if(entr == 2):
        splash = 25
    """
    
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

for exi in range (len(exits.values)):
    exit_array[exi] = np.array([exits.values[exi, 0], exits.values[exi, 1]])
    for i in range (n_samples):
        while(1):
            random_linha = np.random.randn(1, 1)*splash
            random_coluna = np.random.randn(1, 1)*splash
            exit_array[exi, i, 0] += random_linha
            exit_array[exi, i, 1] += random_coluna
            if(exit_array[exi, i, 0] < width and exit_array[exi, i, 1] < height and
               exit_array[exi, i, 0] >= 0 and exit_array[exi, i, 1] >= 0 ):
                break
            else:
                exit_array[exi, i, 0] -= random_linha
                exit_array[exi, i, 1] -= random_coluna
    

# concatenate the two datasets into the final training set
X_train = np.vstack([entrance_array[0], entrance_array[1], entrance_array[2]])

exits_train = np.vstack([exit_array[0], exit_array[1], exit_array[2]])
exits_random = np.vstack([exit_array[0], exit_array[1], exit_array[2]])
np.random.shuffle(exits_random)

entrances_train = np.vstack([entrance_array[0], entrance_array[1], entrance_array[2]])
entrances_random = np.vstack([entrance_array[0], entrance_array[1], entrance_array[2]])
np.random.shuffle(entrances_random)

f = open('coordinates.csv', 'w')
np.savetxt(f, ['id', 'entr x', 'entr y', 'exit x', 'exit y'], newline=", ",  fmt="%s")
f.write("\n")

for x in range(len(X_train)):
    np.savetxt(f, [str(x), str(entrances_random[x, 0]), str(entrances_random[x, 1]),
                   str(exits_random[x, 0]), str(exits_random[x, 1])], newline=", ",  fmt="%s")
    f.write("\n")

f.close()
    
# fit a Gaussian Mixture Model with two components
clf = mixture.GaussianMixture(n_components=3, covariance_type='full')
clf.fit(X_train)
clf1 = mixture.GaussianMixture(n_components=3, covariance_type='full')
clf1.fit(exits_train)

labels = clf.predict(X_train)
#print(X_train)
#print(labels)

entrance_train0 = entrance_array[0]
entrance_train1 = entrance_array[1]
entrance_train2 = entrance_array[2]

exit_train0 = exit_array[0]
exit_train1 = exit_array[1]
exit_train2 = exit_array[2]

#exit_train0 = exits_random[:n_samples-1]
#exit_train1 = exits_random[n_samples:n_samples+n_samples-1]
#exit_train2 = exits_random[n_samples+n_samples:]

#print(exit_train0)
#print(exit_train1)
#print(exit_train2)

"""
X_train0 = np.zeros([1, 2])
X_train1 = np.zeros([1, 2])
X_train2 = np.zeros([1, 2])
flag_first_X_train0 = 0
flag_first_X_train1 = 0
flag_first_X_train2 = 0

for x in range (len(labels)):
    if(labels[x] == 0):
        #print(X_train[x])
        if(flag_first_X_train0 == 0):
            flag_first_X_train0 = 1
            X_train0[0] = X_train[x]
            exits_train
        else:
            X_train0 = np.vstack([X_train0, X_train[x]])    
            #print(X_train0)
    elif(labels[x] == 1):
        if(flag_first_X_train1 == 0):
            flag_first_X_train1 = 1
            X_train1[0] = X_train[x]
        else:
            X_train1 = np.vstack([X_train1, X_train[x]])
    elif(labels[x] == 2):
        if(flag_first_X_train2 == 0):
            flag_first_X_train2 = 1
            X_train2[0] = X_train[x]
        else:
            X_train2 = np.vstack([X_train2, X_train[x]])    
"""            
                        
#display predicted scores by the model as a contour plot
x = np.linspace(0, width-1, width)
y = np.linspace(0, height-1, height)
X, Y = np.meshgrid(x, y)

XX = np.array([X.ravel(), Y.ravel()]).T
Z = -clf.score_samples(XX)
Z = Z.reshape(X.shape)

Z1 = -clf1.score_samples(XX)
Z1 = Z1.reshape(X.shape)

print(Z[50, 50])

entrance_graph = plt.figure(1, figsize=(6,4))
entrance_graph.canvas.manager.window.wm_geometry("+%d+%d" % (1250, 100))

#CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=30.0),
#                 levels=np.logspace(0, 5, 50))
#CB = plt.colorbar(CS, shrink=0.8, extend='both')

#scatter = plt.scatter(X_train[:, 0], X_train[:, 1], 1, c='b')
scatter = plt.scatter(entrance_train0[:, 0], entrance_train0[:, 1], s=1**2, c ='r') 
scatter = plt.scatter(entrance_train1[:, 0], entrance_train1[:, 1], s=1**2, c ='g') 
scatter = plt.scatter(entrance_train2[:, 0], entrance_train2[:, 1], s=1**2, c ='b') 

ax = scatter.axes
#ax.invert_xaxis()
ax.invert_yaxis()

plt.title('entradas')
#plt.axis('tight')
plt.axis('equal')
plt.grid(True)
plt.savefig('entradas_random_distribution.png')

exit_graph = plt.figure(2, figsize=(6,4))
exit_graph.canvas.manager.window.wm_geometry("+%d+%d" % (1250, 570))

#CS = plt.contour(X, Y, Z1, norm=LogNorm(vmin=1.0, vmax=30.0),
#                 levels=np.logspace(0, 5, 50))
#CB = plt.colorbar(CS, shrink=0.8, extend='both')

#scatter = plt.scatter(exits_train[:, 0], exits_train[:, 1], 1, c='b')
scatter = plt.scatter(exit_train0[:, 0], exit_train0[:, 1], s=1**2, c ='r')
scatter = plt.scatter(exit_train1[:, 0], exit_train1[:, 1], s=1**2, c ='g')
scatter = plt.scatter(exit_train2[:, 0], exit_train2[:, 1], s=1**2, c ='b') 

ax = scatter.axes
#ax.invert_xaxis()
ax.invert_yaxis()

plt.title('saidas')
#plt.axis('tight')
plt.axis('equal')
plt.grid(True)
plt.savefig('saidas_random_distribution.png')

plt.show()



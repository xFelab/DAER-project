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

coordinates = pd.read_csv("coordinates.csv", sep=',', header = 0)
#print(coordinates.values)

train_entrances = np.zeros([len(coordinates), 2])
train_exits = np.zeros([len(coordinates), 2]) 

for x in range(len(coordinates)):
    train_entrances[x, 0] = coordinates.values[x, 1]
    train_entrances[x, 1] = coordinates.values[x, 2]
    train_exits[x, 0] = coordinates.values[x, 3]
    train_exits[x, 1] = coordinates.values[x, 4]

#print(train_entrances)
#print(train_exits)
                
# fit a Gaussian Mixture Model with two components
clf = mixture.GaussianMixture(n_components=entrances, covariance_type='full')
clf.fit(train_entrances)
clf1 = mixture.GaussianMixture(n_components=exits, covariance_type='full')
clf1.fit(train_exits)

labels_entrances = clf.predict(train_entrances)
labels_exits = clf1.predict(train_exits)
#print(labels_entrances)
#print(labels_exits)

entrances_dir_0 = np.array([[500,500]])
entrances_dir_1 = np.array([[500,500]])
entrances_dir_2 = np.array([[500,500]])

exits_dir_0 = np.array([[500,500]])
exits_dir_1 = np.array([[500,500]])
exits_dir_2 = np.array([[500,500]])

#separa os blocos para cores
for x in range(len(coordinates.values)):
    if(labels_entrances[x]==0):
        if(entrances_dir_0[0,0] == 500):
            entrances_dir_0[0] = train_entrances[x]
        else:
            entrances_dir_0 = np.vstack((entrances_dir_0, train_entrances[x]))
    if(labels_entrances[x]==1):
        if(entrances_dir_1[0,0] == 500):
            entrances_dir_1[0] = train_entrances[x]
        else:
            entrances_dir_1 = np.vstack((entrances_dir_1, train_entrances[x]))
    if(labels_entrances[x]==2):
        if(entrances_dir_2[0,0] == 500):
            entrances_dir_2[0] = train_entrances[x]
        else:
            entrances_dir_2 = np.vstack((entrances_dir_2, train_entrances[x]))

    if(labels_exits[x]==0):
        if(exits_dir_0[0,0] == 500):
            exits_dir_0[0] = train_exits[x]
        else:
            exits_dir_0 = np.vstack((exits_dir_0, train_exits[x]))
    if(labels_exits[x]==1):
        if(exits_dir_1[0,0] == 500):
            exits_dir_1[0] = train_exits[x]
        else:
            exits_dir_1 = np.vstack((exits_dir_1, train_exits[x]))
    if(labels_exits[x]==2):
        if(exits_dir_2[0,0] == 500):
            exits_dir_2[0] = train_exits[x]
        else:
            exits_dir_2 = np.vstack((exits_dir_2, train_exits[x]))

#save results to file
f = open('coordinates_sorted.csv', 'w')
np.savetxt(f, ['id', 'entr group', 'exit group'], newline=", ",  fmt="%s")
f.write("\n")

for x in range(len(labels_entrances)):
    np.savetxt(f, [str(x), str(labels_entrances[x]),
                   str(labels_exits[x])], newline=", ",  fmt="%s")
    f.write("\n")
f.close()

#display predicted scores by the model as a contour plot
x = np.linspace(0, width-1, width)
y = np.linspace(0, height-1, height)
X, Y = np.meshgrid(x, y)

XX = np.array([X.ravel(), Y.ravel()]).T
Z = -clf.score_samples(XX)
Z = Z.reshape(X.shape)

Z1 = -clf1.score_samples(XX)
Z1 = Z1.reshape(X.shape)

entrance_graph = plt.figure(1, figsize=(6,4))
entrance_graph.canvas.manager.window.wm_geometry("+%d+%d" % (1250, 100))

CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=30.0),
                 levels=np.logspace(0, 5, 50))
CB = plt.colorbar(CS, shrink=0.8, extend='both')

#scatter = plt.scatter(train_entrances[:, 0], train_entrances[:, 1], 1, c='b')
scatter = plt.scatter(entrances_dir_0[:, 0], entrances_dir_0[:, 1], s=1**2, c ='r') 
scatter = plt.scatter(entrances_dir_1[:, 0], entrances_dir_1[:, 1], s=1**2, c ='g') 
scatter = plt.scatter(entrances_dir_2[:, 0], entrances_dir_2[:, 1], s=1**2, c ='b') 

ax = scatter.axes
#ax.invert_xaxis()
ax.invert_yaxis()

plt.title('entrances')
#plt.axis('tight')
plt.axis('equal')
plt.axis([0, 352, 250, 0])
plt.savefig('entradas_clusterization.png')
plt.grid(True)

exit_graph = plt.figure(2, figsize=(6,4))
exit_graph.canvas.manager.window.wm_geometry("+%d+%d" % (1250, 570))

CS = plt.contour(X, Y, Z1, norm=LogNorm(vmin=1.0, vmax=30.0),
                 levels=np.logspace(0, 5, 50))
CB = plt.colorbar(CS, shrink=0.8, extend='both')

#scatter = plt.scatter(train_exits[:, 0], train_exits[:, 1], 1, c='b')
scatter = plt.scatter(exits_dir_0[:, 0], exits_dir_0[:, 1], s=1**2, c ='r')
scatter = plt.scatter(exits_dir_1[:, 0], exits_dir_1[:, 1], s=1**2, c ='g')
scatter = plt.scatter(exits_dir_2[:, 0], exits_dir_2[:, 1], s=1**2, c ='b') 

ax2 = scatter.axes
#ax.invert_xaxis()
ax2.invert_yaxis()

plt.title('exits')
#plt.axis('tight')
plt.axis('equal')
plt.axis([0, 352, 250, 0])
plt.grid(True)
plt.savefig('saidas_clusterization.png')

plt.show()

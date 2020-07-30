#example obtained from:
# - https://docs.w3cub.com/scikit_learn/auto_examples/mixture/plot_gmm_pdf/#sphx-glr-auto-examples-mixture-plot-gmm-pdf-py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture
import time

n_samples = 100

# generate random sample, two components
np.random.seed(int(time.time()))
#print(time.time())

# generate spherical data centered on (20, 30)
shifted_gaussian = np.random.randn(n_samples, 2)*10 + np.array([200, 200])
#print (shifted_gaussian)

# generate zero centered stretched Gaussian data
#C = np.array([[0., -0.7], [3.5, .7]])
#stretched_gaussian = np.dot(np.random.randn(n_samples, 2)*10, C)
stretched_gaussian = np.random.randn(n_samples, 2)*10 + np.array([50, 50])

# concatenate the two datasets into the final training set
X_train = np.vstack([shifted_gaussian, stretched_gaussian])
print(X_train)
# fit a Gaussian Mixture Model with two components
clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
clf.fit(X_train)

# display predicted scores by the model as a contour plot
x = np.linspace(0, 352-1, 352)
y = np.linspace(0, 250-1, 250)
X, Y = np.meshgrid(x, y)

XX = np.array([X.ravel(), Y.ravel()]).T
Z = -clf.score_samples(XX)
Z = Z.reshape(X.shape)

print(Z[50, 50])

CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=100.0),
                 levels=np.logspace(0, 5, 50))
CB = plt.colorbar(CS, shrink=0.8, extend='both')

plt.scatter(X_train[:, 0], X_train[:, 1], .8)

plt.title('probabilidade')
plt.axis('tight')
plt.show()

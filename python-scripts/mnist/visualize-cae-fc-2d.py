# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import cv2
import tsne
from sklearn.cluster import KMeans

caffe_root = '/home/lod/master-thesis/' # The caffe_root is changed to reflect the actual folder in the server.
sys.path.insert(0, caffe_root + 'python') # Correct the python path
import caffe

caffe.set_device(1)
caffe.set_mode_gpu()

model_def = str(sys.argv[1])
model_weights = str(sys.argv[2])

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

total_samples = 1500
targeted_samples =  830 # sum of empty, shredded, and folie samples
no_of_dimensions = 1000

b = np.arange(targeted_samples*no_of_dimensions,dtype=float).reshape(targeted_samples, no_of_dimensions)
labels = np.arange(targeted_samples).reshape(targeted_samples)

x=0
layer = str(sys.argv[3])

for j in range(0,total_samples):
    net.forward()
    label = net.blobs["label"].data[0]
#    if label == 0:
#        labels[x] = 0
#        b[x] = net.blobs[layer].data[0]
#        x = x + 1
    if label == 2:
        labels[x] = 0
        b[x] = net.blobs[layer].data[0]
        x = x + 1
    if label == 3:
        labels[x] = 1
        b[x] = net.blobs[layer].data[0]
        x = x + 1
    if label == 4:
        labels[x] = 2
        b[x] = net.blobs[layer].data[0]
        x = x + 1
#    print labels[j]
    iteration_count = 'Iteration: ' + repr(j)
Y = tsne.tsne(b, no_dims= 2, initial_dims=no_of_dimensions, perplexity=30.0)
n_clusters = 4
#apply kmeans
km = KMeans(n_clusters)
pred = km.fit_predict(b)
centroids = km.cluster_centers_
fig,ax = plt.subplots()
N = 3
cax=plt.scatter(Y[:, 0], Y[:, 1],20,labels,cmap=plt.cm.get_cmap('jet', N), alpha=1,edgecolor = 'face')
#plt.scatter(centroids[:, 0], centroids[:, 1],
#            marker='x', s=200, linewidths=6,
#            color='b', zorder=10)
cbar=plt.colorbar(ticks=[0,1,2])
plt.clim(-0.5,N-0.5)
cbar.ax.set_yticklabels(['Empty', 'Plastic Foil', 'Shredded Paper'])  # vertically oriented colorbar
cbar.set_label('Material Classes', rotation=270)
plt.title('Bi-Modal Network Concat Layer')
plt.xlabel('t-SNE Dimension-1')
plt.ylabel('t-SNE Dimension-2')
fig.tight_layout()
plt.savefig('18-07-2018-bi-modal-concat.png')

import metrics as met
y=labels
print('acc=', met.acc(y, pred), 'nmi=', met.nmi(y, pred), 'ari=', met.ari(y, pred))

plt.scatter(Y[:, 0], Y[:, 1],20,pred,cmap=plt.cm.get_cmap('jet', N), alpha=1,edgecolor = 'face')
plt.savefig('kmeans-output')

# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

# set the model definitions since we are using a pretrained network here.
# this protoype definitions can be changed to make significant changes in the learning method.
model_def = '/home/lod/master-thesis/examples/master-thesis/new_models/caeWithoutFClayer/building_model/adam-conv4-good-results/snapshots/snapshots-fused-28-05-2018-good-results-bn-used/good-3-conv5/train-3-conv5-just-img.prototxt'
model_weights = '/home/lod/master-thesis/examples/master-thesis/new_models/caeWithoutFClayer/building_model/adam-conv4-good-results/snapshots/_iter_50000.caffemodel'


net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

a = np.arange(1152,dtype=float).reshape(32, 36)
b = np.arange(1036800,dtype=float).reshape(900, 1152)
#b = np.arange(20480,dtype=float).reshape(10, 2048)
labels = np.arange(900).reshape(900)
print a.shape

for j in range(0,900):
    net.forward()
    for i in range(0,32):
        if j < 900:
            X = (net.blobs["conv5_r"].data[0,i])
            a[i]=np.concatenate((X[0,:],X[1,:],X[2,:],X[3,:],X[4,:],X[5,:]),axis=0)

    if j < 900:
        b[j] = np.concatenate((a[0,:],a[1,:],a[2,:],a[3,:],a[4,:],a[5,:],a[6,:],a[7,:],a[8,:],a[9,:]
                               ,a[10,:],a[11,:],a[12,:],a[13,:],a[14,:],a[15,:],a[16,:],a[17,:],a[18,:],a[19,:]
                               ,a[20,:],a[21,:],a[22,:],a[23,:],a[24,:],a[25,:],a[26,:],a[27,:],a[28,:],a[29,:]
                               ,a[30,:],a[31,:]),axis=0)

        labels[j] = (net.blobs["label"].data[0])
        print labels[j]
        iteration_count = 'Iteration: ' + repr(j)
        print iteration_count

Y = tsne.tsne(b, no_dims= 2, initial_dims=1152, perplexity=30.0)
n_clusters = 5

kmeans = KMeans(n_clusters).fit(Y)
fig,ax = plt.subplots()
centroids = kmeans.cluster_centers_
cax=ax.scatter(Y[:, 0], Y[:, 1],20,labels, cmap ='rainbow')
ax.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='b', zorder=10)

cbar = fig.colorbar(cax, ticks=[1, 2, 3, 4, 5])
cbar.ax.set_yticklabels(['shredded paper', 'folie', 'empty', 'cardboard', 'pamphlets'])  # vertically oriented colorbar
plt.savefig('17-06-2018.png')


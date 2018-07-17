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

# set the model definitions since we are using a pretrained network here.
# this protoype definitions can be changed to make significant changes in the learning method.
#model_def = '/home/lod/master-thesis/examples/master-thesis/new_models/caeWithoutFClayer/building_model/adam-conv4-good-results/feature_fusion/attempt-3-many-changes-but-poor-results/train-conv3-earlyfusion-update-27-05-2018.prototxt'
model_def = '/home/lod/master-thesis/examples/master-thesis/new_models/autoencoder_on_alexnet/train-autoencode-alexnet.prototxt'
#model_weights = '/home/lod/master-thesis/examples/master-thesis/new_models/caeWithoutFClayer/building_model/adam-conv4-good-results/feature_fusion/attempt-3-many-changes-but-poor-results/snapshots/_iter_40000.caffemodel'
model_weights = '/home/lod/master-thesis/examples/master-thesis/new_models/autoencoder_on_alexnet/snapshots/_iter_100000.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

no_of_samples = 1000
no_of_dimensions = 1000

b = np.arange(no_of_samples*no_of_dimensions,dtype=float).reshape(no_of_samples, no_of_dimensions)
labels = np.arange(no_of_samples).reshape(no_of_samples)

for j in range(0,no_of_samples):
    net.forward()
    b[j] = net.blobs["img/fc3"].data[0]               
    labels[j] = (net.blobs["label"].data[0])
#    print labels[j]
    iteration_count = 'Iteration: ' + repr(j)
#    print iteration_count
Y = tsne.tsne(b, no_dims= 3, initial_dims=no_of_dimensions, perplexity=30.0)

km = KMeans(n_clusters = 5)
pred = km.fit_predict(b)
    
import metrics as met
y=labels

print('acc=', met.acc(y, pred), 'nmi=', met.nmi(y, pred), 'ari=', met.ari(y, pred))

fig = plt.figure()
ax = plt.axes(projection='3d')
xs = Y[:,0]
ys = Y[:,1]
zs = Y[:,2]

print xs.shape
print ys.shape
print zs.shape
print labels.shape

ax.scatter3D(xs, ys, zs,s=20,c=labels,cmap = 'rainbow')

#cbar = fig.colorbar(cax, ticks=[0,1,2,3,4])
#cbar.ax.set_yticklabels(['Cardboard', 'Pamphlets', 'Empty', 'Plastic Foil', 'Shredded Paper'])  # vertically oriented colorbar

for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.savefig('results/30-06-2018_angle:' + str(angle)+'.png')


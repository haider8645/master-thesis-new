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

caffe.set_device(0)
caffe.set_mode_gpu()

# set the model definitions since we are using a pretrained network here.
# this protoype definitions can be changed to make significant changes in the learning method.
#model_def = '/home/lod/master-thesis/examples/master-thesis/new_models/caeWithoutFClayer/building_model/adam-conv4-good-results/feature_fusion/attempt-3-many-changes-but-poor-results/train-conv3-earlyfusion-update-27-05-2018.prototxt'
model_def = '/home/lod/master-thesis/examples/master-thesis/new_models/caeWithoutFClayer/building_model/adam-conv4-good-results/train-4-conv4-smaller.prototxt'
#model_weights = '/home/lod/master-thesis/examples/master-thesis/new_models/caeWithoutFClayer/building_model/adam-conv4-good-results/feature_fusion/attempt-3-many-changes-but-poor-results/snapshots/_iter_40000.caffemodel'
model_weights = '/home/lod/master-thesis/examples/master-thesis/new_models/caeWithoutFClayer/building_model/adam-conv4-good-results/snapshots/_iter_300000.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

no_of_samples = 1500
no_of_dimensions = 2498

b = np.arange(no_of_samples*no_of_dimensions,dtype=float).reshape(no_of_samples, no_of_dimensions)
labels = np.arange(no_of_samples).reshape(no_of_samples)

for j in range(0,no_of_samples):
    net.forward()
    b[j] = net.blobs["img_nir/concat"].data[0]                
    labels[j] = (net.blobs["label"].data[0])
    print labels[j]
    iteration_count = 'Iteration: ' + repr(j)
    print iteration_count
Y = tsne.tsne(b, no_dims= 2, initial_dims=no_of_dimensions, perplexity=30.0)


fig,ax = plt.subplots()

cax=ax.scatter(Y[:, 0], Y[:, 1],20,labels, cmap ='rainbow')
cbar = fig.colorbar(cax, ticks=[0,1,2,3,4])
cbar.ax.set_yticklabels(['Cardboard', 'Pamphlets', 'Empty', 'Plastic Foil', 'Shredded Paper'])  # vertically oriented colorbar

plt.title('Fused Network with BN')
#plt.suptitle('2048 dimensions mapped in 2D space using t-SN)
plt.xlabel('Dimension-1')
plt.ylabel('Dimension-2')
plt.savefig('28-05-2018-completeley-trained.png')


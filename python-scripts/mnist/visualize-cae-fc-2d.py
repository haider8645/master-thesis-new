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
#from sklearn.cluster import KMeans

caffe_root = '/home/lod/master-thesis/' # The caffe_root is changed to reflect the actual folder in the server.
sys.path.insert(0, caffe_root + 'python') # Correct the python path
import caffe

caffe.set_device(2)
caffe.set_mode_gpu()

# set the model definitions since we are using a pretrained network here.
# this protoype definitions can be changed to make significant changes in the learning method.
#model_def = '/home/lod/master-thesis/examples/master-thesis/new_models/caeWithoutFClayer/building_model/adam-conv4-good-results/feature_fusion/attempt-3-many-changes-but-poor-results/train-conv3-earlyfusion-update-27-05-2018.prototxt'
model_def = '/home/lod/master-thesis/examples/master-thesis/new_models/autoencoder_on_alexnet/train-autoencode-alexnet_img_nir_1.prototxt'
#model_weights = '/home/lod/master-thesis/examples/master-thesis/new_models/caeWithoutFClayer/building_model/adam-conv4-good-results/feature_fusion/attempt-3-many-changes-but-poor-results/snapshots/_iter_40000.caffemodel'
model_weights = '/home/lod/master-thesis/examples/master-thesis/new_models/autoencoder_on_alexnet/snapshots/img_nir/_iter_50000.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

no_of_samples = 2000
no_of_dimensions = 1000

b = np.arange(no_of_samples*no_of_dimensions,dtype=float).reshape(no_of_samples, no_of_dimensions)
labels = np.arange(no_of_samples).reshape(no_of_samples)

for j in range(0,no_of_samples):
    net.forward()
    label = net.blobs["label"].data[0]
    if label == 2:
        b[j] = net.blobs["img_nir/concat"].data[0]
        labels[j] = 1
    if label == 3:
        b[j] = net.blobs["img_nir/concat"].data[0]
        labels[j] = 2
    if label == 4:
        b[j] = net.blobs["img_nir/concat"].data[0]
        labels[j] = 3
    if label == 1:
        b[j] = net.blobs["img_nir/concat"].data[0]
        labels[j] = 0

    print labels[j]
    iteration_count = 'Iteration: ' + repr(j)
#    print iteration_count
Y = tsne.tsne(b, no_dims= 2, initial_dims=no_of_dimensions, perplexity=30.0)

fig,ax = plt.subplots()

N = 4

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

plt.scatter(Y[:, 0], Y[:, 1],20,labels, cmap = discrete_cmap(N,'rainbow'), alpha=0.5,edgecolor ='face')
cbar = plt.colorbar(ticks=[0,1,2,3])
plt.clim(-0.5, N - 0.5)
cbar.ax.set_yticklabels(['Pamphlets','Empty', 'Plastic Foil', 'Shredded Paper'])  # vertically oriented colorbar

plt.title('Bi-Modal 4 Classes')
plt.xlabel('Dimension-1')
plt.ylabel('Dimension-2')
plt.savefig('16-07-2018-bi-modal-concat.png')


# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import show, plot, draw
import sys
import os
import cv2
import tsne
import pylab
import caffe


caffe_root = '/home/lod/caffe/' # The caffe_root is changed to reflect the actual folder in the server.
sys.path.insert(0, caffe_root + 'python') # Correct the python path
caffe.set_device(0)
caffe.set_mode_gpu()

# set the model definitions since we are using a pretrained network here.
# this protoype definitions can be changed to make significant changes in the learning method.
model_def = '/home/lod/master-thesis/examples/master-thesis/new_models/caeWithoutFClayer/building_model/10-feature-maps/train-ten-featuremaps.prototxt'
model_weights = '/home/lod/master-thesis/examples/master-thesis/new_models/caeWithoutFClayer/building_model/10-feature-maps/snapshots-10-feature-maps/_iter_340000.caffemodel'


net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

a = np.arange(640,dtype=float).reshape(10, 64)
b = np.arange(480000,dtype=float).reshape(750, 640)
print a.shape

#index = 0
#item_index = 0

for j in range(750):
    
#    print j
    net.forward()
    #print net.blobs["conv4"].data.shape
    for i in range(0,10):
#        if i == 1:
        X = (net.blobs["conv4"].data[0,i])
#        print X[7,:]
        a[i]=np.concatenate((X[0,:],X[1,:],X[2,:],X[3,:],X[4,:],X[5,:],X[6,:],X[7,:]),axis=0)
 #       print a[j,:]
    b[j] = np.concatenate((a[0,:],a[1,:],a[2,:],a[3,:],a[4,:],a[5,:],a[6,:],a[7,:],a[8,:],a[9,:]),axis=0)
    print b[j]    

#    for k in range(0,7):
#        a[j,index] = X.item(item_index)
#        index = index + 1
#        item_index = item_index+1
#        print a[j]

Y = tsne.tsne(b, no_dims= 2, initial_dims=640, perplexity=30.0)
pylab.scatter(Y[:, 0], Y[:, 1],20)
#pylab.show()
pylab.savefig('cae-640.png')


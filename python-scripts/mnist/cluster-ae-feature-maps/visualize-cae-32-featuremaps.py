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


caffe_root = '/home/haider/caffe/' # The caffe_root is changed to reflect the actual folder in the server.
sys.path.insert(0, caffe_root + 'python') # Correct the python path
caffe.set_device(0)
caffe.set_mode_gpu()

# set the model definitions since we are using a pretrained network here.
# this protoype definitions can be changed to make significant changes in the learning method.
model_def = '/home/haider/Desktop/output comparison/from 10-03-2018/conv-4-deconv-4/architecture-2/train-4-conv-updated.prototxt'
model_weights = '/home/haider/Desktop/output comparison/from 10-03-2018/conv-4-deconv-4/architecture-2/_iter_200000.caffemodel'


net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

a = np.arange(2048,dtype=float).reshape(32, 64)
b = np.arange(1536000,dtype=float).reshape(750, 2048)
#b = np.arange(20480,dtype=float).reshape(10, 2048)
labels = np.arange(750).reshape(750)
print a.shape

for j in range(0,750):
    net.forward()
    for i in range(0,32):
        if j < 750:
            X = (net.blobs["conv4"].data[0,i])
            a[i]=np.concatenate((X[0,:],X[1,:],X[2,:],X[3,:],X[4,:],X[5,:],X[6,:],X[7,:]),axis=0)
#            print a[j,:]

            

    if j < 750:
        b[j] = np.concatenate((a[0,:],a[1,:],a[2,:],a[3,:],a[4,:],a[5,:],a[6,:],a[7,:],a[8,:],a[9,:]
                               ,a[10,:],a[11,:],a[12,:],a[13,:],a[14,:],a[15,:],a[16,:],a[17,:],a[18,:],a[19,:]
                               ,a[20,:],a[21,:],a[22,:],a[23,:],a[24,:],a[25,:],a[26,:],a[27,:],a[28,:],a[29,:]
                               ,a[30,:],a[31,:]),axis=0)
        print b[j]    
        labels[j] = (net.blobs["label"].data[0])
        print labels[j]
        iteration_count = 'Iteration: ' + repr(j)
        print iteration_count

Y = tsne.tsne(b, no_dims= 2, initial_dims=2048, perplexity=30.0)
pylab.scatter(Y[:, 0], Y[:, 1],20,labels)
pylab.colorbar()
pylab.show()


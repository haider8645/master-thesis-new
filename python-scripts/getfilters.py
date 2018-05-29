# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import cv2

caffe_root = '/home/lod/master-thesis/' # The caffe_root is changed to reflect the actual folder in the server.
sys.path.insert(0, caffe_root + 'python') # Correct the python path
import caffe

caffe.set_mode_gpu()
caffe.set_device(0)
model_def = '/home/lod/master-thesis/examples/master-thesis/new_models/caeWithoutFClayer/building_model/adam-conv4-good-results/train-4-conv4-smaller.prototxt'
model_weights = '/home/lod/master-thesis/examples/master-thesis/new_models/caeWithoutFClayer/building_model/adam-conv4-good-results/snapshots/snapshots-fused-28-05-2018-good-results-bn-used/_iter_300000.caffemodel'

net = caffe.Net(model_def,
                model_weights,           # defines the structure of the model,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

net.forward()
print (net.params["conv1"][0].data).shape
#print data.shape
#print data
#print net.blobs['conv1'].data[0,2]     


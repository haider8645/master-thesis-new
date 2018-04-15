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


model_def = '/home/lod/master-thesis/examples/master-thesis/new_models/caeWithoutFClayer/building_model/train-10-featuremaps-with-fc.prototxt'

net = caffe.Net(model_def,      # defines the structure of the model,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

print("Blobs:")
for name, blob in net.blobs.iteritems():
    print("{:<5}:  {}".format(name, blob.data.shape))

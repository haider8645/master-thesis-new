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
caffe.set_device(1)


model_def = '/home/lod/master-thesis/examples/master-thesis/new_models/autoencoder_on_alexnet/train-autoencode-alexnet-only-img-loss.prototxt'
model_weights = '/home/lod/master-thesis/examples/master-thesis/new_models/autoencoder_on_alexnet/snapshots/just_image_loss/_iter_15005.caffemodel'

net = caffe.Net(model_def,
                model_weights,           # defines the structure of the model,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)



print("Blobs:")
for name, blob in net.blobs.iteritems():
    print("{:<5}:  {}".format(name, blob.data.shape))

dirname = '/home/lod/master-thesis/graphs/output_kipro21052018'
for j in range(100):

    net.forward()

    for i in range(3):
            cv2.imwrite(os.path.join(dirname,'input_image_' + str(i) + str(j) + '.jpg'), 255 * net.blobs['data/img'].data[0,i])

    for i in range(3):
            cv2.imwrite(os.path.join(dirname,'output_' + str(i)+ str(j)+ '.jpg'), 255 * net.blobs['output'].data[0,i])



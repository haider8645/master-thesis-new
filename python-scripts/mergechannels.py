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


model_def = '/home/lod/master-thesis/examples/master-thesis/new_models/autoencoder_on_alexnet/train-autoencode-alexnet_img_nir_2_with_nir_scaled.prototxt'
model_weights = '/home/lod/master-thesis/examples/master-thesis/new_models/autoencoder_on_alexnet/snapshots/_iter_50000.caffemodel'

net = caffe.Net(model_def,
                model_weights,           # defines the structure of the model,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)



print("Blobs:")
for name, blob in net.blobs.iteritems():
    print("{:<5}:  {}".format(name, blob.data.shape))

dirname = '/home/lod/master-thesis/graphs/output_kipro21052018/'
for j in range(1000,1200):

    net.forward()
    b = 255*net.blobs['data/img'].data[0,0]
    g = 255*net.blobs['data/img'].data[0,1] 
    r = 255*net.blobs['data/img'].data[0,2]

    img = cv2.merge((b,g,r)) 
    cv2.imwrite(os.path.join(dirname,'input_image_' + str(j) + '.png'), img)   

    b = 255*net.blobs['output'].data[0,0]
    g = 255*net.blobs['output'].data[0,1]
    r = 255*net.blobs['output'].data[0,2]

    img = cv2.merge((b,g,r)) 
    cv2.imwrite(os.path.join(dirname,'output_image_' + str(j) + '.png'), img)   


# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import cv2

caffe_root = '/home/haider/caffe/' # The caffe_root is changed to reflect the actual folder in the server.
sys.path.insert(0, caffe_root + 'python') # Correct the python path
import caffe

caffe.set_mode_gpu()
caffe.set_device(0)


model_def = '/home/haider/Desktop/desktop/biba-server-updated-rsync/caeWithoutFClayer/building_model/adam-conv4-good-results/smaller-conv4-8-feature-maps-trained/train-4-conv4-smaller.prototxt'
model_weights = '/home/haider/Desktop/desktop/biba-server-updated-rsync/caeWithoutFClayer/building_model/adam-conv4-good-results/smaller-conv4-8-feature-maps-trained/snapshots/_iter_340000.caffemodel'


net = caffe.Net(model_def,
                model_weights,           # defines the structure of the model,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)



print("Blobs:")
for name, blob in net.blobs.iteritems():
    print("{:<5}:  {}".format(name, blob.data.shape))

dirname = '/home/haider/Desktop'
for j in range(7):

    net.forward()
    b = 255*net.blobs['data'].data[0,0]
    g = 255*net.blobs['data'].data[0,1] 
    r = 255*net.blobs['data'].data[0,2]

    img = cv2.merge((b,g,r)) 
    cv2.imwrite(os.path.join(dirname,'input_image_' + '.jpg'), img)   

    b = 255*net.blobs['deconv0'].data[0,0]
    g = 255*net.blobs['deconv0'].data[0,1]
    r = 255*net.blobs['deconv0'].data[0,2]

    img = cv2.merge((b,g,r)) 
    cv2.imwrite(os.path.join(dirname,'output_image_' + '.jpg'), img)   


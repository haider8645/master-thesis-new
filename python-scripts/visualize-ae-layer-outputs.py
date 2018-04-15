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
model_weights = '/home/lod/master-thesis/examples/master-thesis/new_models/caeWithoutFClayer/building_model/adam-conv4-good-results/snapshots/_iter_340000.caffemodel'


net = caffe.Net(model_def,
                model_weights,           # defines the structure of the model,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)



print("Blobs:")
for name, blob in net.blobs.iteritems():
    print("{:<5}:  {}".format(name, blob.data.shape))

dirname = '/home/lod/master-thesis/graphs/output_kipro15042018'
for j in range(11):

    net.forward()

    for i in range(3):
        if j == 5:
            cv2.imwrite(os.path.join(dirname,'input_image_' + str(i) + '.jpg'), 255*net.blobs['data'].data[0,i])
    
  #  for i in range(20):
  #      if j == 10:
  #          cv2.imwrite(os.path.join(dirname,'conv1_' + str(i) + '.jpg'), 255*net.blobs['conv1'].data[0,i])

  #  for i in range(20):
  #      if j == 10:
  #          cv2.imwrite(os.path.join(dirname,'pool1_' + str(i) + '.jpg'), 255*net.blobs['pool1'].data[0,i])
    
  #  for i in range(10):
  #      if j == 10:
  #          cv2.imwrite(os.path.join(dirname,'conv2_' + str(i) + '.jpg'), 255*net.blobs['conv2'].data[0,i])
  #  for i in range(10):
  #      if j == 10:
  #          cv2.imwrite(os.path.join(dirname,'pool2_' + str(i) + '.jpg'), 255*net.blobs['pool2'].data[

  #  for i in range(5):
  #      if j== 10:
  #          cv2.imwrite(os.path.join(dirname,'deconv3_' + str(i) + '.jpg'), 255*net.blobs['deconv3'].data[0,i]) 

#    for i in range(5):
#        if j== 10:
#            cv2.imwrite(os.path.join(dirname,'pool3_' + str(i) + '.jpg'), 255*net.blobs['pool3'].data[0,i])




#    for i in range(3):
#        if j== 10:
#            cv2.imwrite(os.path.join(dirname,'deconv1_' + str(i) + '.jpg'), 255*net.blobs['deconv1'].data[0,i])

    for i in range(3):
        if j== 5:
            cv2.imwrite(os.path.join(dirname,'deconv0_' + str(i) + '.jpg'), 255*net.blobs['deconv0'].data[0,i])


#    for i in range(10):
#        cv2.imwrite(os.path.join(dirname,'conv2_neuron_' + str(j) + '.jpg'), 255*net.blobs['conv2'].data[0,i])

#    for i in range(10):
#        cv2.imwrite(os.path.join(dirname,'conv3_neuron_' + str(j) + '.jpg'), 255*net.blobs['conv3'].data[0,i])

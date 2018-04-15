# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import cv2

# display plots in this notebook
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
caffe_root = '/home/haider/caffe/' # The caffe_root is changed to reflect the actual folder in the server.
sys.path.insert(0, caffe_root + 'python') # Correct the python path
import caffe
#matplotlib inline
# set display defaults
# these are for the matplotlib figure's.
#plt.rcParams['figure.figsize'] = (10, 10)        # large images
#plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
#plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap
caffe.set_device(0)
caffe.set_mode_gpu()
# set the model definitions since we are using a pretrained network here.
# this protoype definitions can be changed to make significant changes in the learning method.
model_def = '/home/haider/caffe/examples/master-thesis/trashnet/train-CAE100trashset.prototxt'
model_weights = '/home/haider/caffe/examples/master-thesis/snapshots_iter_19000.caffemodel'


net = caffe.Net(model_def,      # defines the structure of the model,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
for name, layer in zip(net._layer_names, net.layers):
    print("{:<7}: {:17s}({} blobs)".format(name, layer.type, len(layer.blobs)))
print("Blobs:")
for name, blob in net.blobs.iteritems():
    print("{:<5}:  {}".format(name, blob.data.shape))
#bs = len(fnames)
#print bs
#img = cv2.imread('mnist_five.png', 0)
#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#img_blobinp = img[np.newaxis, np.newaxis, :, :]
#net.blobs['data'].reshape(*img_blobinp.shape)
#net.blobs['data'].data[...] = img_blobinp
dirname = '/home/haider/caffe/graphs/output_cae100'
#os.mkdir(dirname)
for j in range(1):
#os.path.join(dirname, face_file_name)
    net.forward()
#    print net.blobs
    print net.blobs['data'].data[0,1]
    for i in range(1):
        cv2.imwrite(os.path.join(dirname,'input_image_' + str(j) + '.jpg'), 255*net.blobs['data'].data[0,i])

    for i in range(1):
        cv2.imwrite(os.path.join(dirname,'output_image_' + str(j) + '.jpg'), 255*net.blobs['deconv1neur'].data[0,i])

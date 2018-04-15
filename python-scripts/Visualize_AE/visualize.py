import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe
caffe.set_device(0)
caffe.set_mode_gpu()
net= caffe.Net('examples/master-thesis/train-mnistCAE12sym0302.prototxt',caffe.TEST)	
print net.inputs
print net.blobs['data']
print net.blobs['conv1']
print net.params['conv1'][0]
print net.params['conv1'][1]
print net.blobs['conv1'].data.shape
im = np.array(Image.open('examples/images/cat_gray.jpg'))
im_input = im[np.newaxis, np.newaxis, :, :]
net.blobs['data'].reshape(*im_input.shape)
net.blobs['data'].data[...] = im_input
net.forward()


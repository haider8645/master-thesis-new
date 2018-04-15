from PIL import Image
import os
import glob
import random
import numpy as np
import cv2
import caffe
from caffe.proto import caffe_pb2
import lmdb

#Size of images
IMAGE_WIDTH = 384
IMAGE_HEIGHT = 384

trash_pics = '/home/haider/caffe/python-scripts/trashnet/data-center-384/*.jpg'
addrs = glob.glob(trash_pics)
#print os.path.basename(addrs[1])
head, tail = os.path.split(addrs[1])
cropped=tail[0:6]
print cropped



print 'Length of addrs' 
print len(addrs)
# Divide the hata into 60% train, 20% validation, and 20% test
train_data = addrs[0:int(0.7*len(addrs))]
test_data = addrs[int(0.7*len(addrs)):]

print 'Length of train_addrs' 
print len(train_data)

print 'Length of test_addrs' 
print len(test_data)

'''
Title           :create_lmdb.py
Description     :This script divides the training images into 2 sets and stores them in lmdb databases for training and validation.
Author          :Adil Moujahid
Date Created    :20160619
Date Modified   :20160625
version         :0.2
usage           :python create_lmdb.py
python_version  :2.7.11
'''

import os
import glob
import random
import numpy as np

import cv2

import caffe
from caffe.proto import caffe_pb2
import lmdb

#Size of images
IMAGE_WIDTH = 127
IMAGE_HEIGHT = 127

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img


def make_datum(img):
    #image is numpy.ndarray format. BGR instead of RGB
    return caffe_pb2.Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
#        label=label,
        data=np.rollaxis(img, 2).tostring())

trash_pics = '/home/lod/master-thesis/LMDB-datasets/data-127/*.jpg'
addrs = glob.glob(trash_pics)
print 'Length of addrs' 
print len(addrs)
# Divide the hata into 60% train, 20% validation, and 20% test
train_data = addrs[0:int(0.7*len(addrs))]
#train_labels = labels[0:int(0.6*len(labels))]
#val_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addrs))]
#val_labels = labels[int(0.6*len(addrs)):int(0.8*len(addrs))]
test_data = addrs[int(0.7*len(addrs)):]
#test_labels = labels[int(0.8*len(labels)):]
print 'Length of train_addrs' 
print len(train_data)
#print 'Length of val_addrs' 
#print len(val_addrs)
print 'Length of test_addrs' 
print len(test_data)



train_lmdb = '/home/lod/master-thesis/LMDB-datasets/lmdb_127/train_lmdb'
test_lmdb = '/home//lod/master-thesis/LMDB-datasets/lmdb_127/test_lmdb'

os.system('rm -rf  ' + train_lmdb)
os.system('rm -rf  ' + test_lmdb)


#train_data = [img for img in glob.glob("/home/haider/caffe/python-scripts/train/*jpg")]
#test_data = [img for img in glob.glob("/home/haider/caffe/python-scripts/test1/*jpg")]

#Shuffle train_data
#random.shuffle(train_data)

#print train_data[0]
print 'Creating train_lmdb'

in_db = lmdb.open(train_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path in enumerate(train_data):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        datum = make_datum(img)
        in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
        print '{:0>5d}'.format(in_idx) + ':' + img_path
in_db.close()


print '\nCreating test_lmdb'

in_db = lmdb.open(test_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path in enumerate(test_data):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        datum = make_datum(img)
        in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
        print '{:0>5d}'.format(in_idx) + ':' + img_path
in_db.close()

print '\nFinished processing all images'

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
print 'Length of addrs' 
print len(addrs)
# Divide the hata into 60% train, 20% validation, and 20% test
train_data = addrs[0:int(0.7*len(addrs))]
test_data = addrs[int(0.7*len(addrs)):]

print 'Length of train_addrs' 
print len(train_data)

print 'Length of test_addrs' 
print len(test_data)



train_lmdb = '/home/haider/caffe/LMDB-datasets/trashnet384x384/train_lmdb'
test_lmdb = '/home/haider/caffe/LMDB-datasets/trashnet384x384/test_lmdb'

os.system('rm -rf  ' + train_lmdb)
os.system('rm -rf  ' + test_lmdb)


print train_data[0]
print 'Creating train_lmdb'

in_db = lmdb.open(train_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path in enumerate(train_data):
        im = np.array(Image.open(img_path)) # or load whatever ndarray you need
        im = im[:,:,::-1]
        im = im.transpose((2,0,1))
        im_dat = caffe.io.array_to_datum(im)
        in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())



        print '{:0>5d}'.format(in_idx) + ':' + img_path
in_db.close()


print '\nCreating test_lmdb'

in_db = lmdb.open(test_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path in enumerate(test_data):
        im = np.array(Image.open(img_path)) # or load whatever ndarray you need
        im = im[:,:,::-1]
        im = im.transpose((2,0,1))
        im_dat = caffe.io.array_to_datum(im)
        in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())       
        print '{:0>5d}'.format(in_idx) + ':' + img_path
in_db.close()

print '\nFinished processing all images'

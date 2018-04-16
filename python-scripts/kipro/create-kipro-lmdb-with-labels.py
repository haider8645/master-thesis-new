from PIL import Image
import os
import glob
import random
import numpy as np
import cv2
import caffe
from caffe.proto import caffe_pb2
import lmdb

trash_pics = '/home/lod/master-thesis/LMDB-datasets/kipro/hendrik-dataset-transformed/center_cropped_hendrik_dataset/test_train_images/*.png'
addrs = glob.glob(trash_pics)

# Divide the hata into 60% train, 20% validation, and 20% test
train_data = addrs[0:int(0.7*len(addrs))]
train_labels = addrs[0:int(0.7*len(addrs))]
test_data = addrs[int(0.7*len(addrs)):]
test_labels = addrs[int(0.7*len(addrs)):]
#validation_data = addrs[0:int(len(addrs))]
#validation_labels = addrs[0:int(len(addrs))]

train_lmdb = '/home/lod/master-thesis/LMDB-datasets/kipro/train_lmdb'
test_lmdb = '/home/lod/master-thesis/LMDB-datasets/kipro/test_lmdb'
#validation_lmdb = '/home/lod/master-thesis/LMDB-datasets/kipro/validation_lmdb'

os.system('rm -rf  ' + train_lmdb)
os.system('rm -rf  ' + test_lmdb)
#os.system('rm -rf  ' + validation_lmdb)

print 'Creating train_lmdb'

in_db = lmdb.open(train_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path in enumerate(train_data):
	head, tail = os.path.split(img_path) # read the path and file name
	cropped=tail[0:5] #crop out the first 5 characters from the file name
	print cropped
        label = cropped
        if label == "shred":
	    label_num=1
        if label == "folie":
            label_num=2
        if label == "empty":
	    label_num=3
        if label == "cardb":
            label_num=4   
        if label == "pamph":
	    label_num=5

        print label_num        
        im = np.array(Image.open(img_path)) # or load whatever ndarray you need
        im = im[:,:,::-1]
        im = im.transpose((2,0,1))
        im_dat = caffe.io.array_to_datum(im,label_num) # add label with the data to datum
        in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
        print '{:0>5d}'.format(in_idx) + ':' + img_path
in_db.close()
print '\nCreating test_lmdb'

in_db = lmdb.open(test_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path in enumerate(test_data):
	head, tail = os.path.split(img_path)
	cropped=tail[0:5]
	print cropped
        label = cropped
        if label == "shred":
	    label_num=1
        if label == "folie":
            label_num=2
        if label == "empty":
	    label_num=3
        if label == "cardb":
            label_num=4   
        if label == "pamph":
	    label_num=5
           
        print label_num         
        im = np.array(Image.open(img_path)) # or load whatever ndarray you need
        im = im[:,:,::-1]
        im = im.transpose((2,0,1))
        im_dat = caffe.io.array_to_datum(im,label_num)
        in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())       
        print '{:0>5d}'.format(in_idx) + ':' + img_path
in_db.close()
print 'Length of train_addrs' 
print len(train_data)

print 'Length of test_addrs' 
print len(test_data)

#print 'Length of validation-addrs'
#print len(validation_data)

print '\nFinished processing all images'

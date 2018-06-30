from PIL import Image
import os
import glob
import random
import numpy as np
import cv2
import sys
caffe_root = '/home/lod/master-thesis/' # The caffe_root is changed to reflect the actual folder in the se$
sys.path.insert(0, caffe_root + 'python') # Correct the python path
import caffe
from caffe.proto import caffe_pb2
import lmdb

def convert_lmdb_to_numpy():
    db_path = '/home/lod/master-thesis/LMDB-datasets/prepared_dataSets/dataSet_5c_250_5_2_eval/img_eval_db/'
   
    N = 15000

    x = np.zeros((N, 3, 227, 227 ), dtype=np.uint8)
    y = np.zeros(N, dtype=np.int64)
    j = 0

    lmdb_env = lmdb.open(db_path)    # equivalent to mdb_env_open()
    lmdb_txn = lmdb_env.begin()      # equivalent to mdb_txn_begin()
    lmdb_cursor = lmdb_txn.cursor()  # equivalent to mdb_cursor_open()
    for key, value in lmdb_cursor:
         datum = caffe_pb2.Datum()
         datum.ParseFromString(value)
         flat_x = np.fromstring(datum.data, dtype=np.uint8)
         # flat_x = np.fromstring(datum.data, dtype=np.float32)
         x[j] = flat_x.reshape(datum.channels, datum.height, datum.width)
         #image = caffe.io.datum_to_array(datum)
         #x[j] = image
         y[j] = datum.label
         print y[j]
         j=j+1
    return x,y

trash_net_pics = '/home/lod/datasets/trashnet/data/dataset-resized/resized_to_227/*.jpg'
addrs = glob.glob(trash_net_pics)
print 'Length of addrs' 
print len(addrs)

test_data = addrs[0:int(len(addrs)*2)]
test_labels = addrs[0:int(len(addrs)*2)]

print 'Length of test_addrs' 
print len(test_data)

test_lmdb = '/home/lod/datasets/trashnet/data/test_lmdb'
os.system('rm -rf  ' + test_lmdb)

print '\nCreating test_lmdb'

in_db = lmdb.open(test_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path in enumerate(test_data):
	head, tail = os.path.split(img_path)
	cropped=tail[0:5]
	print cropped
        label = cropped

        if label == "glass":
            label_num=5
           
        print label_num
           
        im = np.array(Image.open(img_path)) # or load whatever ndarray you need
        im = im[:,:,::-1]
        im = im.transpose((2,0,1))
        im_dat = caffe.io.array_to_datum(im,label_num)
        in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())       
        print '{:0>5d}'.format(in_idx) + ':' + img_path

    print 'Now adding KIPro Images'

    x = np.zeros((15000,3,227,227 ), dtype=np.uint8)
    y = np.zeros(15000, dtype=np.int64)

    x,y = convert_lmdb_to_numpy()
    j=0
    for in_idx in range (501, 1000):
        
        print 'label',y[j]
        print 'j',j 
        im = x[j] # or load whatever ndarray you need
        im_dat = caffe.io.array_to_datum(im,y[j])
        in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
        print '{:0>5d}'.format(in_idx)
        j=j+1
      
print 'x', len(x)
print 'y', len(y)
                                 
in_db.close()

print '\nFinished processing all images'

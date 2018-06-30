import lmdb
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import cv2
import os
import sys

caffe_root = '/home/lod/master-thesis' 
sys.path.insert(0, caffe_root + 'python')
import caffe

db_path = '/home/lod/datasets/trashnet/data/test_lmdb'
dirname = '/home/lod/master-thesis/LMDB-datasets/read_lmdb_outputs/'

N = 1000

x = np.zeros((N, 227, 227, 3), dtype=np.uint8)
y = np.zeros(N, dtype=np.int64)
j = 0

lmdb_env = lmdb.open(db_path)  # equivalent to mdb_env_open()
lmdb_txn = lmdb_env.begin()  # equivalent to mdb_txn_begin()
lmdb_cursor = lmdb_txn.cursor()  # equivalent to mdb_cursor_open()
for key, value in lmdb_cursor:
     datum = caffe.proto.caffe_pb2.Datum()
     datum.ParseFromString(value)
     image = np.zeros((datum.channels, datum.height, datum.width))
     image = caffe.io.datum_to_array(datum)
     image = np.transpose(image, (1, 2, 0))
     image = image[:, :, (2, 1, 0)]
     image = image.astype(np.uint8)
     matplotlib.image.imsave('/home/lod/master-thesis/LMDB-datasets/read_lmdb_outputs/data'+str(j), image)     
     x[j] = image
     print j
     j += 1 

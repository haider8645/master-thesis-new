import h5py
import numpy as np

filelist = []

image1 = '/home/haider/caffe/mnist_five.png'
image2 = '/home/haider/caffe/mnist_zero.png'
filename = '/tmp/my_hdf5%d.h5'
with h5py.File(filename, 'w') as f:
    f['data1'] = np.transpose(image1, (2, 0, 1))
    f['data2'] = np.transpose(image2, (2, 0, 1))
filelist.append(filename)
with open('/tmp/filelist.txt', 'w') as f:
    for filename in filelist:
        f.write(filename + '\n')

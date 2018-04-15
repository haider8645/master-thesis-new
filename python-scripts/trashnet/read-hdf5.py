import h5py
import numpy as np
import matplotlib.pyplot as plt

hdf5_path = '/home/haider/caffe/python scripts/trashnet/test.hdf5'

hdf5_file = h5py.File(hdf5_path, "r")

data_num = hdf5_file["data"].shape[0]

d = hdf5_file["data"]
print d.shape
print d.dtype
print d.size

i_s = 0
i_e = 10
images = hdf5_file["data"][i_s:i_e, ...]

#images = d[0:10, ...]
#print images[0]
#plt.imshow(images[5])
#plt.show()

hdf5_file.close()

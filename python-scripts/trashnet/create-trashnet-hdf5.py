################################
import numpy as np 
import h5py
import cv2
#dh5 = h5py.File('test.hdf5','r')
#print dh5.shape
#print dh5.dtype
#f = h5py.File("trash.hdf5", "w")
#dset=f.create_dataset("data", (1000,1),dtype = 'd')
#print dset.shape
#print dset.dtype
################################
from random import shuffle
import glob
shuffle_data = True
hdf5_path_train = '/home/haider/caffe/python scripts/trashnet/train.hdf5'
hdf5_path_test = '/home/haider/caffe/python scripts/trashnet/test.hdf5'
#trash_pics = '/home/haider/scripts-test-area/data/*.jpg'
trash_pics = '/home/haider/caffe/python scripts/trashnet/data/*.jpg'


addrs = glob.glob(trash_pics)

#if shuffle_data:
#    c = list(zip(addrs))
#    shuffle(c)
#    addrs = zip(*c)

#print addrs
print 'Length of addrs' 
print len(addrs)

# Divide the hata into 60% train, 20% validation, and 20% test
train_addrs = addrs[0:int(0.8*len(addrs))]
#train_labels = labels[0:int(0.6*len(labels))]
#val_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addrs))]
#val_labels = labels[int(0.6*len(addrs)):int(0.8*len(addrs))]
test_addrs = addrs[int(0.8*len(addrs)):]
#test_labels = labels[int(0.8*len(labels)):]

print 'Length of train_addrs' 
print len(train_addrs)
#print 'Length of val_addrs' 
#print len(val_addrs)
print 'Length of test_addrs' 
print len(test_addrs)

train_shape = (len(train_addrs), 3, 127, 127)
#val_shape = (len(val_addrs),3, 224, 224)
test_shape = (len(test_addrs),3, 127, 127)

hdf5_file_train = h5py.File(hdf5_path_train, mode='w')
hdf5_file_test = h5py.File(hdf5_path_test, mode='w')
hdf5_file_train.create_dataset("data", train_shape, np.int8)
#hdf5_file.create_dataset("val", val_shape, np.int8)
hdf5_file_test.create_dataset("data", test_shape, np.int8)
hdf5_file_train.create_dataset("train_mean", train_shape[1:], np.float32)
mean = np.zeros(train_shape[1:], np.float32)

# loop over train addresses
for i in range(len(train_addrs)):
    # print how many images are saved every 1000 images
    if i % 100 == 0 and i > 1:
        print 'Train data: {}/{}'.format(i, len(train_addrs))
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr = train_addrs[i]
    img = cv2.imread(addr)
    #img = cv2.resize(img, (127, 127), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # add any image pre-processing here
    # if the data order is Theano, axis orders should change
   # if data_order == 'th':
    img = np.rollaxis(img, 2,0)
    # save the image and calculate the mean so far
    hdf5_file_train["data"][i, ...] = img[None]
    #mean += img / float(len(train_labels))
# loop over validation addresses
#for i in range(len(val_addrs)):
    # print how many images are saved every 1000 images
#    if i % 100 == 0 and i > 1:
#        print 'Validation data: {}/{}'.format(i, len(val_addrs))
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
 #   addr = val_addrs[i]
#    img = cv2.imread(addr)
#    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
#    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # add any image pre-processing here
    # if the data order is Theano, axis orders should change
   # if data_order == 'th':
#    img = np.rollaxis(img, 2,0)
    # save the image
#    hdf5_file["val"][i, ...] = img[None]
# loop over test addresses
for i in range(len(test_addrs)):
    # print how many images are saved every 1000 images
    if i % 100 == 0 and i > 1:
        print 'Test data: {}/{}'.format(i, len(test_addrs))
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr = test_addrs[i]
    img = cv2.imread(addr)
    #img = cv2.resize(img, (127, 127), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # add any image pre-processing here
    # if the data order is Theano, axis orders should change
   # if data_order == 'th':
    img = np.rollaxis(img, 2,0)
    # save the image
    hdf5_file_test["data"][i, ...] = img[None]
# save the mean and close the hdf5 file
hdf5_file_train["train_mean"][...] = mean
#hdf5_file.close()hdf5_file["test_labels"][...] = test_labels

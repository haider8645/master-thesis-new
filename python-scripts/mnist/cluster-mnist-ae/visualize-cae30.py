# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import show, plot, draw
import sys
import os
import cv2
import tsne
import pylab

# display plots in this notebook
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
caffe_root = '/home/lod/master-thesis/' # The caffe_root is changed to reflect the actual folder in the server.
sys.path.insert(0, caffe_root + 'python') # Correct the python path
import caffe
#matplotlib inline
# set display defaults
# these are for the matplotlib figure's.
#plt.rcParams['figure.figsize'] = (10, 10)        # large images
#plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
#plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap
caffe.set_device(0)
caffe.set_mode_gpu()
# set the model definitions since we are using a pretrained network here.
# this protoype definitions can be changed to make significant changes in the learning method.
model_def = '/home/lod/master-thesis/examples/master-thesis/pooling-test/train-poolinglayer.prototxt'
model_weights = '/home/lod/master-thesis/examples/master-thesis/pooling-test/snapshots_iter_20000.caffemodel'


net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)


#print("Network layers:")
#for name, layer in zip(net._layer_names, net.layers):
#    print("{:<7}: {:17s}({} blobs)".format(name, layer.type, len(layer.blobs)))
#print("Blobs:")
#for name, blob in net.blobs.iteritems():
#    print("{:<5}:  {}".format(name, blob.data.shape))
#x_data_plot = []
#y_data_plot = []
#labels=[]
#X_old = 0

a = np.arange(150000).reshape(5000, 30)
print a.shape
labels = np.arange(5000).reshape(5000)
print labels.shape
for j in range(5000):
#    show(block=False)
    net.forward()
    X = (net.blobs["ip2encode"].data[0])
    print X
    a[j]=X
    print a[j]
    labels[j]=net.blobs["label"].data[0]

  
Y = tsne.tsne(a, no_dims= 2, initial_dims=30, perplexity = 30.0)
#fig = plt.figure()
pylab.scatter(Y[:, 0], Y[:, 1],20, labels)
#plt.plot(Y[:,0], Y[:,1])
pylab.savefig('cae-30plupl.png')






  
#print("output")
   # if net.blobs["label"].data[0] == 1:
   #     print "ONE"
#    T=net.blobs["label"].data[0]
#    labels.append(T[0])
#    print T[0]
#    print j
#    print T[0]
#    print X[0]
#    print X[1]

#    print X[0] - X_old
#    if T[0] == 0:
#        print "hit"
 #       plt.scatter(X[0],X[1],c="black",alpha=1.0)
#    if T[0] == 1:
#        print "hit"
 #       plt.scatter(X[0],X[1],c="red",alpha=1.0)
##    if T[0] == 2:
#        print "hit"
 #       plt.scatter(X[0],X[1],c="gold",alpha=1.0)
#    if T[0] == 3:
#        plt.scatter(X[0],X[1],c="blue",alpha=1.0)
#    if T[0] == 4:
#        plt.scatter(X[0],X[1],c="green",alpha=1.0)
 #   if T[0] == 5:
#        plt.scatter(X[0],X[1],c="orange",alpha=1.0)
#    if T[0] == 6:
#        plt.scatter(X[0],X[1],c="magenta",alpha=1.0)
#    if T[0] == 7:
#        plt.scatter(X[0],X[1],c="pink",alpha=1.0)
#    if T[0] == 8:
#        plt.scatter(X[0],X[1],c="brown",alpha=1.0)
 #   if T[0] == 9:
#        plt.scatter(X[0],X[1],c="yellow",alpha=1.0)
##    plt.draw()
#    X_old = X[0]
#plt.show()
#labels = ("zero","one","two","three","four","five","six","seven","eight","nine")
#colors = ("black","red","gold","blue","indigo","orange","darkblue","crimson","lightpink","olive")
#fig = plt.figure()
#ax = fig.add_subplot(1, 1, 1, axisbg="1.0")

#for data, color, group in zip(data, colors, labels):
#    x, y = data
#    ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)

#plt.title('Matplot scatter plot')
#plt.legend(loc=2)
#plt.show()


#plt.scatter(x_data_plot,y_data_plot,s=50,c=colors,alpha = 0.8)
#print X[0]
#print X[1]
#fig = plt.figure()
#ax = fig.add_subplot(1,1,1,axisbg="1.0")
#for data_x,data_y,lab in zip(x_data_plot,y_data_plot,labels):
#ax.scatter(data_x,data_y,label=lab)
#plt.show()

#plt.plot(x_data_plot,y_data_plot,'ro')
#plt.show()



#bs = len(fnames)
#print bs
#img = cv2.imread('mnist_five.png', 0)
#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#img_blobinp = img[np.newaxis, np.newaxis, :, :]
#net.blobs['data'].reshape(*img_blobinp.shape)
#net.blobs['data'].data[...] = img_blobinp
#dirname = '/home/lod/master-thesis/graphs/output_cae02'
#os.mkdir(dirname)
#for j in range(10):
#os.path.join(dirname, face_file_name)
#    net.forward()
#    for i in range(1):
#        cv2.imwrite(os.path.join(dirname,'input_image_' + str(j) + '.jpg'), 255*net.blobs['data'].data[0,i])

#    for i in range(1):
#        cv2.imwrite(os.path.join(dirname,'output_image_' + str(j) + '.jpg'), 255*net.blobs['deconv1neur'].data[0,i])
||||||| merged common ancestors
=======
# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import show, plot, draw
import sys
import os
import cv2
import tsne
import pylab

# display plots in this notebook
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
caffe_root = '/home/haider/caffe/' # The caffe_root is changed to reflect the actual folder in the server.
sys.path.insert(0, caffe_root + 'python') # Correct the python path
import caffe
#matplotlib inline
# set display defaults
# these are for the matplotlib figure's.
#plt.rcParams['figure.figsize'] = (10, 10)        # large images
#plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
#plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap
caffe.set_device(0)
caffe.set_mode_gpu()
# set the model definitions since we are using a pretrained network here.
# this protoype definitions can be changed to make significant changes in the learning method.
model_def = '/home/haider/Desktop/biba-trained-models/cae--30/train-mnistCAE30.prototxt'
model_weights = '/home/haider/Desktop/biba-trained-models/cae--30/snapshots_iter_30000.caffemodel'


net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)


print("Network layers:")
for name, layer in zip(net._layer_names, net.layers):
    print("{:<7}: {:17s}({} blobs)".format(name, layer.type, len(layer.blobs)))
print("Blobs:")
for name, blob in net.blobs.iteritems():
    print("{:<5}:  {}".format(name, blob.data.shape))
x_data_plot = []
y_data_plot = []
#labels=[]
X_old = 0

a = np.arange(150000).reshape(5000, 30)
print a.shape
labels = np.arange(5000).reshape(5000)
print labels.shape
for j in range(5000):
#    show(block=False)
    net.forward()
    X = (net.blobs["ip2encode"].data[0])
    print X
    a[j]=X
    print a[j]
    labels[j]=net.blobs["label"].data[0]

  
Y = tsne.tsne(a, no_dims= 2, initial_dims=30, perplexity=30.0)
pylab.scatter(Y[:, 0], Y[:, 1],20, labels)
pylab.show()






  
#print("output")
   # if net.blobs["label"].data[0] == 1:
   #     print "ONE"
#    T=net.blobs["label"].data[0]
#    labels.append(T[0])
#    print T[0]
#    print j
#    print T[0]
#    print X[0]
#    print X[1]

#    print X[0] - X_old
#    if T[0] == 0:
#        print "hit"
 #       plt.scatter(X[0],X[1],c="black",alpha=1.0)
#    if T[0] == 1:
#        print "hit"
 #       plt.scatter(X[0],X[1],c="red",alpha=1.0)
##    if T[0] == 2:
#        print "hit"
 #       plt.scatter(X[0],X[1],c="gold",alpha=1.0)
#    if T[0] == 3:
#        plt.scatter(X[0],X[1],c="blue",alpha=1.0)
#    if T[0] == 4:
#        plt.scatter(X[0],X[1],c="green",alpha=1.0)
 #   if T[0] == 5:
#        plt.scatter(X[0],X[1],c="orange",alpha=1.0)
#    if T[0] == 6:
#        plt.scatter(X[0],X[1],c="magenta",alpha=1.0)
#    if T[0] == 7:
#        plt.scatter(X[0],X[1],c="pink",alpha=1.0)
#    if T[0] == 8:
#        plt.scatter(X[0],X[1],c="brown",alpha=1.0)
 #   if T[0] == 9:
#        plt.scatter(X[0],X[1],c="yellow",alpha=1.0)
##    plt.draw()
#    X_old = X[0]
#plt.show()
#labels = ("zero","one","two","three","four","five","six","seven","eight","nine")
#colors = ("black","red","gold","blue","indigo","orange","darkblue","crimson","lightpink","olive")
#fig = plt.figure()
#ax = fig.add_subplot(1, 1, 1, axisbg="1.0")

#for data, color, group in zip(data, colors, labels):
#    x, y = data
#    ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)

#plt.title('Matplot scatter plot')
#plt.legend(loc=2)
#plt.show()


#plt.scatter(x_data_plot,y_data_plot,s=50,c=colors,alpha = 0.8)
#print X[0]
#print X[1]
#fig = plt.figure()
#ax = fig.add_subplot(1,1,1,axisbg="1.0")
#for data_x,data_y,lab in zip(x_data_plot,y_data_plot,labels):
#ax.scatter(data_x,data_y,label=lab)
#plt.show()

#plt.plot(x_data_plot,y_data_plot,'ro')
#plt.show()



#bs = len(fnames)
#print bs
#img = cv2.imread('mnist_five.png', 0)
#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#img_blobinp = img[np.newaxis, np.newaxis, :, :]
#net.blobs['data'].reshape(*img_blobinp.shape)
#net.blobs['data'].data[...] = img_blobinp
#dirname = '/home/lod/master-thesis/graphs/output_cae02'
#os.mkdir(dirname)
#for j in range(10):
#os.path.join(dirname, face_file_name)
#    net.forward()
#    for i in range(1):
#        cv2.imwrite(os.path.join(dirname,'input_image_' + str(j) + '.jpg'), 255*net.blobs['data'].data[0,i])

#    for i in range(1):
#        cv2.imwrite(os.path.join(dirname,'output_image_' + str(j) + '.jpg'), 255*net.blobs['deconv1neur'].data[0,i])
>>>>>>> abf168246847a24b7d208e847f053cc7c28f8559

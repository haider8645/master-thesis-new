# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import cv2
import tsne
from sklearn.cluster import KMeans

caffe_root = '/home/lod/master-thesis/' # The caffe_root is changed to reflect the actual folder in the server.
sys.path.insert(0, caffe_root + 'python') # Correct the python path
import caffe

caffe.set_device(0)
caffe.set_mode_gpu()

# set the model definitions since we are using a pretrained network here.
# this protoype definitions can be changed to make significant changes in the learning method.
model_def =     '/home/lod/master-thesis/examples/master-thesis/new_models/caeWithoutFClayer/building_model/adam-conv4-good-results//snapshots/snapshots-fused-28-05-2018-good-results-bn-used/train-3-conv5.prototxt'
model_weights = '/home/lod/master-thesis/examples/master-thesis/new_models/caeWithoutFClayer/building_model/adam-conv4-good-results/snapshots/_iter_300000.caffemodel'


net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

no_of_dimensions = 900
no_of_samples = 1500
b = np.arange(no_of_samples*no_of_dimensions,dtype=float).reshape(no_of_samples, no_of_dimensions)
labels = np.arange(no_of_samples).reshape(no_of_samples)

number_of_specific_samples = 260

activations= np.arange(number_of_specific_samples * no_of_dimensions,dtype=float).reshape(number_of_specific_samples,no_of_dimensions)


label = 4
count = 0

for j in range(0,no_of_samples):

    net.forward()

    labels[j] = net.blobs["label"].data[0]
    if labels[j] == label: 
        activations[count] = net.blobs["img_nir/concat"].data[0]
        count = count + 1
    iteration_count = 'Iteration: ' + repr(j)
    print iteration_count

print count 

fired_neurons_position = []

fired_neurons=0

for i in range(0,number_of_specific_samples):

    for j in range(0,no_of_dimensions):
        if activations[i][j]>0:
            fired_neurons_position.append(j)

fired_neurons_counts = []


for k in range(0,no_of_dimensions):
    fired_neurons_counts.append(fired_neurons_position.count(k))    

#print fired_neurons_counts[0]

fig = plt.figure()
N = len(fired_neurons_counts)
x = range(N)
width = 1
plt.title('concat layer activations for shredded paper')
plt.bar(x, fired_neurons_counts, width, color="blue")
plt.xlabel('neurons')
plt.ylabel('activation count')
#print "Neuron 1 fired: ", fired_neurons_position.count(1) , "times"
plt.savefig("test_histogram")
#print fired_neurons_position
         
        

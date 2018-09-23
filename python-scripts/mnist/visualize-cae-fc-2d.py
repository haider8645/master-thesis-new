# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import cv2
import tsne
from sklearn.cluster import KMeans

caffe_root = '/home/lod/master-thesis/' # The caffe_root is changed to reflect the actual folder in the server.
sys.path.insert(0, caffe_root + 'python') # Correct the python path
import caffe

caffe.set_device(1)
caffe.set_mode_gpu()

model_def = str(sys.argv[1])
model_weights = str(sys.argv[2])

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

#total_samples = 5000
targeted_samples =  1000 # sum of empty, shredded, and folie samples
no_of_dimensions = 1000

b = np.arange(targeted_samples*no_of_dimensions,dtype=float).reshape(targeted_samples, no_of_dimensions)
labels = np.arange(targeted_samples).reshape(targeted_samples)

#x=0
#layer = str(sys.argv[3])

for j in range(0,targeted_samples):
    net.forward()
    labels[j] = net.blobs["label"].data[0]
    b[j] = net.blobs["img_nir/concat"].data[0]

#    if label == 0:
#        labels[x] = 0
#        b[x] = net.blobs[layer].data[0]
#        x = x + 1
#    if label == 2:
#        labels[x] = 0
#        b[x] = net.blobs[layer].data[0]
#        x = x + 1
#    if label == 3:
#        labels[x] = 1
#        b[x] = net.blobs[layer].data[0]
#        x = x + 1
#    if label == 4:
#        labels[x] = 2
#        b[x] = net.blobs[layer].data[0]
#        x = x + 1
#    print labels[j]
    iteration_count = 'Iteration: ' + repr(j)
Y = tsne.tsne(b, no_dims= 2, initial_dims=no_of_dimensions, perplexity=30.0)
n_clusters = 5
#apply kmeans
km = KMeans(n_clusters)
pred = km.fit_predict(b)
centroids = km.cluster_centers_

MEDIUM = 14
BIG = 14
SMALL = 9

plt.rc('font', size=MEDIUM)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM)    # legend fontsize
plt.rc('figure', titlesize=BIG)  # fontsize of the figure title
from matplotlib import rcParams
rcParams['axes.titlepad'] = 20
plt.rcParams["font.family"] = "serif"


fig,ax = plt.subplots()
N = 5
cax=plt.scatter(Y[:, 0], Y[:, 1],20,labels,cmap=plt.cm.get_cmap('jet', N), alpha=1,edgecolor = 'face')
#plt.scatter(centroids[:, 0], centroids[:, 1],
#            marker='x', s=200, linewidths=6,
#            color='b', zorder=10)
cbar=plt.colorbar(ticks=[0,1,2,3,4])
plt.clim(-0.5,N-0.5)
ticks_mnist = ['zero','one','two', 'three','four','five','six','seven','eight','nine']
ticks_kipro = ['Cardboard','Pamphlets','Empty', 'Plastic Foil', 'Shredded Paper']
cbar.ax.set_yticklabels(ticks_kipro)  # vertically oriented colorbar
cbar.ax.tick_params(labelsize=14)
#cbar.set_label('Digits', rotation=270)
plt.title('Result of Split Concatenation')
plt.xlabel('t-SNE dimension-1')
plt.ylabel('t-SNE dimension-2')
fig.tight_layout()
plt.savefig('img_nir_1_fine_tuned_alexnet.png')

def score_(y_train,pred):
    from sklearn.metrics import homogeneity_score,completeness_score,v_measure_score
    import metrics as met
    print('acc=', met.acc(y_train, pred), 'nmi=', met.nmi(y_train, pred), 'ari=', met.ari(y_train, pred))
    print 'homegeniety score: ', homogeneity_score(y_train,pred)
    print 'completeness_score: ', completeness_score(y_train,pred)
    print 'v_measure_score: ', v_measure_score(y_train,pred)
print 'kmeans scores'
score_(labels,pred)

def GMM_(x_train,y_train,N):
    from sklearn.mixture import GMM
    gmm = GMM(n_components=N).fit(x_train)
    pred = gmm.predict(x_train)
    score_(y_train,pred)
    return pred
print 'gmm scores'
pred_gmm = GMM_(b,labels,N)

plt.gcf().clear()
plt.scatter(Y[:, 0], Y[:, 1],20,pred,cmap=plt.cm.get_cmap('jet', N), alpha=1,edgecolor = 'face')

cbar=plt.colorbar(ticks=[0,1,2,3,4,5,6,7,8,9])
plt.clim(-0.5,N-0.5)
cbar.ax.set_yticklabels(ticks_mnist)  # vertically oriented colorbar
cbar.ax.tick_params(labelsize=14)
#cbar.set_label('Digits', rotation=270)
plt.title('Finetuned Bimodal Auto-encoder + K-means + t-SNE')
plt.xlabel('t-SNE dimension-1')
plt.ylabel('t-SNE dimension-2')
#fig.tight_layout()
plt.rcParams["font.family"] = "serif"
plt.savefig('ae_kmeans_alexnet.png')

plt.gcf().clear()
plt.scatter(Y[:, 0], Y[:, 1],20,pred_gmm,cmap=plt.cm.get_cmap('jet', N), alpha=1,edgecolor = 'face')
cbar=plt.colorbar(ticks=[0,1,2,3,4])
plt.clim(-0.5,N-0.5)
cbar.ax.set_yticklabels(ticks_kipro)  # vertically oriented colorbar
cbar.ax.tick_params(labelsize=14)
#cbar.set_label('Digits', rotation=270)
plt.title('Finetuned Bimodal Auto-encoder + GMM + t-SNE')
plt.xlabel('t-SNE dimension-1')
plt.ylabel('t-SNE dimension-2')
#fig.tight_layout()
plt.savefig('ae_gmm_pred.png')



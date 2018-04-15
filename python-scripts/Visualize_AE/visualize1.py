import glob
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
import caffe
from caffe.proto import caffe_pb2
from caffe.io import blobproto_to_array
import argparse

def main(deploy_file, model_file, layer_name):
	net = caffe.Net(deploy_file, model_file, caffe.TEST)

	kernels = net.params[layer_name][0].data

	sampling_interval = 1
	if kernels.shape[1] == 3:
		size = int(math.ceil(math.sqrt(kernels.shape[0])))
		samples = xrange(kernels.shape[0])
	else:
		N = kernels.shape[0] * kernels.shape[1];
		if N > 100:
			size = 10
			samples = np.random.choice(xrange(N), 100, replace=False)
		else:
			size = int(math.ceil(math.sqrt(N)))
			samples = xrange(N)
			
		sampling_interval = int(math.ceil(float(kernels.shape[0]) * kernels.shape[1] / size / size))
		
	gs = gridspec.GridSpec(size, size)
	
	if kernels.shape[1] == 3: # HACK for color image
		for i in xrange(kernels.shape[0]):
			g = gs[i]
			ax = plt.subplot(g)
			ax.grid()
			ax.set_xticks([])
			ax.set_yticks([])
			img = (kernels[i][0] + kernels[i][1] + kernels[i][2]) / 3
			if kernels[i][0].shape[0] == 1 and kernels[i][0].shape[1] == 1:
				ax.imshow(np.array(img, dtype=np.float32), vmin=-0.1, vmax=0.1, cmap='Greys_r', interpolation="nearest")
			else:
				ax.imshow(np.array(img, dtype=np.float32), cmap='Greys_r', interpolation="nearest")
	else:
		for r in xrange(size):
			for c in xrange(size):
				s = r * size + c
				if s >= N: break
				
				i = samples[s] / kernels.shape[1]
				j = samples[s] % kernels.shape[1]
								
				g = gs[s]
				ax = plt.subplot(g)
				ax.grid()
				ax.set_xticks([])
				ax.set_yticks([])
				if kernels[i][j].shape[0] == 1 and kernels[i][j].shape[1] == 1:
					ax.imshow(np.array(kernels[i][j], dtype=np.float32), vmin=-0.1, vmax=0.1, cmap='Greys_r', interpolation="nearest")
				else:
					ax.imshow(np.array(kernels[i][j], dtype=np.float32), cmap='Greys_r', interpolation="nearest")

	plt.show()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("deploy_file", help="deploy file")
	parser.add_argument("model_file", help="model file")
	parser.add_argument("layer_name", help="layer name")
	args = parser.parse_args()
	

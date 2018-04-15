import numpy as np
import bhtsne

data = np.loadtxt("mnist2500_X.txt", skiprows=1)

print data.shape[1]

#print data.ndim

#print data.size

#print data[0]

a = np.arange(40000).reshape(10000, 4)

#print ("size of a: ", a.shape) 

#for i in range(0,10):
 #   print a[i]

#print data.data

embedding_array = bhtsne.run_bh_tsne(data, initial_dims=data.shape[1])

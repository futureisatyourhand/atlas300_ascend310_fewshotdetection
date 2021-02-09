import numpy as np

a = np.load('tmp.npy')
#a = np.load('/home/liqian/inferences/InferLib.bak/data/atoms.npy')


print (a.shape, a.dtype, type(a))

a = a.astype(np.float32)
#a = a[0].astype(np.float32)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(formatter={'float': '{: 0.7f}'.format})
print (a)

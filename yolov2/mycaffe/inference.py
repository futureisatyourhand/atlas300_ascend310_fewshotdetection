import caffe
import numpy as np
deploy="demo.prototxt"
caffe_model="new_parameter.caffemodel"
net=caffe.Net(deploy,caffe_model,caffe.TEST)
im=np.load("../005767.npy")#("/home/huawei/chems/bioavailability_model/atlas_data/005767.npy")
#im=np.ones((1,3,416,416))
print(im.shape)
#im=np.transpose(im,[0,3,1,2])
#im=np.array([im,im])
#im=np.concanate((im,im),0)
#im=np.tile(im,(2,1,1,1))
#im=np.concatenate((im, im), axis=0)
#im=np.expand_dims(im,axis=-1)
net.blobs['data'].data[...]=np.array(im,dtype=np.float32)
out=net.forward()
print(net.params.keys())
#a=net.params['layer22-conv'][0].data
a=net.blobs['layer23-conv'].data
np.save("reorg-concat.npy",a)
#a=np.transpose(a,[0,1,3,2])
#dd=a.copy()
#for i in range(128):
#    for j in range(13):
#        dd[i,:,:][:]=a[int(2*i),,:].copy()
#        dd[int(256+i),:,:][:]=a[int(2*i+1),:,:].copy()
print(a.shape)
#output1=np.transpose(output1,[0,3,1,2])
#a=np.reshape(a,[-1])
for i in range(13):
    print(i,a[0,0,i,:],a[1,29,i,:])
    
#a=np.reshape(a,[-1])
#for i in range(90,180):
#    print("scale===",dd[i],i,a[i])
#print(out.reshape([-1])[:90])
#print(np.reshape(out['dyconv-scale'],[-1])[:90])

#x=np.reshape(im,[-1])
#for i in range(int(x.shape[0]//2)):
#    print(x[i],x[int(i+x.shape[0]//2)])

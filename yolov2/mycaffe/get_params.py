#coding=utf-8
import os
import caffe
import csv
import numpy as np
import pickle
# np.set_printoptions(threshold='nan')

##to get all npy

MODEL_FILE = 'demo.prototxt'

#PRETRAIN_FILE = 'snapshot/solver_iter_1.caffemodel'
PRETRAIN_FILE = '../voc_weights/2shot/snapshot/solver_iter_1.caffemodel'
with open("../voc_weights/5shot/voc_novel2_5shot.pkl",'rb') as f:
    dyconv_weight=np.array(pickle.load(f))[0]
    f.close()
net = caffe.Net(MODEL_FILE,caffe.TEST, weights=PRETRAIN_FILE)

p = []

##init to parmeter according ir.py

flag=0
for param_name in net.params.keys():
    print("===================",param_name,net.params[param_name][0].data.shape)
    if 'dyconv-scale' in param_name:
        #with open("../voc_novel2_.pkl",'rb') as f:
        #    weight=np.reshape(np.array(pickle.load(f))[0],[-1,1,1,1])
        #    print(weight)
        #    f.close()
        print(net.params[param_name][0].data.shape)
        #net.params[param_name][0].data[:]=dyconv_weight.reshape([-1,])
        idsx=param_name.split('dyconv-scale')[-1]
        net.params[param_name][0].data[:]=dyconv_weight[int(idsx)-1].reshape([-1])
        continue        
    if '-conv' in param_name:
        ids=param_name[:-5].split('layer')[-1]
        weight="../voc_weights/5shot/detect/conv"+ids+"_weight.npy"
        #if 'layer22-conv' is param_name:
        #    aa=np.load(weight)
        #    ww=aa
        #    for i in range(128):
        #         ww[:,int(2*i),:,:]=aa[:,i,:,:]
        #         ww[:,int(2*i+1),:,:]=aa[:,256+i,:,:]
        #    net.params[param_name][0].data[:]=ww
        #else:
        #if 'layer22-conv' in param_name:
        #    net.params[param_name][0].data[:]=np.load(weight)*10.
        #    print("--------------------------------------")
        #else:
        net.params[param_name][0].data[:]=np.load(weight)
        #bias="../voc_weights/conv"+ids+"_bias.npy"
        #if os.path.exists(bias):
        #    net.params[param_name][1].data[:]=np.load(bias)       
        #    print("==============bias==========",np.load(bias).shape) 
    elif '-bn' in param_name:
        print("bn:",net.params[param_name][0].data.shape)
        continue
    elif '-scale' in param_name:
        ids=param_name[:-6].split('layer')[-1]
        weight="../voc_weights/5shot/detect/conv"+ids+"_bn_gamma.npy"
        bias="../voc_weights/5shot/detect/conv"+ids+"_bn_beta.npy"
        net.params[param_name][0].data[:]=np.load(weight)
        net.params[param_name][1].data[:]=np.load(bias)
        
net.save('../voc_weights/5shot/5shot_parameter.caffemodel')


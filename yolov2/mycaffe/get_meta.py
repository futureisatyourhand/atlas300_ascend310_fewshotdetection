#coding=utf-8
import os
import caffe
import csv
import numpy as np
import pickle
# np.set_printoptions(threshold='nan')

##to get all npy

MODEL_FILE = 'meta_test.prototxt'

#PRETRAIN_FILE = 'snapshot/solver_iter_1.caffemodel'
PRETRAIN_FILE = 'meta/solver_iter_1.caffemodel'
net = caffe.Net(MODEL_FILE,caffe.TEST, weights=PRETRAIN_FILE)

p = []

##init to parmeter according ir.py

flag=0
for param_name in net.params.keys():
    print("===================",param_name,net.params[param_name][0].data.shape)
    if '-conv' in param_name:
        ids=param_name[:-5].split('layer')[-1]
        weight="../voc_weights/meta/conv"+ids+"_weight.npy"
        net.params[param_name][0].data[:]=np.load(weight)
        bias="../voc_weights/meta/conv"+ids+"_bias.npy"
        if os.path.exists(bias):
            net.params[param_name][1].data[:]=np.load(bias)       
    elif '-scale' in param_name:
        ids=param_name[:-6].split('layer')[-1]
        weight="../voc_weights/meta/conv"+ids+"_bn_gamma.npy"
        bias="../voc_weights/meta/conv"+ids+"_bn_beta.npy"
        net.params[param_name][0].data[:]=np.load(weight)
        net.params[param_name][1].data[:]=np.load(bias)
        
net.save('meta_parameter.caffemodel')


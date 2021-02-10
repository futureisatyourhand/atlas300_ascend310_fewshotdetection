# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/12/15$ 12:12$
# @Author  : Qian Li
# @Email   : 1844857573@qq.com
# @File    : yolov2$.py
# Description :yolo2——darknet19.
# --------------------------------------

import os
import tensorflow as tf
import numpy as np
from config import class_names
import pickle
#################basis layers：conv/pool/reorg(with passthrough) #############################################
# activation function
def leaky_relu(x):
    return tf.nn.leaky_relu(x,alpha=0.1,name='leaky_relu') # 或者tf.maximum(0.1*x,x)
def dyconv(x,nums=1024):
    b,h,w=x.shape[:3]
    x=tf.tile(x,(len(class_names),1,1,1))
    with open("voc_novel2_.pkl",'rb') as f:
        weight=np.array(pickle.load(f,encoding='bytes'))[0]
        f.close()
    #x=tf.reshape(x,[-1,20*1024,w,h])
    #weight=tf.reshape()
    
    rtn1=tf.reshape(tf.transpose(x,perm=[0,3,1,2]),[-1,13,13])*tf.reshape(weight,[-1,1,1])
    
    x=tf.transpose(x,perm=[0,3,1,2])*tf.constant(weight,dtype=tf.float32)
    rtn2=tf.reshape(x,[-1,13,13])
    #print("b:",sess.run(tf.reshape(x,[-1,13,13])))
    return tf.transpose(x,perm=[0,2,3,1]),rtn1,rtn2

# Conv+BN
def conv2d(x,filters_shape,pad_size=0,strides=[1,1,1,1],batch_normalize=True,trainable=False,
		   activation=leaky_relu,use_bias=False,name='conv2d'):
    #weights from torch:(out_channels, in_channels,kernel_size)
    #weight=np.transpose(np.load("voc_weights/conv"+name.split('conv')[-1]+"_weight.npy"),[2,3,1,0]).astype(np.float32)
    with tf.variable_scope(name):
        # padding
        if pad_size > 0:
            x = tf.pad(x,[[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]])
        
        weight=np.transpose(np.load("voc_weights/conv"+name.split('conv')[-1]+"_weight.npy"),[2,3,1,0]).astype(np.float32)
        weights=tf.get_variable(name="weight",dtype=tf.float32, trainable=False,initializer=tf.constant(weight))
        import time
        start=time.time()
        out = tf.nn.conv2d(x,weights,strides=strides,padding='VALID',data_format='NHWC',name=name)
        return out
        print("conv time:",(time.time()-start)*1000)
        # BN
        if batch_normalize:
            bn_gamma=tf.constant(np.load("voc_weights/conv"+name.split('conv')[-1]+"_bn_gamma.npy"),dtype=tf.float32)
            bn_beta=tf.constant(np.load("voc_weights/conv"+name.split('conv')[-1]+"_bn_beta.npy"),dtype=tf.float32)
            start=time.time()
            out=out*bn_gamma+bn_beta
            print("bn time:",(time.time()-start)*1000)
        else:
            if use_bias:
                bias=tf.constant(np.load("voc_weights/conv"+name.split('conv')[-1]+"_bias.npy"),dtype=tf.float32)
                start=time.time()
                out=tf.nn.bias_add(out,bias,name=name+"_bias")
                print("bias time:",(time.time()-start)*1000)
        if activation:
            out = activation(out)
    return out

# max_pool
def maxpool(x,size=2,stride=2,name='maxpool'):
    return tf.nn.max_pool(x,ksize=[1,size,size,1],strides=[1,stride,stride,1],padding='VALID')
def glomax(inputs,size=6,name='glomax'):
    return tf.nn.avg_pool(conv10,ksize=[1,size,size,1],strides=[1,1,1,1],padding='VALID')
    
# reorg layer(with passthrough
def reorg(x,stride):
    return tf.space_to_depth(x,block_size=stride)
    # 或者return tf.extract_image_patches(x,ksizes=[1,stride,stride,1],strides=[1,stride,stride,1],
    # 								rates=[1,1,1,1],padding='VALID')
#########################################################################################################

################################### Darknet19 ###########################################################
# anchor_num*(class_num+5)=5*(80+5)=425
def darknet(images,n_last_channels=30):
    #tf.reset_default_graph()
    import time
    start1=time.time()
    net = conv2d(images, (3,3,3,32), pad_size=1, name='conv1')
    if True:
        return net
    net = maxpool(net, size=2, stride=2, name='pool1')

    net = conv2d(net,(3,3,32,64), pad_size=1, name='conv2')
    net = maxpool(net, 2, 2, name='pool2')
    net = conv2d(net, (3,3,64,128), pad_size=1, name='conv3')
    net = conv2d(net, (1,1,128,64), pad_size=0, name='conv4')
    net = conv2d(net, (3,3,64,128), pad_size=1, name='conv5')
    net = maxpool(net, 2, 2, name='pool3')
    
    net = conv2d(net, (3,3,128,256),pad_size=1, name='conv6')
    net = conv2d(net, (1,1,256,128),pad_size=0, name='conv7')
    net = conv2d(net, (3,3,128,256), pad_size=1, name='conv8')
    net = maxpool(net, 2, 2, name='pool4')
    net = conv2d(net, (3,3,256,512), pad_size=1, name='conv9')
    net = conv2d(net,(1,1,512,256), pad_size=0,name='conv10')
    net = conv2d(net,(3,3,256,512),pad_size=1, name='conv11')
    net = conv2d(net, (1,1,512,256),pad_size=0, name='conv12')
    net = conv2d(net, (3,3,256,512),pad_size=1, name='conv13')
    shortcut = net #
    net = maxpool(net, 2, 2, name='pool5')

    net = conv2d(net, (3,3,512,1024), pad_size=1, name='conv14')
    net = conv2d(net, (1,1,1024,512),pad_size=0, name='conv15')
    net = conv2d(net,(3,3,512,1024), pad_size=1, name='conv16')
    net = conv2d(net, (1,1,1024,512),pad_size=0, name='conv17')
    net = conv2d(net, (3,3,512,1024),pad_size=1, name='conv18')

    net = conv2d(net, (3,3,1024,1024),pad_size=1, name='conv19')
    net = conv2d(net, (3,3,1024,1024), pad_size=1, name='conv20')
    # shortcut
    # 26*26*512 -> 26*26*64 -> 13*13*256
    shortcut = conv2d(shortcut, (1,1,512,64), pad_size=0, name='conv21')
    shortcut = reorg(shortcut, 2)
    net = tf.concat([shortcut, net], axis=-1) # channel
    net = conv2d(net, (3,3,1280,1024), pad_size=1, name='conv22')#####
    ##dynamic conv
    with tf.variable_scope('dconv'):
        net,rtn1,rtn2 = dyconv(net)
    # detection layer
    
    output = conv2d(net,(1,1,1024,n_last_channels),pad_size=0, batch_normalize=False,activation=None, use_bias=True, name='conv23')
    return output
    
   
def get_meta(metas):
    net = conv2d(images, filters_num=32, filters_size=3, pad_size=1, name='conv1')
    net = maxpool(net, size=2, stride=2, name='pool1')

    net = conv2d(net, 64, 3, 1, name='conv2')
    net = maxpool(net, 2, 2, name='pool2')

    net = conv2d(net, 128, 3, 1, name='conv3_1')
    net = maxpool(net, 2, 2, name='pool2')

    net = conv2d(net, 256, 1, 0, name='conv3_2')
    net = maxpool(net, 2, 2, name='pool2')

    net = conv2d(net, 512, 3, 1, name='conv3_3')
    net = maxpool(net, 2, 2, name='pool3')

    net = conv2d(net, 1024, 3, 1, name='conv3_3')
    output = glomax(net, net.get_shape().as_list()[1], name='glomax')
    return output

#########################################################################################################
"""
if __name__ == '__main__':
	x = tf.random_normal([1, 416, 416, 3])
	model_output = darknet(x,)
       
	saver = tf.train.Saver()
	with tf.Session() as sess:
		
		saver.restore(sess, "./yolo2_model/yolo2_coco.ckpt")
		print(sess.run(model_output).shape) # (1,13,13,425)
"""

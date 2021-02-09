#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : convert_weight.py
#   Author      : YunYang1994
#   Created date: 2019-02-28 13:51:31
#   Description :
#
#================================================================

import argparse
import tensorflow as tf
from model import Darknet
parser = argparse.ArgumentParser()
parser.add_argument("--train_from_coco", action='store_true')
flag = parser.parse_args()

org_weights_path = "./yolov2_model/yolov2_coco.ckpt"
cur_weights_path = "./yolov2_model/yolov2.ckpt"


org_weights_mess = []
tf.Graph().as_default()
load = tf.train.import_meta_graph(org_weights_path + '.meta')
with tf.Session() as sess:
    load.restore(sess, org_weights_path)
    for var in tf.global_variables():
        var_name = var.op.name
        var_name_mess = str(var_name).split('/')
        var_shape = var.shape
        print("==name==",var_name_mess)
        if (var_name_mess[-1] not in ['weights', 'gamma', 'beta', 'moving_mean', 'moving_variance']) or \
                                       ('pred' in var_name_mess or 'decode' in var_name_mess): 
            continue
        #org_weights_mess.append([var_name, var_shape])
        #print(var_name,var_shape)
        #print("=> " + str(var_name).ljust(50), var_shape)
print()
tf.reset_default_graph()

cur_weights_mess = []
tf.Graph().as_default()
with tf.name_scope('input'):
    input_data = tf.placeholder(dtype=tf.float32, shape=(1, 416, 416, 3), name='input_data')
    #meta_weight=tf.placeholder(dtype=tf.float32,shape=(1,1,20*1024,1),name='meta_weight')
    #training = tf.placeholder(dtype=tf.bool, name='trainable')
model = Darknet(input_data, False)
for var in tf.global_variables():
    var_name = var.op.name
    var_name_mess = str(var_name).split('/')
    var_shape = var.shape
    #print(var_name_mess[0])
    #if flag.train_from_coco:
    #if 'dyconv' in var_name_mess and 'weights' in var_name_mess:
    #    cur_weights_mess.append([var_name,])
    if (var_name_mess[-1] not in ['weights', 'gamma', 'beta', 'moving_mean', 'moving_variance']) or 'pred' in var_name_mess or 'decode' in var_name_mess:
        print("===========",var_shape)
        continue
    #cur_weights_mess.append([var_name, var_shape])
    #print("=> " + str(var_name).ljust(50), var_shape)

org_weights_num = len(org_weights_mess)
cur_weights_num = len(cur_weights_mess)
if cur_weights_num != org_weights_num:
    raise RuntimeError

print('=> Number of weights that will rename:\t%d' % cur_weights_num)
cur_to_org_dict = {}
for index in range(org_weights_num):
    org_name, org_shape = org_weights_mess[index]
    cur_name, cur_shape = cur_weights_mess[index]
    if cur_shape != org_shape:
        #print(org_weights_mess[index])
        #print(cur_weights_mess[index])
        raise RuntimeError
    cur_to_org_dict[cur_name] = org_name
    #print("=> " + str(cur_name).ljust(50) + ' : ' + org_name)

with tf.name_scope('load_save'):
    name_to_var_dict={}
    #for var in tf.global_variables():
        #if 'dyconv' in var.op.name and 'weights' in var.op.name:
        #    name_to_var_dict[var.op.name]=
        #else:
        #    name_to_var_dict[var.op.name]=var
    name_to_var_dict = {var.op.name: var for var in tf.global_variables()}
    restore_dict = {cur_to_org_dict[cur_name]: name_to_var_dict[cur_name] for cur_name in cur_to_org_dict}
    load = tf.train.Saver(restore_dict)
    save = tf.train.Saver(tf.global_variables())
    #for var in tf.global_variables():
    #    print(var.op.name,var.shpe)
    #for var in tf.global_variables():
    #    print("=> " + var.op.name)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('=> Restoring weights from:\t %s' % org_weights_path)
    load.restore(sess, org_weights_path)
    save.save(sess, cur_weights_path)
tf.reset_default_graph()



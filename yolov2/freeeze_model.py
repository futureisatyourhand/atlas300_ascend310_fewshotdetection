# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/12/15$ 12:12$
# @Author  : Qian Li
# @Email   : 1844857573@qq.com
# @File    : freeeze_model.py
# --------------------------------------
import tensorflow as tf
from model import Darknet
from config import class_names
pb_file = "./yolov2.pb"
ckpt_file = "./yolov2_model/yolov2_coco.ckpt"

#output_node_names = ["input/input_data","input/meta_weight", "decode/obj_probs/Sigmoid","decode/class_probs/Reshape","decode/stack"]
#output_node_names = ["input/input_data","obj_probs/Sigmoid","class_probs/Reshape","pred/stack"]
output_node_names=["conv23/conv23_bias"]
#with tf.name_scope('input'):
#    input_data = tf.placeholder(dtype=tf.float32,shape=(None,416,416,3), name='input_data')
    #meta_weight=tf.placeholder(dtype=tf.float32,shape=(1,1,len(class_names)*1024,1), name='meta_weight')

#model = Darknet(input_data, trainable=False)
#print(model.conv_sbbox, model.conv_mbbox, model.conv_lbbox)
saver=tf.train.import_meta_graph(ckpt_file+".meta",clear_devices=True)
graph=tf.get_default_graph()
input_graph_def=graph.as_graph_def()

sess  = tf.Session()#(config=tf.ConfigProto(allow_soft_placement=True))
#saver = tf.train.Saver()
saver.restore(sess,ckpt_file)#tf.train.latest_checkpoint(ckpt_file))
#tensor_name_list = [(tensor.name,tensor.attr) for tensor in tf.get_default_graph().as_graph_def().node]
#print(tensor_name_list)
converted_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                            input_graph_def  = input_graph_def,#sess.graph.as_graph_def(),
                            output_node_names = output_node_names)
#for var in tf.global_variables():
#        print(var.op.name,var.shape)
with tf.gfile.GFile(pb_file, "wb") as f:
    f.write(converted_graph_def.SerializeToString())

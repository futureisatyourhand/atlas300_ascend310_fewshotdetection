# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/12/17$ 12:12$
# @Author  : Qian Li
# @Email   : 1844857573@qq.com
# @File    : pb.py
# Description :convert ckpt into pb
# --------------------------------------
import tensorflow as tf
from tensorflow.python.platform import gfile

tf.reset_default_graph() 
output_graph_path = 'yolov2.pb'
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    output_graph_def = tf.GraphDef()
   
    graph = tf.get_default_graph()
    with gfile.FastGFile(output_graph_path, 'rb') as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")
        
        print("%d ops in the final graph." % len(output_graph_def.node))

        tensor_name = [tensor.attr for tensor in output_graph_def.node]
        print(tensor_name)
        print('---------------------------')
        
        # summaryWriter = tf.summary.FileWriter('log_graph/', graph)

        #for op in graph.get_operations():
     
        #    print(op.name)

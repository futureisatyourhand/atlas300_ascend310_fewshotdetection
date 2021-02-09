"""import tensorflow as tf
with tf.Session() as sess:
    with open('./yolov2.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read()) 
        print(graph_def)
"""
# coding:utf-8
import tensorflow as tf
from tensorflow.python.platform import gfile

tf.reset_default_graph()  # 重置计算图
output_graph_path = 'yolov2.pb'
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    output_graph_def = tf.GraphDef()
    # 获得默认的图
    graph = tf.get_default_graph()
    with gfile.FastGFile(output_graph_path, 'rb') as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")
        # 得到当前图有几个操作节点
        print("%d ops in the final graph." % len(output_graph_def.node))

        tensor_name = [tensor.attr for tensor in output_graph_def.node]
        print(tensor_name)
        print('---------------------------')
        # 在log_graph文件夹下生产日志文件，可以在tensorboard中可视化模型
        # summaryWriter = tf.summary.FileWriter('log_graph/', graph)

        #for op in graph.get_operations():
        #    # print出tensor的name和值
        #    print(op.name)

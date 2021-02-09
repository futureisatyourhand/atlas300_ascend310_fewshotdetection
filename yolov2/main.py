# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2018/5/16$ 17:17$
# @Author  : KOD Chen
# @Email   : 821237536@qq.com
# @File    : Main$.py
# Description :YOLO_v2主函数.
# --------------------------------------
import sys

#reload(sys)


import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

#from decode import decode
from utils import preprocess_image#, postprocess, draw_detection
from config import class_names
from model import Darknet

def main():
    input_size = (416,416)
    image_file = './005767.jpg'
    image = cv2.imread(image_file)
    image_shape = image.shape[:2] #只取wh，channel=3不取
    tf.reset_default_graph()
    # copy、resize416*416、归一化、在第0维增加存放batchsize维度
    image_cp = preprocess_image(image,input_size)
    #image_cp=np.ones([1,416,416,3]).astype(np.float32)
    image_cp=np.load("005767.npy")#("/home/huawei/chems/bioavailability_model/atlas_data/005767.npy")
    image_cp=np.transpose(image_cp,[0,2,3,1])
    np.save("atoms.npy",image_cp)
    # 【1】输入图片进入darknet19网络得到特征图，并进行解码得到：xmin xmax表示的边界框、置信度、类别概率
    with tf.name_scope('input'):
        tf_image = tf.placeholder(tf.float32,[1,input_size[0],input_size[1],3],name='input_data')
        #meta_variable=tf.placeholder(tf.float32,[1,1,len(class_names)*1024,1],name='meta_weigiht')
    model_output = Darknet(tf_image) # darknet19网络输出的特征图
    #meta=np.ones([1,1,len(class_names)*1024,1], dtype=np.float32)
    model_path = "./yolov2_model/yolov2_coco.ckpt"
    saver = tf.train.Saver()
    with tf.Session() as sess:
        #sess.run(model_output,feed_dict={tf_image:image_cp,meta_variable:meta})
        sess.run(tf.global_variables_initializer())
        a=sess.run(model_output.rtn(),feed_dict={tf_image:image_cp})
        a=np.transpose(a,[0,3,1,2])
        #a=np.transpose(a,[0,3,2,1])
        a=np.reshape(a,[-1])[:90]
        #print(a)
        for i in range(90):
        #    print("=============================")
            print(a[i],i)
        saver.save(sess,model_path)
    ###############################################
    #exit()
    # 【2】筛选解码后的回归边界框——NMS(post process后期处理)
    #bboxes,scores,class_max_index = postprocess(bboxes,obj_probs,class_probs,image_shape=image_shape)

    # 【3】绘制筛选后的边界框
    #img_detection = draw_detection(image, bboxes, scores, class_max_index, class_names)
    #cv2.imwrite("./yolo2_data/detection.jpg", img_detection)
    #print('YOLO_v2 detection has done!')
    #cv2.imshow("detection_results", img_detection)
    #cv2.waitKey(0)

if __name__ == '__main__':
    main()

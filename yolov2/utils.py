# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2018/5/16$ 14:48$
# @Author  : KOD Chen
# @Email   : 821237536@qq.com
# @File    : utils$.py
# Description :功能函数，包含：预处理输入图片、筛选边界框NMS、绘制筛选后的边界框。
# --------------------------------------

import random
import colorsys
import cv2
import numpy as np
import tensorflow as tf 

# 【1】图像预处理(pre process前期处理)
def preprocess_image(image,image_size=(416,416)):
	# 复制原图像
	image_cp = np.copy(image).astype(np.float32)

	# resize image
	image_rgb = cv2.cvtColor(image_cp,cv2.COLOR_BGR2RGB)
	image_resized = cv2.resize(image_rgb,image_size)

	# normalize归一化
	image_normalized = image_resized.astype(np.float32)# / 225.0

	# 增加一个维度在第0维——batch_size
	image_expanded = np.expand_dims(image_normalized,axis=0)

	return image_expanded



# 【3】绘制筛选后的边界框
def draw_detection(im, bboxes, scores, cls_inds, labels, thr=0.3):
	# Generate colors for drawing bounding boxes.
	hsv_tuples = [(x/float(len(labels)), 1., 1.)  for x in range(len(labels))]
	colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
	colors = list(
		map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),colors))
	random.seed(10101)  # Fixed seed for consistent colors across runs.
	random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
	random.seed(None)  # Reset seed to default.
	# draw image
	imgcv = np.copy(im)
	h, w, _ = imgcv.shape
	for i, box in enumerate(bboxes):
		if scores[i] < thr:
			continue
		cls_indx = cls_inds[i]

		thick = int((h + w) / 300)
		cv2.rectangle(imgcv,(box[0], box[1]), (box[2], box[3]),colors[cls_indx], thick)
		mess = '%s: %.3f' % (labels[cls_indx], scores[i])
		if box[1] < 20:
			text_loc = (box[0] + 2, box[1] + 15)
		else:
			text_loc = (box[0], box[1] - 10)
		# cv2.rectangle(imgcv, (box[0], box[1]-20), ((box[0]+box[2])//3+120, box[1]-8), (125, 125, 125), -1)  # puttext函数的背景
		cv2.putText(imgcv, mess, text_loc, cv2.FONT_HERSHEY_SIMPLEX, 1e-3*h, (255,255,255), thick//3)
	return imgcv







###################################################################################################

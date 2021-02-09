import numpy as np
import ctypes
from utils import *
import ctypes
import sys
import os
import sys
#mylib = ctypes.cdll.LoadLibrary("../inferences/InferLib/out/infer.so")
#data = np.ones((1,416,416,3), dtype=np.float32)
#data = np.repeat(data, batch_size, axis=0).reshape(-1)

#dataptr = data.ctypes
#dataptr = (ctypes.c_float*data.shape[0])(*data)
#result = np.ones((batch_size), np.float32)
#resultptr = (ctypes.c_float*result.shape[0])(*result)
#end = datetime.datetime.now()

#package_data = np.ones((1,416,416,3), dtype=np.float32)
n_cls=20
nms_thresh=0.45
classes=['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
fps=[0]*n_cls
anchors=[1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071]#[0.57273, 0.677385,1.87446, 2.06253,3.33843, 5.47434,7.88282, 3.52778,9.77052, 9.16828]
    
mylib = ctypes.cdll.LoadLibrary("./infer.so")
for i, cls_name in enumerate(classes):
    buf = 'results/comp4_det_test_%s.txt' % (cls_name)
    fps[i] = open(buf, 'a+')

#for ids in os.listdir("convert_data/"):

ids=sys.argv[1]
ids=ids.strip("\n").strip("")+".npy"
package_data=np.load("convert_data/"+ids)#("atlas_data/"+ids)
bs=1
#if os.path.exists("output.txt"):
#    os.remove("output.txt")
#package_data=np.transpose(package_data,[0,3,1,2])
#package_data=np.tile(package_data,(100,1,1,1))

b,rows,cols,channels = package_data.shape
data = package_data.reshape(-1)
data = np.repeat(package_data.reshape(-1), 1)
data = (ctypes.c_float*data.shape[0])(*data)

size=bs*20*30*169#(bs*20)30 169
detect = np.ones((size), np.float32)
detectptr = (ctypes.c_float*detect.shape[0])(*detect)    

cnt_time=1
result = np.ones((cnt_time), np.float32)
resultptr = (ctypes.c_float*result.shape[0])(*result)

mylib._Z5inferPfiiiiiiiS_i(data, 1, channels, rows, cols,3*416*416, 4, 5, resultptr,0)
#del mylib

f=open("output.txt",'r')
#print("====",len(f.readlines()))
content=f.readlines()[0]
#content=content[0]
content=content.strip(" ")
content=content.split(" ")
f.close()
#os.remove("output.txt")
#os.mknod("output.txt")

output=[]
ctn=30*169
for c in content:
    output.append(float(c))
output=np.array(output)


##########################
bias=np.load("../yolov2/voc_weights/5shot/detect/conv23_bias.npy")
bias=bias[np.newaxis,:,np.newaxis,np.newaxis]
output=np.reshape(output,[20,30,13,13])+np.tile(bias,(20,1,13,13))

output=torch.Tensor(output)
batch_boxes = get_region_boxes_v2(output, 20,0.005,1,anchors, 5, 0, 1)
print(len(batch_boxes)) 
for b in range(bs):
    width, height = get_image_size("/home/huawei/JPEGImages/"+ids.split('.')[0]+".jpg")
    for i in range(n_cls):
        # oi = i * bs + b
        oi = b * n_cls + i
        boxes = batch_boxes[oi]
        boxes = nms(boxes, nms_thresh)
        for box in boxes:
            x1 = (box[0] - box[2]/2.0)*width
            y1 = (box[1] - box[3]/2.0)*height
            x2 = (box[0] + box[2]/2.0)*width
            y2 = (box[1] + box[3]/2.0)*height

            det_conf = box[4]
            for j in range(int((len(box)-5)/2)):
                cls_conf = box[5+2*j]
                cls_id = box[6+2*j]
                prob =det_conf * cls_conf
                fps[i].write('%s %f %f %f %f %f %f\n' % (ids.split('.')[0],cls_conf, prob, x1, y1, x2, y2))
for i in range(n_cls):
    fps[i].close()
#os.kill()#system("pause")

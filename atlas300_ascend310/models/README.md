## Object Detection Model of SSD or YOLOv3

### Train Model Project:

**SSD:**
https://github.com/weiliu89/caffe/tree/ssd

**YOLOv3:**
https://github.com/wizyoung/YOLOv3_TensorFlow/releases/

### Original Network Link and Pre-trained Model Link:

**SSD:**
https://drive.google.com/file/d/0BzKzrI_SkD1_NDlVeFJDc2tIU1k/view?usp=drivesdk

**YOLOv3:**
https://github.com/wizyoung/YOLOv3_TensorFlow

### Convert checkpoint file to pb file

**Instructions:**
https://bbs.huaweicloud.com/forum/thread-45383-1-1.html

### Dependency

**SSD:**
Change the type of layer "detection_out", from "DetectionOutput" to "SSDDetectionOutput", in model file -- deploy.prototxt

### Convert caffe/tensorflow file To Ascend om file

**SSD:**
```bash
omg --framework 0 --model ./deploy.prototxt --weight ./VGG_coco_SSD_300x300.caffemodel --output vgg_ssd_300x300 --insert_op_conf aipp_vgg_picture.cfg
```

**YOLOv3:**
```bash
omg --model ./Epoch_32.pb --framework 3 --output ./yolov3 --insert_op_conf ./aipp_yolov3_picture.cfg --input_shape "input_data:1,416,416,3"
```

### Versions that have been verified: 

- Atlas 300
- Atlas 500

#### Node
- The file of aipp_ssd_picture.cfg is the configuration parameter for converting the picture input_format of YUV420SP_U8 to RGB888_U8, another is for vedio. The crop parameter can be determined by the size of the model input.

- The file of aipp_yolov3_picture.cfg is the configuration parameter for converting the picture input_format of YUV420SP_U8 to BGR888_U8, another is for vedio. The crop parameter can be determined by the size of the model input.

- For more parameter configuration, please refer to the Atlas 300 AI Accelerator Card Model Conversion Guide (Model 3010).




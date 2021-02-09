from yolov2 import darknet
import numpy as np
import tensorflow as tf
from config import anchors,class_names,input_size,x_offset,y_offset
class Darknet(object):
    def __init__(self,input_data,trainable=False):
        self.anchors=anchors
        self.class_names=class_names
        self.output_sizes = input_size[0]//32, input_size[1]//32
        self.ncls=len(class_names)
        self.image_shape=input_size
        self.x_offset=np.reshape(x_offset,[1,-1,1])
        self.y_offset=np.reshape(y_offset,[1,-1,1])
        import time
        start=time.time()
        self.output=darknet(input_data)
        print((time.time()-start)*1000)
        #with tf.variable_scope('output'):
        #    self.result=tf.reshape(self.output,[-1,20*169,5,6])
        #with tf.variable_scope('output'):
        #self.result1,self.result2,self.result3=self.decode(self.output)#tf.reshape(output,[20,169,5,6])

    def rtn(self):
        #obj_probs:tf.reshape(tf.transpose(obj_probs,perm=[2,0,1]),[1,-1])#obj_probs
        #return tf.reshape(tf.transpose(self.result,perm=[1,2,0]),[1,-1]) #class_probs
        #return tf.reshape(tf.transpose(self.result,perm=[0,3,1,2]),[1,-1])#boxes
        #return tf.reshape(tf.transpose(self.result1,perm=[2,0,1]),[1,-1])#boxes
        return self.output
    def decode(self,model_output):
        '''
        model_output:darknet19网络输出的特征图
        output_sizes:darknet19网络输出的特征图大小，默认是13*13(默认输入416*416，下采样32)
        '''
        H, W = self.output_sizes
        num_anchors = len(self.anchors) # 这里的anchor是在configs文件中设置的,这里为5，和pytorch的不同
        anchors = tf.constant(self.anchors, dtype=tf.float32)  # 将传入的anchors转变成tf格式的常量列表
        bs=model_output.shape[0]//self.ncls
        nA=num_anchors
        nC=1
        cs=self.ncls
        batch=model_output.shape[0]
        # 13*13*num_anchors*(num_class+5)，第一个维度自适应batchsize
        x_offset=tf.constant(self.x_offset,dtype=tf.float32)
        y_offset=tf.constant(self.y_offset,dtype=tf.float32)
        model_output =tf.reshape(model_output,[batch,H*W,num_anchors,5+nC])
        with tf.variable_scope('xy_offset'): 
            # darknet19网络输出转化——偏移量、置信度、类别概率
            xy_offset = tf.nn.sigmoid(model_output[:,:,:,0:2]) # 中心坐标相对于该cell左上角的偏移量，sigmoid函数归一化到0-1
            bbox_x=xy_offset[:,:,:,0]#bbox_x = (tf.tile(x_offset,[batch,1,num_anchors]) + xy_offset[:,:,:,0])#()/W
            bbox_y=xy_offset[:,:,:,1]#bbox_y = (tf.tile(y_offset,[batch,1,num_anchors]) + xy_offset[:,:,:,1])#()/H
            
        with tf.variable_scope('wh_offset'):
            wh_offset = tf.exp(model_output[:,:,:,2:4]) #相对于anchor的wh比例，通过e指数解码
            bbox_w=wh_offset[:,:,:,0]#bbox_w = (anchors[:,0] * wh_offset[:,:,:,0])/W#()/W
            bbox_h=wh_offset[:,:,:,1]#bbox_h = (anchors[:,1] * wh_offset[:,:,:,1])/H#()/H
        with tf.variable_scope('obj_probs'):
            obj_probs =tf.sigmoid(model_output[:,:,:,4]) # 置信度，sigmoid函数归一化到0-1
        #class_probs =tf.softmax( tf.reshape(model_output[:,:,:,5],[bs,cs,h*w,num_anchors]),1)
        print(model_output[:,:,:,5].shape)
        class_probs = tf.reshape(model_output[:,:,:,5],[bs,cs,H*W,num_anchors]) # 网络回归的是'得分',用softmax转变成类别概率
        print(bs,H*W,num_anchors,cs)
        class_probs=tf.nn.softmax(tf.reshape(tf.transpose(class_probs,perm=[0,2,3,1],conjugate=True),[-1,cs]),axis=-1)
        class_probs=tf.reshape(class_probs,[bs,H*W,num_anchors,cs])   
        class_probs=tf.transpose(class_probs,perm=[0,3,1,2])
       
        with tf.variable_scope('class_probs'):
            class_probs=tf.reshape(class_probs,[bs*cs,H*W,num_anchors]) #[1,2,0]
            #class_probs=tf.reshape(class_probs,[-1,cs,H*W,num_anchors])
        with tf.variable_scope('pred'):
            #bboxes=tf.concat([obj_probs,class_probs],axis=-1)
            #bbox_w=tf.reshape(bbox_w,[-1,cs,H*W,num_anchors],name="bbox_w")
            #bbox_h=tf.reshape(bbox_h,[-1,cs,H*W,num_anchors],name="bbox_h")
            #bbox_x=tf.reshape(bbox_x,[-1,cs,H*W,num_anchors],name="bbox_x")
            #bbox_y=tf.reshape(bbox_y,[-1,cs,H*W,num_anchors],name="bbox_y")
            #bboxes = tf.stack([(bbox_x/100.-bbox_w/200.),(bbox_y/100.-bbox_h/200.),(bbox_x/100.+bbox_w/200.),(bbox_y/100.+bbox_h/200.)], axis=3)
            bboxes = tf.stack([bbox_w,bbox_h,bbox_x,bbox_y], axis=3)
        print(obj_probs.shape,class_probs.shape,bboxes.shape)
        #print(bboxes)
        return obj_probs,class_probs,bboxes


        ##############tensorflow true##############        
        """
        class_probs=tf.reshape(tf.transpose(class_probs,perm=[0,2,3,1]),[-1,cs])
        class_probs=tf.nn.softmax(class_probs,axis=-1)
        class_probs=tf.reshape(class_probs,[bs,H*W,num_anchors,cs])
        with tf.variable_scope('class_probs'):
            class_probs=tf.reshape(tf.transpose(class_probs,perm=[0,3,1,2]),[bs*cs,H*W,num_anchors])
        # 中心坐标+宽高box(x,y,w,h) -> xmin=x-w/2 -> 左上+右下box(xmin,ymin,xmax,ymax)
        with tf.variable_scope('pred'):
            
            bboxes = tf.stack([bbox_x,bbox_y,bbox_w,bbox_h,obj_probs,class_probs], axis=3)
            #bboxes = tf.stack([bbox_x-bbox_w/2, bbox_y-bbox_h/2,bbox_x+bbox_w/2, bbox_y+bbox_h/2,obj_probs,class_probs], axis=3)

            bboxes=tf.transpose(bboxes,perm=[0,2,3,1])
        return bboxes
        """

        # 构建特征图每个cell的左上角的xy坐标
        # 变成x_cell=[[0,1,...,12],...,[0,1,...,12]]和y_cell=[[0,0,...,0],[1,...,1]...,[12,...,12]]
        #y_cell=tf.tile(tf.expand_dims(tf.range(H, dtype=tf.float32),axis=-1,name="y_cell_dim1"), [1,H],name="y_cell_tile1")
        #x_cell=tf.tile(tf.expand_dims(tf.range(H, dtype=tf.float32),axis=0,name="x_cell_dim1"), [H, 1],name="x_cell_tile1")
        #print("----------------",x_cell.shape,y_cell.shape)
        #s=H*H
        #x_cell = tf.expand_dims(x_cell,axis=-1,name="x_cell_dim2")# 和上面[H*W,num_anchors,num_class+5]对应
        #y_cell = tf.expand_dims(tf.expand_dims(tf.reshape(y_cell,[H*H],name="y_cell_reshape1"),axis=0,name="y_cell_dim2"),axis=-1,name="y_cell_dim3")

        # decode
        #bbox_x = (tf.reshape(x_cell,[1,-1,1]) + xy_offset[:,:,:,0])/W
        #bbox_y = (tf.reshape(y_cell,[1,-1,1]) + xy_offset[:,:,:,1])/H
        #bbox_w = (anchors[:,0] * wh_offset[:,:,:,0])/W
        #bbox_h = (anchors[:,1] * wh_offset[:,:,:,1])/H
        # 中心坐标+宽高box(x,y,w,h) -> xmin=x-w/2 -> 左上+右下box(xmin,ymin,xmax,ymax)
        #bboxes = tf.stack([bbox_x-bbox_w/2, bbox_y-bbox_h/2,
        #        bbox_x+bbox_w/2, bbox_y+bbox_h/2], axis=3)

        #return bboxes, obj_probs, class_probs

# atlas300_ascend310_fewshotdetection

## Caffe
We add permute and Reorg for YOLOv2. 
```
su huawei  
cd caffe  
make clean  
make pycaffe  
make distribute  
```
Our code is based on https://github.com/BVLC/caffe.git.  
The version is PyTorch 0.3.1.post2,
```
class Reorg(nn.Module):  
    def __init__(self, stride=2):  
        super(Reorg, self).__init__()  
        self.stride = stride  
    def forward(self, x):   
        stride = self.stride  
        assert(x.data.dim() == 4)  
        B = x.data.size(0)  
        C = x.data.size(1)  
        H = x.data.size(2)  
        W = x.data.size(3)  
        assert(H % stride == 0)  
        assert(W % stride == 0)  
        ws = stride  
        hs = stride  
        x = x.view(B, C, H/hs, hs, W/ws, ws).transpose(3,4).contiguous()  
        x = x.view(B, C, H/hs*W/ws, hs*ws).transpose(2,3).contiguous()  
        x = x.view(B, C, hs*ws, H/hs, W/ws).transpose(1,2).contiguous()  
        x = x.view(B, hs*ws*C, H/hs, W/ws)  
        return x  
```
However reorg is wrong for caffe and atlas,we make right reorg layer by 
```
layer{
    type:"Reshape"
    name:"reorg1"
    bottom:"layer21-act"
    top:"reorg1"
    reshape_param{
        shape{dim:-1 dim:13 dim:2 dim:26}
    }
}
layer{
    type:"Reshape"
    name:"reorg2"
    bottom:"reorg1"
    top:"reorg2"
    reshape_param{
        shape{dim:-1 dim:2  dim:26}
    }
}
layer{
    type:"Reshape"
    name:"reorg3"
    bottom:"reorg2"
    top:"reorg3"
    reshape_param{
        shape{dim:-1 dim:2  dim:13 dim:2}
    }
}


layer{
    bottom:"reorg3"
    top:"reorg4"
    name:"reorg4"
    type:"Permute"
    permute_param {
       order: 0
        order: 2
        order: 1
        order: 3
    }
}
layer{
    type:"Reshape"
    name:"reorg5"
    bottom:"reorg4"
    top:"reorg5"
    reshape_param{
        shape{dim:-1 dim:13 dim:4}
    }
}
layer{
    type:"Reshape"
    name:"reorg6"
    bottom:"reorg5"
    top:"reorg6"
    reshape_param{
        shape{dim:-1 dim:13 dim:13 dim:4}
    }
}
layer{
    type:"Reshape"
    name:"reorg7"
    bottom:"reorg6"
    top:"reorg7"
    reshape_param{
        shape{dim:-1 dim:169 dim:4}
    }
}
layer{
    type:"Reshape"
    name:"reorg8"
    bottom:"reorg7"
    top:"reorg8"
    reshape_param{
        shape{dim:-1 dim:64 dim:169 dim:4}
    }
}
layer{
    bottom:"reorg8"
    top:"reorg9"
    name:"reorg9"
    type:"Permute"
    permute_param {
       order: 0
        order: 3
        order: 1
        order: 2
    }
}
layer{
    type:"Reshape"
    name:"reorg10"
    bottom:"reorg9"
    top:"reorg10"
    reshape_param{
        shape{dim:-1 dim:256 dim:169}
    }
}
layer{
    type:"Reshape"
    name:"reorg11"
    bottom:"reorg10"
    top:"reorg11"
    reshape_param{
        shape{dim:-1 dim:256 dim:13 dim:13}
    }
}
```
Get our lmdb for voc dataset  
```
build/tools/convert_imageset --resize_height=416 --resize_width=416 /home/huawei/JPEGImages/ /home/huawei/yolov2/mycaffe/train.txt /home/huawei/yolov2/mycaffe/lmdb
```

Train caffe  
```
cd yolov2/mycaffe/
./caffe/build/tools/caffe train -iterations 1  -solver ./solver.prototxt 2>&1| tee caffe_train.log
```

Assign weights from Torch.  
```
python get_params.py
```

Validate our model  
```
python inference.py
```
## Atlas300  
according to PDF from https://ascend.huawei.com/#/document?tag=developer, you can install drivers and envs to execute your program.     
create offline-model for Atlas300 Ascend310.  
 ```
cd /home/huawei/ddk/ddk/uihost/bin  
```
```
./omg --model=/home/huawei/yolov2/mycaffe/demo.prototxt --weight=/home/huawei/yolov2/voc_weights/3shot/3shot_parameter.caffemodel --mode=0 --output=/home/huawei/samples/Samples/my_right/data/models/3shot --framework=0  --input_format="NCHW" --output_type=FP32 --input_shape=data:1,3,416,416
```

Create program for atlas300  
```
cd atlas300_ascend310/InferLib
bash build.sh A300
cd ../
sh update.sh
```
Update graph0.config for your offline-model and batch-size to inference by Atlas300.   
```
 engines {
        id: 2
        engine_name: "MindInferenceEngine"
        so_name: "./libmind_inference_engine.so"
        thread_num: 2

        ai_config {
            items {
                name: "model_path"
                value: "data/models/5shot.om" #batch = 1
            }
            items {
                name: "batch_size"
                value: "1"
            }
        }
        side: DEVICE
    }
```

## Inference  
First, update test.sh for test-kshot.  
Second, update detect.py for k-shot(e.g., k=5).  
setting bias=np.load("./yolov2/voc_weights/5shot/detect/conv23_bias.npy").   
```
sh test.sh  
cd results  
scp comp4_det_test_* liqian@10.2.151.160:/home1/liqian/Fewshot_Detection/results/get_weights_test_plot/atlas  
```
## Evaluate results from Atlas300 by envs of PyTorch 0.3.1.post2.  
Finally, ssh liqian@10.2.151.160
Evaluate APs for few-shot detection on atlas300 Ascend310.  
```
cd /home1/liqian/Fewshot_Detection/  
source activate darknet  
python2 script/voc_eval results/get_weights_test_plot/atlas/
```
![image](https://github.com/futureisatyourhand/atlas300_ascend310_fewshotdetection/blob/master/%E5%9B%BE%E7%89%87/atlas.png)


Visual results from k-shot on Atlas300 Ascend310.    
```
python2 plot_box.py results/get_weights_test_plot/atlas/
```
## License and Citation

If you use these methods in you research about algorithm, please cite:
```
@misc{li2020toprelated,
      title={Top-Related Meta-Learning Method for Few-Shot Detection}, 
      author={Qian Li and Nan Guo and Xiaochun Ye and Duo Wang and Dongrui Fan and Zhimin Tang},
      year={2020},
      eprint={2007.06837},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

Please cite atlas combing python with c++ in your publications if it helps your research.

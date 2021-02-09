#include "meta2post.hpp"
#include "fastmath.hpp"
#include <string>
#include <vector>
#include<iostream>

using NetInfo = struct {
    int classNum;
    int anchor_nums;
    int netWidth;
    int netHeight;
//    std::vector<OutputLayer> outputLayers;
} ;


void InitNetInfo(NetInfo& netInfo, 
                 int netWidth, 
                 int netHeight, 
                 int classNum,int anchor_nums)
{
    netInfo.anchor_nums=anchor_nums;
    netInfo.classNum = classNum;//voc:20
    netInfo.netWidth = netWidth;//13
    netInfo.netHeight = netHeight;//13

}

float BoxIou(DetectBox a, DetectBox b) 
{
    float ixmin = std::min(a.xmin,b.xmin);
    float ixmax = std::max(a.xmax,b.xmax);
    float iymin = std::min(a.ymin,b.ymin);
    float iymax = std::max(a.ymax,b.ymax);
    float uw=ixmax-ixmin+1;
    float uh=iymax-iymin+1;
    float cw=a.xmax-a.xmin+b.xmax-b.xmin-uw+1;
    float ch=a.ymax-a.ymin+b.ymax-b.ymin-uh+1;
    if (cw<=0.0 || ch<=0.0)
        return 0.0;
    float area=ch*cw;
    
    //float int_h=std::max(iymax-iymin,0.0f);
    //float int_w=std::max(ixmax-iymin,0.0f);
    float area1=(a.xmax-a.xmin)*(a.ymax-a.ymin);
    float area2=(b.xmax-b.xmin)*(b.ymax-b.ymin);
    //float area = (ixmax-ixmin)*(iymax -iymin);
    return area/(area1+area2-area);
}

void NmsSort(std::vector<DetectBox>& detBoxes, int classNum, float nmsThresh)
{
    //if (nmsThresh <0.0f) return;

    std::vector<DetectBox> sortBoxes;
    std::vector<std::vector<DetectBox>> resClass;
    resClass.resize(classNum);

    for (const auto& item : detBoxes) {
        resClass[item.classID].push_back(item);
    }

    for (int i = 0; i < classNum; ++i) {
        auto& dets = resClass[i];
        if (dets.size() == 0) continue;

        std::sort(dets.begin(), dets.end(), [=](const DetectBox& a, const DetectBox& b) {
            return a.dets_conf > b.dets_conf;
        });
        

        //std::cout<<"dets:"<<dets.size()<<std::endl;
        for (unsigned int m = 0;m < dets.size() ; ++m) {
            auto& item = dets[m];
            sortBoxes.push_back(item);
            for (unsigned int n = m + 1;n < dets.size() ; ++n) {
                  //std::cout<<"output:"<<BoxIou(item,dets[n])<<std::endl;
                if (BoxIou(item, dets[n]) > 0.45) {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }

    detBoxes = std::move(sortBoxes);
}

//output:batch*20,13,13,30
//batch*height*width*channel
//get[b,h,w,c]=b*height*width*channel+h*width*channel+w*channel+c
void GenerateBbox(std::vector<float *>singleResult,std::vector<DetectBox> &detBoxes,NetInfo info,float conf_thresh=0.005){
    int num_anchors=5;
    int cls=20;
    const float *output=singleResult[0];
    const float *class_probs=singleResult[1];
    int bs_i=169*20,cls_c=169,x_c=13;
        for(int c=0;c<cls;++c){
        //int c_i=bs_i+c*cls_c;//c_i=c*cls_c;
        for(int x=0;x<13;++x){
            //int x_i=c_i+x*x_c;//x*x_c;
            for(int y=0;y<13;++y){
               //int y_i=x_i+y;//30*y    ====169*20*30+c*169*30+x*13*30+y*30+6*anchor
               for(int anchor=0;anchor<5;++anchor){
                //int location=y_i*30+6*anchor;//bs_i+c_i+x_i+y_i+6*anchor
                int x_id=0,y_id=0,w_id=0,h_id=0,obj_id=0;
                x_id=169*25*c+169*5*anchor+0*169+x*13+y;
                y_id=169*25*c+169*5*anchor+1*169+x*13+y;
                w_id=169*25*c+169*5*anchor+2*169+x*13+y;
                y_id=169*25*c+169*5*anchor+3*169+x*13+y;
                obj_id=169*25*c+169*5*anchor+4*169+x*13+y;
                int cls_ids=5*169*c+169*anchor+x*13+y;

                float obj_probs=fastmath::sigmoid(output[obj_id]);                 
                float cls_probs=class_probs[cls_ids];
                float dets_conf=obj_probs*cls_probs;
                if(dets_conf<=0.0f || dets_conf>=1.0 || dets_conf<0.15)
                    continue;

                float x_offset=fastmath::sigmoid(output[x_id]);
                float y_offset=fastmath::sigmoid(output[y_id]);  
                float w_offset=fastmath::exp(output[w_id]);
                float h_offset=fastmath::exp(output[h_id]);
                
                DetectBox det;
                det.classID=c;
                det.class_conf=cls_probs;
                det.dets_conf=obj_probs;
                w_offset=(anchors[int(2*anchor)]*w_offset)/13.0;
                h_offset=(anchors[int(2*anchor+1)]*h_offset)/13.0;
                x_offset=(x_offset+y)/13.0;
                y_offset=(y_offset+x)/13.0;
                det.xmin=x_offset-w_offset/2.0;
                det.ymin=y_offset-h_offset/2.0;
                det.xmax=x_offset+w_offset/2.0;
                det.ymax=y_offset+h_offset/2.0;
                if(det.xmax<=det.xmin || det.ymax<=det.ymin)
                    continue;
                detBoxes.emplace_back(det);
               }
            }
        }
    }
}


//netinfo:net_width,net_height,anchor_nums
//output:xmin,ymin,xmax,ymax,objs,clsss_conf;
void GenerateMetaBbox(std::vector<float *> singleResult,std::vector<DetectBox> &detBoxes,NetInfo info,float conf_thresh=0.005){
    int count=169;
    const float *obj_probs=singleResult[1]; //obj_probs
    const float *class_probs=singleResult[2];//class_probs
    const float *bboxes=singleResult[0];//bboxes [cls,4,169,5]
    float pred_w=0.f,pred_h=0.0f,pred_x=0.0f,pred_y=0.0f;
    //std::vector<DetectBox> detectBox;
    for(int anchor=0;anchor<5;++anchor){
        for(int cls=0;cls<20;++cls){
            for(int h=0;h<13;++h){
                 for(int w=0;w<13;++w){
                     int obj=20*count*anchor+count*cls+h*13+w;
                     int class_id=(h*13+w)*5*20 +20*anchor+cls;
                     if(obj_probs[obj]*class_probs[class_id]<0.0f || obj_probs[obj]*class_probs[class_id]>=1.0)
                         continue;
                     //int begin=4*count*5*cls+    (13*h+w)
                     int w_point=4*count*5*cls+(h*13+w)*5+anchor; //4*count*anchor+4*13*h+4*w;
                     int h_point=4*count*5*cls+5*169+(h*13+w)*5+anchor;
                     int x_point=4*count*5*cls+2*5*169+(h*13+w)*5+anchor;
                     int y_point=4*count*5*cls+3*5*169+(h*13+w)*5+anchor;
                     DetectBox det;
                     det.classID=cls;
                     det.class_conf=class_probs[class_id];
                     det.dets_conf=obj_probs[obj];
                     pred_w=bboxes[w_point];//*100.f;
                     pred_h=bboxes[h_point];//*100.f;
                     pred_x=bboxes[x_point];//*100.f;
                     pred_y=bboxes[y_point];//*100.f;
                     
                     pred_w=(anchors[int(2*anchor)]*pred_w)/13.0;
                     pred_h=(anchors[int(2*anchor+1)]*pred_h)/13.0;
                     pred_x=(pred_x+w)/13.0;
                     pred_y=(pred_y+h)/13.0;
                     det.xmin=pred_x-pred_w/2.0;
                     det.ymin=pred_y-pred_h/2.0;
                     det.xmax=pred_x+pred_w/2.0;
                     det.ymax=pred_y+pred_h/2.0;
                     detBoxes.emplace_back(det);
                 }
            }
        }
    }
}
//out meta result
void GetRealBox(std::vector<DetectBox>& detBoxes, int imgWidth, int imgHeight)
{
    for (auto& dBox : detBoxes) {
        dBox.xmin *= imgWidth;
        dBox.ymin *= imgHeight;
        dBox.xmax *= imgWidth;
        dBox.ymax *= imgHeight;
    }
}

//out few-shot detection output
std::vector<DetectBox> FewShotDetectionOutput(std::vector<DetectBox> &detBoxes,std::vector<float *> outputData, 
                                             int netWidth, 
                                             int netHeight, 
                                             int imgWidth,
                                             int imgHeight,
                                             int classNum, 
                                             int anchor_nums,
                                             float conf_thresh, 
                                             float nmsThresh,
                                             bool usePad) 
{
    static NetInfo netInfo;
    //printf("###########FewShotDetectionOutput########");
    //if (netInfo.outputLayers.empty()) {
    InitNetInfo(netInfo, netWidth, netHeight, classNum,anchor_nums);
    //}

    //std::vector <DetectBox> detBoxes;
    //GenerateMetaBbox(outputData,detBoxes, netInfo, conf_thresh);
    GenerateBbox(outputData,detBoxes,netInfo,conf_thresh);
    //CorrectBbox(detBoxes, netWidth, netHeight, imgWidth, imgHeight, usePad);
    NmsSort(detBoxes, classNum, nmsThresh);

    return detBoxes;
}

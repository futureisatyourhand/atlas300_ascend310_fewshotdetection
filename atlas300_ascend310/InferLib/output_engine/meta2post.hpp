#ifndef __META2POST_H__
#define __META2POST_H__
#include <algorithm>
#include <vector>
static float anchors[10]={0.57273, 0.677385,1.87446, 2.06253,3.33843, 5.47434,7.88282, 3.52778,9.77052, 9.16828};
struct DetectBox{
    int classID;
    float class_conf;
    float dets_conf;
    double xmin;
    double xmax;
    double ymin;
    double ymax;
};

std::vector<DetectBox> FewShotDetectionOutput(
    std::vector<DetectBox> &detectBox,
    std::vector<float*> outputData,
    int netWidth,
    int netHeight,
    int imgWidth,
    int imgHeight,
    int classNum,
    int anchor_nums=5,
    float thresh = 0.005,
    float nmsThresh = 0.45,
    bool usePad = false
);

#endif

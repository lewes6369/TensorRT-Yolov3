#ifndef _YOLO_CONFIGS_H_
#define _YOLO_CONFIGS_H_

namespace Yolo
{
    const int CLUSTER_NUM = 9;
    float ANCHORS[CLUSTER_NUM*2] = {10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326};
    const float IGNORE_THRESH = 0.5;
    const float NMS_THRESH = 0.45;

    const int YOLO_LAYER_COUNT = 3;

    //Yolo608
    const float YOLO_KERNEL_SIZE[YOLO_LAYER_COUNT] = {19 , 38 , 76};
    //Yolo416
    //const float YOLO_KERNEL_SIZE[YOLO_LAYER_COUNT] = {13 , 26 , 52};

    const float YOLO_MASK[YOLO_LAYER_COUNT][3] = {{6,7,8},{3,4,5},{1,2,3}};
}

#endif
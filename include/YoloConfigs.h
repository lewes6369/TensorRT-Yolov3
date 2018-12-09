#ifndef _YOLO_CONFIGS_H_
#define _YOLO_CONFIGS_H_

namespace Yolo
{
    float ANCHORS[ ] = {10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326};
    const float IGNORE_THRESH = 0.5;
    const float NMS_THRESH = 0.45;
}

#endif
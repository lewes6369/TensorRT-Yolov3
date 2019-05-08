#ifndef _YOLO_CONFIGS_H_
#define _YOLO_CONFIGS_H_


namespace Yolo
{
    static constexpr int CHECK_COUNT = 3;
    static constexpr float IGNORE_THRESH = 0.3f;
    static constexpr int CLASS_NUM = 20;

    struct YoloKernel
    {
        int width;
        int height;
        float anchors[CHECK_COUNT*2];
    };

    //YOLO 608
    YoloKernel yolo1 = {
        10,
        10,
        {81,82,  135,169,  344,319}
    };
    YoloKernel yolo2 = {
        20,
        20,
        {10,14,  23,27,  37,58}
    };
    YoloKernel yolo3 = {
        40,
        40,
        {10,13,  16,30,  33,23}
    };

    //YOLO 416
    // YoloKernel yolo1 = {
    //     13,
    //     13,
    //     {116,90,  156,198,  373,326}
    // };
    // YoloKernel yolo2 = {
    //     26,
    //     26,
    //     {30,61,  62,45,  59,119}
    // };
    // YoloKernel yolo3 = {
    //     52,
    //     52,
    //     {10,13,  16,30,  33,23}
    // };
}

#endif
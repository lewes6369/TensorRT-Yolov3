#ifndef _CONFIGS_H_
#define _CONFIGS_H_

#include <string>
namespace Tn
{
    const int INPUT_CHANNEL = 3;
    const std::string INPUT_PROTOTXT ="yolov3_trt.prototxt";
    const std::string INPUT_CAFFEMODEL = "yolov3.caffemodel";
    const std::string INPUT_IMAGE = "test.jpg";
    const std::string CALIBRATION_LIST = "";
    const std::string MODE = "fp32";
    const std::string OUTPUTS= "layer82-conv,layer94-conv,layer106-conv";
    const int INPUT_WIDTH = 416;
    const int INPUT_HEIGHT = 416;

    const int DETECT_CLASSES = 20;
}

#endif
#include <string>
#include <sstream>
#include <algorithm>
#include <memory>
#include <opencv2/opencv.hpp>
#include "TrtNet.h"
#include "argsParser.h"
#include "configs.h"

using namespace std;
using namespace argsParser;
using namespace Tn;

vector<float> prepareImage(const string& fileName)
{
    using namespace cv;

    Mat img = imread(fileName);
    if(img.data== nullptr)
    {
        cout << "can not open image :" << fileName  << endl;
        return {}; 
    } 

    int c = parser::getIntValue("C");
    int h = parser::getIntValue("H");
    int w = parser::getIntValue("W");  

    cv::Mat resized;
    float scale = min(float(w)/img.cols,float(h)/img.rows);
    auto scaleSize = cv::Size(img.cols * scale,img.rows * scale);
    cv::resize(img, resized, cv::Size(w,h),(0.0),(0.0),INTER_CUBIC);

    Rect rect((w- scaleSize.width)/2, (h-scaleSize.height)/2, scaleSize.width,scaleSize.height); 
    cv::Mat cropped = resized(rect);

    cv::Mat img_float;
    if (INPUT_CHANNEL == 3)
        cropped.convertTo(img_float, CV_32FC3, 1/255.0);
    else
        cropped.convertTo(img_float, CV_32FC1 ,1/255.0);

    //HWC TO CHW
    cv::Mat input_channels[INPUT_CHANNEL];
    cv::split(img_float, input_channels);

    vector<float> result(h*w*c);
    auto data = result.data();
    int channelLength = h * w;
    for (int i = 0; i < c; ++i) {
        memcpy(data,input_channels[i].data,channelLength*sizeof(float));
        data += channelLength;
    }

    return result;
}

vector<string> split(const string& str, char delim)
{
    stringstream ss(str);
    string token;
    vector<string> container;
    while (getline(ss, token, delim)) {
        container.push_back(token);
    }

    return container;
}

int main( int argc, char* argv[] )
{
    parser::ADD_ARG_FLOAT("prototxt",Desc("input yolov3 deploy"),DefaultValue(INPUT_PROTOTXT),ValueDesc("file"));
    parser::ADD_ARG_FLOAT("caffemodel",Desc("input yolov3 caffemodel"),DefaultValue(INPUT_CAFFEMODEL),ValueDesc("file"));
    parser::ADD_ARG_STRING("input",Desc("input image file"),DefaultValue(INPUT_IMAGE),ValueDesc("file"));
    parser::ADD_ARG_INT("C",Desc("channel"),DefaultValue(to_string(INPUT_CHANNEL)));
    parser::ADD_ARG_INT("H",Desc("height"),DefaultValue(to_string(INPUT_HEIGHT)));
    parser::ADD_ARG_INT("W",Desc("width"),DefaultValue(to_string(INPUT_WIDTH)));
    parser::ADD_ARG_STRING("calib",Desc("calibration image List"),DefaultValue(CALIBRATION_LIST),ValueDesc("file"));
    parser::ADD_ARG_STRING("mode",Desc("runtime mode"),DefaultValue(MODE), ValueDesc("fp32/fp16/int8"));
    parser::ADD_ARG_STRING("outputNodes",Desc("output nodes name"),DefaultValue(OUTPUTS));

    if(argc < 2){
        parser::printDesc();
        exit(-1);
    }

    parser::parseArgs(argc,argv);

    string deployFile = parser::getStringValue("prototxt");
    string caffemodelFile = parser::getStringValue("caffemodel");

    vector<vector<float>> calibData;
    string calibFileList = parser::getStringValue("calib");
    string mode = parser::getStringValue("mode");
    if(calibFileList.length() > 0 && mode == "int8")
    {   
        cout << "find calibration file,loading ..." << endl;
      
        ifstream file(calibFileList);  
        if(!file.is_open())
        {
            cout << "read file list error,please check file :" << calibFileList << endl;
            exit(-1);
        }

        string strLine;  
        while( getline(file,strLine) )                               
        { 
            std::cout << strLine << std::endl;
            auto data = prepareImage(strLine);
            calibData.emplace_back(data);
        } 
        file.close();
    }

    RUN_MODE run_mode = RUN_MODE::FLOAT32;
    if(mode == "int8")
    {
        if(calibFileList.length() == 0)
            cout << "run int8 please input calibration file, will run in fp32" << endl;
        else
            run_mode = RUN_MODE::INT8;
    }
    else if(mode == "fp16")
    {
        run_mode = RUN_MODE::FLOAT16;
    }
    
    string outputNodes = parser::getStringValue("outputNodes");
    auto outputNames = split(outputNodes,',');

    trtNet net(deployFile,caffemodelFile,outputNames,calibData,run_mode);

    string imageName = parser::getStringValue("input");
    auto inputData = prepareImage(imageName);
    int outputCount = net.getOutputSize()/sizeof(float);
    unique_ptr<float[]> outputData(new float[outputCount]);

    net.doInference(inputData.data(), outputData.get());

    net.printTime();

    auto result = outputData.get();
    cout << "*************result************" << endl;
    //result
    for (int i = 0 ;i< outputCount; ++i)
        cout << " " << result[i] << " " << endl;
    
    //ADD: need to do yolo layer
    //processs yolo

    return 0;
}

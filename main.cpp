#include <string>
#include <sstream>
#include <algorithm>
#include <memory>
#include <opencv2/opencv.hpp>
#include "TrtNet.h"
#include "argsParser.h"
#include "configs.h"
#include <chrono>
#include "YoloLayer.h"

using namespace std;
using namespace argsParser;
using namespace Tn;

std::vector<float> prepareImage(const string& fileName,int* width = nullptr,int* height = nullptr)
{
    using namespace cv;

    Mat img = imread(fileName);
    if(img.data== nullptr)
    {
        std::cout << "can not open image :" << fileName  << std::endl;
        return {}; 
    } 

    if(width)
        *width = img.cols;

    if(height)
        *height = img.rows;

    int c = parser::getIntValue("C");
    int h = parser::getIntValue("H");
    int w = parser::getIntValue("W");  

    float scale = min(float(w)/img.cols,float(h)/img.rows);
    auto scaleSize = cv::Size(img.cols * scale,img.rows * scale);

    cv::Mat rgb ;
    cv::cvtColor(img, rgb, CV_BGR2RGB);
    cv::Mat resized;
    cv::resize(rgb, resized,scaleSize,0,0,INTER_CUBIC);

    cv::Mat cropped(h, w,CV_8UC3, 127);
    Rect rect((w- scaleSize.width)/2, (h-scaleSize.height)/2, scaleSize.width,scaleSize.height); 
    resized.copyTo(cropped(rect));

    cv::Mat img_float;
    if (INPUT_CHANNEL == 3)
        cropped.convertTo(img_float, CV_32FC3, 1/255.0);
    else
        cropped.convertTo(img_float, CV_32FC1 ,1/255.0);

    //HWC TO CHW
    cv::Mat input_channels[INPUT_CHANNEL];
    cv::split(img_float, input_channels);

    std::vector<float> result(h*w*c);
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
    parser::ADD_ARG_STRING("outputs",Desc("output nodes name"),DefaultValue(OUTPUTS));
    parser::ADD_ARG_INT("class",Desc("num of classes"),DefaultValue(to_string(DETECT_CLASSES)));

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

    
    //can load from file
    string saveName = "yolov3_" + mode + ".engine";
#ifdef LOAD_FROM_ENGINE    
    trtNet net(saveName);
#else
    trtNet net(deployFile,caffemodelFile,outputNames,calibData,run_mode);
    net.saveEngine(saveName);
#endif

    string inputFileName = parser::getStringValue("input");
    int width,height;
    auto inputData = prepareImage(inputFileName,&width,&height);

    int outputCount = net.getOutputSize()/sizeof(float);
    unique_ptr<float[]> outputData(new float[outputCount]);
    //for (int i = 0;i < 100 ;++i)
    net.doInference(inputData.data(), outputData.get());
    
    net.printTime();

    int nboxes = 0;
    int classes = parser::getIntValue("class");
    int h = parser::getIntValue("H");
    int w = parser::getIntValue("W");  
    auto t_start = std::chrono::high_resolution_clock::now();
    auto detects = get_detections(outputData.get(),width,height,w,h,&nboxes,classes);
    auto t_end = std::chrono::high_resolution_clock::now();
    float total = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    std::cout << "Time taken for yolo is " << total << " ms." << std::endl;
    
    cv::Mat img = cv::imread(inputFileName);
    printBox(detects,width,height,nboxes,classes,&img);
    free_detections(detects,nboxes);
    cv::imwrite("result.jpg",img);
    cv::imshow("result",img);
    cv::waitKey(0);


    return 0;
}

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
using namespace Yolo;

vector<float> prepareImage(cv::Mat& img)
{
    using namespace cv;

    int c = parser::getIntValue("C");
    int h = parser::getIntValue("H");   //net h
    int w = parser::getIntValue("W");   //net w

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

    vector<float> result(h*w*c);
    auto data = result.data();
    int channelLength = h * w;
    for (int i = 0; i < c; ++i) {
        memcpy(data,input_channels[i].data,channelLength*sizeof(float));
        data += channelLength;
    }

    return result;
}

void printBox(vector<Detection>& dets,int width,int height,int classes,cv::Mat* img /*= nullptr*/)
{
    for(const auto& item : dets)
    {
        auto& b = item.bbox;
        int left  = max((b[0]-b[2]/2.)*width,0.0);
        int right = min((b[0]+b[2]/2.)*width,double(width));
        int top   = max((b[1]-b[3]/2.)*height,0.0);
        int bot   = min((b[1]+b[3]/2.)*height,double(height));
        if (img) //draw rect
            cv::rectangle(*img,cv::Point(left,top),cv::Point(right,bot),cv::Scalar(0,0,255),3,8,0);
        cout << "class=" << item.classId << " prob=" << item.prob*100 << endl;
        cout << "left=" << left << " right=" << right << " top=" << top << " bot=" << bot << endl;
    }
}

void DoNms(vector<Detection>& detections,int classes ,float nmsThresh)
{
   auto t_start = chrono::high_resolution_clock::now();

    vector<vector<Detection>> resClass;
    resClass.resize(classes);

    for (const auto& item : detections)
        resClass[item.classId].push_back(item);

    auto iouCompute = [](float * lbox, float* rbox)
    {
        float interBox[] = {
            max(lbox[0] - lbox[2]/2.f , rbox[0] - rbox[2]/2.f), //left
            min(lbox[0] + lbox[2]/2.f , rbox[0] + rbox[2]/2.f), //right
            max(lbox[1] - lbox[3]/2.f , rbox[1] - rbox[3]/2.f), //top
            min(lbox[1] + lbox[3]/2.f , rbox[1] + rbox[3]/2.f), //bottom
        };
        
        if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
            return 0.0f;

        float interBoxS =(interBox[1]-interBox[0])*(interBox[3]-interBox[2]);
        return interBoxS/(lbox[2]*lbox[3] + rbox[2]*rbox[3] -interBoxS);
    };

    vector<Detection> result;
    for (int i = 0;i<classes;++i)
    {
        auto& dets =resClass[i]; 
        if(dets.size() == 0)
            continue;

        sort(dets.begin(),dets.end(),[=](const Detection& left,const Detection& right){
            return left.prob > right.prob;
        });

        for (unsigned int m = 0;m < dets.size() ; ++m)
        {
            auto& item = dets[m];
            result.push_back(item);
            for(unsigned int n = m + 1;n < dets.size() ; ++n)
            {
                if (iouCompute(item.bbox,dets[n].bbox) > nmsThresh)
                {
                    dets.erase(dets.begin()+n);
                    --n;
                }
            }
        }
    }

    //swap(detections,result);
    detections = move(result);

    auto t_end = chrono::high_resolution_clock::now();
    float total = chrono::duration<float, milli>(t_end - t_start).count();
    cout << "Time taken for nms is " << total << " ms." << endl;
}


void postProcessImg(cv::Mat& img,vector<Detection>& detections,int classes)
{
    using namespace cv;

    int h = parser::getIntValue("H");   //net h
    int w = parser::getIntValue("W");   //net w

    //scale bbox to img
    int width = img.cols;
    int height = img.rows;
    float scale = min(float(w)/width,float(h)/height);
    float scaleSize[] = {width * scale,height * scale};

    //correct box
    for (auto& item : detections)
    {
        auto& bbox = item.bbox;
        bbox[0] = (bbox[0] * w - (w - scaleSize[0])/2.f) / scaleSize[0];
        bbox[1] = (bbox[1] * h - (h - scaleSize[1])/2.f) / scaleSize[1];
        bbox[2] /= scaleSize[0];
        bbox[3] /= scaleSize[1];
    }
    
    //nms
    float nmsThresh = parser::getFloatValue("nms");
    if(nmsThresh > 0) 
        DoNms(detections,classes,nmsThresh);
    
    printBox(detections,width,height,classes,&img);
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
    parser::ADD_ARG_STRING("prototxt",Desc("input yolov3 deploy"),DefaultValue(INPUT_PROTOTXT),ValueDesc("file"));
    parser::ADD_ARG_STRING("caffemodel",Desc("input yolov3 caffemodel"),DefaultValue(INPUT_CAFFEMODEL),ValueDesc("file"));
    parser::ADD_ARG_STRING("input",Desc("input image file"),DefaultValue(INPUT_IMAGE),ValueDesc("file"));
    parser::ADD_ARG_INT("C",Desc("channel"),DefaultValue(to_string(INPUT_CHANNEL)));
    parser::ADD_ARG_INT("H",Desc("height"),DefaultValue(to_string(INPUT_HEIGHT)));
    parser::ADD_ARG_INT("W",Desc("width"),DefaultValue(to_string(INPUT_WIDTH)));
    parser::ADD_ARG_STRING("calib",Desc("calibration image List"),DefaultValue(CALIBRATION_LIST),ValueDesc("file"));
    parser::ADD_ARG_STRING("mode",Desc("runtime mode"),DefaultValue(MODE), ValueDesc("fp32/fp16/int8"));
    parser::ADD_ARG_STRING("outputs",Desc("output nodes name"),DefaultValue(OUTPUTS));
    parser::ADD_ARG_INT("class",Desc("num of classes"),DefaultValue(to_string(DETECT_CLASSES)));
    parser::ADD_ARG_FLOAT("nms",Desc("non-maximum suppression value"),DefaultValue(to_string(NMS_THRESH)));

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
            cv::Mat img = cv::imread(strLine);
            auto data = prepareImage(img);
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
    
    string outputNodes = parser::getStringValue("outputs");
    auto outputNames = split(outputNodes,',');
    
    //can load from file
    string saveName = "yolov3_" + mode + ".engine";
//#define LOAD_FROM_ENGINE
#ifdef LOAD_FROM_ENGINE    
    trtNet net(saveName);
#else
    trtNet net(deployFile,caffemodelFile,outputNames,calibData,run_mode);
    cout << "save Engine..." <<endl;
    net.saveEngine(saveName);
    cout << "save Engine ok (you can manual load next time)" <<endl;
#endif

    string inputFileName = parser::getStringValue("input");
    cv::Mat img = cv::imread(inputFileName);
    vector<float> inputData = prepareImage(img);

    int outputCount = net.getOutputSize()/sizeof(float);
    unique_ptr<float[]> outputData(new float[outputCount]);
    
    //for (int i = 0;i < 1000 ;++i)
    net.doInference(inputData.data(), outputData.get());
    
    net.printTime();

    //Get Output    
    auto output = outputData.get();
    int count = output[0];
    std::cout << "detCount: "<< count << std::endl;
    
    vector<Detection> result;
    result.resize(count);
    //first detect count
    output ++ ;
    //later detect result
    memcpy(result.data(), output , count*sizeof(Detection));

    int classes = parser::getIntValue("class");
    postProcessImg(img,result,classes);

    cv::imwrite("result.jpg",img);
    cv::imshow("result",img);
    cv::waitKey(0);
    return 0;
}

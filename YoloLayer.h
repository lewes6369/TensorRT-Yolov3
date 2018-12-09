#ifndef __YOLO_LAYER_H_
#define __YOLO_LAYER_H_

#include <string>
#include <vector>
#include <memory>
#include <string>
#include <opencv2/opencv.hpp>

typedef struct{
    float x,y,w,h;
}box;

typedef struct{
    box bbox;
    int classes;
    float* prob;
    float* mask;
    float objectness;
    int sort_class;
}detection;

typedef struct layer{
    int batch;
    int total;
    int n,c,h,w;
    int out_n,out_c,out_h,out_w;
    int classes;
    int inputs,outputs;
    int *mask;
    float* biases;
    float* output;
    float* output_gpu;
}layer;

detection* get_detections(const void* data,int img_w,int img_h,int net_w,int net_h,int *nboxes,int classes);

void printBox(detection* dets,int width,int height,int nboxes,int classes,cv::Mat* img = nullptr);

void free_detections(detection *dets,int nboxes);

#endif

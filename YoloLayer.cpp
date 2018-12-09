#include "YoloLayer.h"
#include "YoloConfigs.h"
#include "box.h"
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <vector>
#include <algorithm>

using namespace std;
using namespace Yolo;

layer make_yolo_layer(int batch,int w,int h,int n,int total,int classes)
{
    layer l = {0};
    l.n = n;
    l.total = total;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = n*(classes+ 4 + 1);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.inputs = l.w*l.h*l.c;

    l.biases = (float*)calloc(total*2,sizeof(float));
    for(int i =0;i<total*2;++i){
        l.biases[i] = ANCHORS[i];
    }
    l.mask = (int*)calloc(n,sizeof(int));
    if(l.w == 13){
        int j = 6;
        for(int i =0;i<l.n;++i)
            l.mask[i] = j++;
    }
    if(l.w == 26){
        int j = 3;
        for(int i =0;i<l.n;++i)
            l.mask[i] = j++;
    }
    if(l.w == 52){
        int j = 0;
        for(int i =0;i<l.n;++i)
            l.mask[i] = j++;
    }
    l.outputs = l.inputs;
    l.output = (float*)calloc(batch*l.outputs,sizeof(float));
    return l;
}

void free_yolo_layer(layer l)
{
    if(NULL != l.biases){
        free(l.biases);
        l.biases = NULL;
    }

    if(NULL != l.mask){
        free(l.mask);
        l.mask = NULL;
    }
    if(NULL != l.output){
        free(l.output);
        l.output = NULL;
    }

}

static int entry_index(layer l,int batch,int location,int entry)
{
    int n = location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    return batch*l.outputs + n*l.w*l.h*(4 + l.classes + 1) + entry*l.w*l.h + loc;
 }


 void forward_yolo_layer(const float* input,layer l)
{
    memcpy(l.output, (float*)input, l.outputs*l.batch*sizeof(float));

    int b,n;
    for(b = 0;b < l.batch;++b){
    for(n =0;n< l.n;++n){

            int index = entry_index(l,b,n*l.w*l.h,0);
            auto data = l.output + index;
            for(int i = 0; i < 2*l.w*l.h; ++i){
                data[i] = 1./(1. + exp(-data[i]));
            }

            index = entry_index(l,b,n*l.w*l.h,4);
            data = l.output + index;
            for(int i = 0; i < (1 + l.classes)*l.w*l.h; ++i){
                data[i] = 1./(1. + exp(-data[i]));
            }
        }
    }
}

int yolo_num_detections(layer l,float thresh)
{
    int i,n;
    int count = 0;
    for(i=0;i<l.w*l.h;++i){
        for(n=0;n<l.n;++n){
            int obj_index = entry_index(l,0,n*l.w*l.h+i,4);
            if(l.output[obj_index] > thresh)
                ++count;
        }
    }
    return count;
}

int num_detections(vector<layer> layers_params,float thresh)
{
    int i;
    int s=0;
    for(i=0;i<layers_params.size();++i){
        layer l  = layers_params[i];
        s += yolo_num_detections(l,thresh);
    }
    return s;

}

detection* make_network_boxes(vector<layer> layers_params,float thresh,int* num)
{
    layer l = layers_params[0];
    int i;
    int nboxes = num_detections(layers_params,thresh);
    if(num) *num = nboxes;
    detection *dets = (detection*)calloc(nboxes,sizeof(detection));
    for(i=0;i<nboxes;++i){
        dets[i].prob = (float*)calloc(l.classes,sizeof(float));
    }
    return dets;
}


void correct_yolo_boxes(detection* dets,int n,int w,int h,int netw,int neth)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h)){
        new_w = netw;
        new_h = (h * netw)/w;
    }
    else{
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (i = 0; i < n; ++i){
        box b = dets[i].bbox;
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw);
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth);
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        dets[i].bbox = b;
    }
}


box get_yolo_box(float* x,float* biases,int n,int index,int i,int j,int lw, int lh,int w, int h,int stride)
{
    box b;
    b.x = (i + x[index + 0*stride]) / lw;
    b.y = (j + x[index + 1*stride]) / lh;
    b.w = exp(x[index + 2*stride]) * biases[2*n] / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n + 1] / h;
    return b;
}


int get_yolo_detections(layer l,int w, int h, int netw,int neth,float thresh,detection *dets)
{
    int i,j,n;
    float* predictions = l.output;
    int count = 0;
    for(i=0;i<l.w*l.h;++i){
        int row = i/l.w;
        int col = i%l.w;
        for(n = 0;n<l.n;++n){           
            int obj_index = entry_index(l,0,n*l.w*l.h + i,4);
            float objectness = predictions[obj_index];
            if(objectness <= thresh) continue;
            int box_index = entry_index(l,0,n*l.w*l.h + i,0);

            dets[count].bbox = get_yolo_box(predictions,l.biases,l.mask[n],box_index,col,row,l.w,l.h,netw,neth,l.w*l.h);
            dets[count].objectness = objectness;
            dets[count].classes = l.classes;
            for(j=0;j<l.classes;++j){
                int class_index = entry_index(l,0,n*l.w*l.h+i,4+1+j);
                float prob = objectness*predictions[class_index];
                //std::cout<<"("<<i<<","<<j<<"):"<<prob<<" "<<std::endl;
                dets[count].prob[j] = (prob > thresh) ? prob : 0;
            }
            ++count;
        }
    }
    correct_yolo_boxes(dets,count,w,h,netw,neth);
    return count;
}


detection* get_network_boxes(vector<layer> layers_params,
                             int img_w,int img_h,int net_w,int net_h,float thresh,int *num)
{
    //make network boxes
    detection *dets = make_network_boxes(layers_params,thresh,num);

    auto result = dets;
    for(int j=0;j<layers_params.size();++j){
        layer l = layers_params[j];
        int count = get_yolo_detections(l,img_w,img_h,net_w,net_h,thresh,dets);
        dets += count;
    }
    
    return result;
}

//get detection result
detection* get_detections(const void* data,int img_w,int img_h,int net_w,int net_h,int *nboxes,int classes)
{
    vector<layer> layers_params;
    layers_params.clear();

    static std::pair<int,int> yolo_size[3] = {
        {13,13},
        {26,26},
        {52,52}
    };

    const float* f_data  = static_cast<const float*>(data);

    for(int i=0;i< 3;++i){
        int width = yolo_size[i].first;
        int height = yolo_size[i].second;
        int chn_size =  width*height*(3*(classes + 4 + 1));
        layer l_params = make_yolo_layer(1,width,height,3,9,classes);
        layers_params.push_back(l_params);
        forward_yolo_layer(f_data ,l_params);
        f_data = f_data + chn_size;
    }

    //get network boxes
    detection* dets = get_network_boxes(layers_params,img_w,img_h,net_w,net_h,IGNORE_THRESH,nboxes);
    std::cout<<*nboxes<<std::endl;
    //release layer memory
    for(int index =0;index < layers_params.size();++index){
        free_yolo_layer(layers_params[index]);
    }

    if(NMS_THRESH) do_nms_sort(dets,(*nboxes),classes,NMS_THRESH);

    return dets;       
}

//release detection memory
void free_detections(detection *dets,int nboxes)
{
    int i;
    for(i = 0;i<nboxes;++i){
        free(dets[i].prob);
    }
    free(dets);
}

void printBox(detection* dets,int width,int height,int nboxes,int classes,cv::Mat* img /*= nullptr*/)
{
    for(int i=0;i< nboxes;++i){
        int cls = -1;
        float maxprob = IGNORE_THRESH;
        for(int j=0;j<classes;++j){
            if(dets[i].prob[j] > maxprob){
                cls = j;
                maxprob = dets[i].prob[j];
                std::cout << "class :  "<<cls << " prob: " << maxprob *100;
            }
        }
        if(cls >= 0){
            box b = dets[i].bbox;
            int left  = std::max((b.x-b.w/2.)*width,0.0);
            int right = std::min((b.x+b.w/2.)*width,double(width));
            int top   = std::max((b.y-b.h/2.)*height,0.0);
            int bot   = std::min((b.y+b.h/2.)*height,double(height));
            if (img) //draw rect
                cv::rectangle(*img,cv::Point(left,top),cv::Point(right,bot),cv::Scalar(0,0,255),3,8,0);
            std::cout << "left=" << left << "right=" << right << "top=" << top << "bot=" << bot << std::endl;
        }
    }
}


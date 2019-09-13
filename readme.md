# TRTForYolov3

## Desc

    tensorRT for Yolov3

### Test Enviroments

    Ubuntu  16.04
    TensorRT 5.0.2.6/4.0.1.6
    CUDA 9.2

### Models

Download the caffe model converted by official model:

+ Baidu Cloud [here](https://pan.baidu.com/s/1VBqEmUPN33XrAol3ScrVQA) pwd: gbue
+ Google Drive [here](https://drive.google.com/open?id=18OxNcRrDrCUmoAMgngJlhEglQ1Hqk_NJ)


If run model trained by yourself, comment the "upsample_param" blocks, and modify the prototxt the last layer as:
```
layer {
    #the bottoms are the yolo input layers
    bottom: "layer82-conv"
    bottom: "layer94-conv"
    bottom: "layer106-conv"
    top: "yolo-det"
    name: "yolo-det"
    type: "Yolo"
}
```

It also needs to change the yolo configs in "YoloConfigs.h" if different kernels.

### Run Sample

```bash
#build source code
git submodule update --init --recursive
mkdir build
cd build && cmake .. && make && make install
cd ..

#for yolov3-608
./install/runYolov3 --caffemodel=./yolov3_608.caffemodel --prototxt=./yolov3_608.prototxt --input=./test.jpg --W=608 --H=608 --class=80

#for fp16
./install/runYolov3 --caffemodel=./yolov3_608.caffemodel --prototxt=./yolov3_608.prototxt --input=./test.jpg --W=608 --H=608 --class=80 --mode=fp16

#for int8 with calibration datasets
./install/runYolov3 --caffemodel=./yolov3_608.caffemodel --prototxt=./yolov3_608.prototxt --input=./test.jpg --W=608 --H=608 --class=80 --mode=int8 --calib=./calib_sample.txt

#for yolov3-416 (need to modify include/YoloConfigs for YoloKernel)
./install/runYolov3 --caffemodel=./yolov3_416.caffemodel --prototxt=./yolov3_416.prototxt --input=./test.jpg --W=416 --H=416 --class=80
```


### Performance

Model | GPU | Mode | Inference Time
-- | -- | -- | -- 
Yolov3-416 |  GTX 1060 | Caffe | 54.593ms
Yolov3-416 |  GTX 1060 | float32 | 23.817ms
Yolov3-416 |  GTX 1060 | int8 | 11.921ms
Yolov3-608 |  GTX 1060 | Caffe | 88.489ms
Yolov3-608 | GTX 1060 | float32 | 43.965ms
Yolov3-608 |  GTX 1060 | int8 | 21.638ms
Yolov3-608 | GTX 1080 Ti | float32 | 19.353ms
Yolov3-608 | GTX 1080 Ti | int8 | 9.727ms
Yolov3-416 |  GTX 1080 Ti | float32 | 9.677ms
Yolov3-416 |  GTX 1080 Ti | int8 | 6.129ms  | li

### Eval Result

run above models with appending ```--evallist=labels.txt```

int8 calibration data made from 200 pics selected in val2014 (see scripts dir)

Model | GPU | Mode | dataset | MAP(0.50) | MAP(0.75)
-- | -- | -- | -- | -- | --
Yolov3-416 | GTX 1060 | Caffe(fp32) | COCO val2014 | 50.33 | 33.00
Yolov3-416 | GTX 1060 | float32 | COCO val2014 | 50.27 | 32.98
Yolov3-416 | GTX 1060 | int8 | COCO val2014 | 44.15 | 30.24
Yolov3-608 | GTX 1060 | Caffe(fp32) | COCO val2014 | 52.89 | 35.31
Yolov3-608 | GTX 1060 | float32 | COCO val2014 |  52.84 | 35.26
Yolov3-608 | GTX 1060 | int8 | COCO val2014 |  48.55 | 35.53 | li


Notice: 
+ caffe implementation is little different in yolo layer and nms, and it should be the similar result compared to tensorRT fp32. 

### Details About Wrapper

see link [TensorRTWrapper](https://github.com/lewes6369/tensorRTWrapper)

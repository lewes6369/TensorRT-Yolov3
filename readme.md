# TRTForYolov3

<a href="https://996.icu"><img src="https://img.shields.io/badge/link-996.icu-red.svg" alt="996.icu" /></a>

## Desc

    tensorRT for Yolov3

### Test Enviroments

    Ubuntu  16.04
    TensorRT 5.1/5.0.2.6/4.0.1.6
    CUDA 9.2 or CUDA 9.0 or CUDA 10.0

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

# build source code

git submodule update --init --recursive
mkdir build
cd build && cmake .. && make && make install
cd ..


# what I do

1.Added multithreading

2.Added tag name

3.Added video inference


# for yolov3-608

## video

./install/runYolov3 --caffemodel=./yolov3_608.caffemodel --prototxt=./yolov3_608.prototxt --display=1 --inputstream=video --videofile=sample_720p.mp4 --classname=coco.names

## cam

./install/runYolov3 --caffemodel=./yolov3_608.caffemodel --prototxt=./yolov3_608.prototxt --display=1 --inputstream=cam --cam=0 --classname=coco.names

## example

![图片alt](https://github.com/talebolano/TensorRT-Yolov3/image/example.png ''example'')

### Performance

Model | GPU | Mode | Inference Time | FPS
-- | -- | -- | -- | -- |
Yolov3-608 | GTX 1060(laptop) | float32 | 58.965ms | 15
Yolov3-608 | P40 | float32 | 20.353ms | 40




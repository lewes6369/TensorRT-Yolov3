# TRTForYolov3

## Desc

    tensorRT for Yolov3

### Test Enviroments

    Ubuntu  16.04
    TensorRT 5.0.2.6/4.0.1.6
    CUDA 9.2

### Models

Download the model converted by official model [here](https://pan.baidu.com/s/1VBqEmUPN33XrAol3ScrVQA) pwd: gbue

If run model trained by yourself, modify the prototxt the last layer as:(the bottoms are the yolo input layers)
```
layer {
    bottom: "layer82-conv"
    bottom: "layer94-conv"
    bottom: "layer106-conv"
    top: "yolo-det"
    name: "yolo-det"
    type: "Yolo"
}
```

It needs to change the yolo configs in "YoloConfigs.h" if different kernels.

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
Yolov3-608 | GTX 1060 | float32 | 43.965ms
Yolov3-416 |  GTX 1060 | float32 | 23.817ms
Yolov3-608 | GTX 1080 Ti | float32 | 19.353ms
Yolov3-608 | GTX 1080 Ti | int8 | 9.727ms
Yolov3-416 |  GTX 1080 Ti | float32 | 9.677ms
Yolov3-416 |  GTX 1080 Ti | int8 | 6.129ms  | li


### Details About Wrapper

see link [TensorRTWrapper](https://github.com/lewes6369/tensorRTWrapper)

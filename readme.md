# TRTForYolov3

### Desc

    tensorRT for Yolov3

### Test Enviroments

    Ubuntu  16.04
    TensorRT 4.0.1.6
    CUDA 9.2

### Run Sample
```bash
cd sample
mkdir build
cd build && cmake .. && make && make install
cd ..
#for yolov3-608
./install/runYolov3 --caffemodel=./yolov3_608.caffemodel --prototxt=./yolov3_608.prototxt --input=./test.jpg --W=608 --H=608 --class=80

#for fp16
./install/runYolov3 --caffemodel=./yolov3_608.caffemodel --prototxt=./yolov3_608.prototxt --input=./test.jpg --W=608 --H=608 --class=80 --mode=fp16

#for int8 with calibration datasets
./install/runYolov3 --caffemodel=./yolov3_608.caffemodel --prototxt=./yolov3_608.prototxt --input=./test.jpg --W=608 --H=608 --class=80 --mode=int8 --calib=./calib_sample.txt

#for yolov3-416 (need to modify include/YoloConfigs for YOLO_KERNEL_SIZE)
./install/runYolov3 --caffemodel=./yolov3_416.caffemodel --prototxt=./yolov3_416.prototxt --input=./test.jpg --W=416 --H=416 --class=20
```



### Performance
Model | GPU | Mode | Inference Time
-- | -- | -- | -- 
Yolov3-608 | GTX 1060 | float32 | 48.599ms
Yolov3-416 |  GTX 1060 | float32 | 24.177ms
Yolov3-608 | GTX 1080 Ti | float32 | 19.353ms
Yolov3-608 | GTX 1080 Ti | int8 | 9.727ms
Yolov3-416 |  GTX 1080 Ti | float32 | 9.677ms
Yolov3-416 |  GTX 1080 Ti | int8 | 6.129ms  | li

(Ps:The yolo layer time are not included. It may cost 1~2ms depending on cpu)

### Details About Wrapper
see link [TensorRTWrapper](https://github.com/lewes6369/tensorRTWrapper)

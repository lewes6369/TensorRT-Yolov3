# TRTForYolov3

### Desc

    tensorRT for Yolov3

### Test Enviroments

    Ubuntu  16.04
    TensorRT 4.0.1.6
    CUDA 9.2

### Run Sample
```bash
#for classification
cd sample
mkdir build
cd build && cmake .. && make && make install
cd ..
#for yolov3-608
./install/runYolov3 --caffemodel=./yolov3_608.caffemodel --prototxt=./yolov3_608.prototxt --input=./test.jpg --W=608 --H=608 --class=80

#for yolov3-416 (need to modify include/YoloConfigs for YOLO_KERNEL_SIZE)
./install/runYolov3 --caffemodel=./yolov3_416.caffemodel --prototxt=./yolov3_416.prototxt --input=./test.jpg --W=416 --H=416 --class=20
```



### Performance

### Details About Wrapper
see link [TensorRTWrapper](https://github.com/lewes6369/tensorRTWrapper)

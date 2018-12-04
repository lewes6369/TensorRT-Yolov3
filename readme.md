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
./install/runNet --input=./test.jpg
```
### More Details
see link [TensorRTWrapper](https://github.com/lewes6369/tensorRTWrapper)

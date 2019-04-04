#ifndef _UPSAMPLE_LAYER_H
#define _UPSAMPLE_LAYER_H

#include <assert.h>
#include <cmath>
#include <string.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include "NvInfer.h"
#include "Utils.h"
#include <iostream>

namespace nvinfer1
{
    class UpsampleLayerPlugin: public IPluginExt
    {
    public:
        explicit UpsampleLayerPlugin(const float scale, const int cudaThread = 512);
        // create the plugin at runtime from a byte stream
        UpsampleLayerPlugin(const void* data, size_t length);

        ~UpsampleLayerPlugin();

        int getNbOutputs() const override
        {
            return 1;
        }

        Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

        bool supportsFormat(DataType type, PluginFormat format) const override { 
            //std::cout << "supportsFormat=== type:"  << int(type) << "format" << int(format) << std::endl;
            return (type == DataType::kFLOAT || type == DataType::kHALF || type == DataType::kINT8 ) 
            && format == PluginFormat::kNCHW; 
        }

        void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override;

        int initialize() override;

        virtual void terminate() override {
        };

        virtual size_t getWorkspaceSize(int maxBatchSize) const override { return 0;}

        virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override;

        virtual size_t getSerializationSize() override
        {
            return sizeof(nvinfer1::Dims) + sizeof(mDataType) + sizeof(mScale)
            + sizeof(mOutputWidth) + sizeof(mOutputHeight) + sizeof(mThreadCount);
        }

        virtual void serialize(void* buffer) override;

        template <typename Dtype>
        void forwardGpu(const Dtype* input,Dtype * outputint ,int N,int C,int H ,int W);

    private:
        nvinfer1::Dims mCHW;
        DataType mDataType{DataType::kFLOAT};
        float mScale;
        int mOutputWidth;
        int mOutputHeight;
        int mThreadCount;
        
        void* mInputBuffer  {nullptr}; 
        void* mOutputBuffer {nullptr}; 
    };
};

#endif 

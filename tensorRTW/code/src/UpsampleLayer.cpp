#include "UpsampleLayer.h"

namespace nvinfer1
{
  UpsampleLayerPlugin::UpsampleLayerPlugin(const float scale, const int cudaThread /*= 512*/)
  : mScale(scale),mThreadCount(cudaThread)
  {
  }
  
  UpsampleLayerPlugin::~UpsampleLayerPlugin()
  {
  
  }
  
  // create the plugin at runtime from a byte stream
  UpsampleLayerPlugin::UpsampleLayerPlugin(const void* data, size_t length)
  {
      using namespace Tn;
      const char *d = reinterpret_cast<const char *>(data), *a = d;
      read(d, mCHW);
      read(d, mDataType);
      read(d, mScale);
      read(d, mOutputWidth);
      read(d, mOutputHeight);
      read(d, mThreadCount);
  
      //std::cout << "read:" << a << " " << mOutputWidth<< " " <<mOutputHeight<<std::endl;
      assert(d == a + length);
  }
  
  void UpsampleLayerPlugin::serialize(void* buffer)
  {
      using namespace Tn;
      char* d = static_cast<char*>(buffer), *a = d;
      write(d, mCHW);
      write(d, mDataType);
      write(d, mScale);
      write(d, mOutputWidth);
      write(d, mOutputHeight);
      write(d, mThreadCount);
  
      //std::cout << "write:" << a << " " << mOutputHeight<< " " <<mOutputWidth<<std::endl;
      assert(d == a + getSerializationSize());
  }
  
  int UpsampleLayerPlugin::initialize()
  {
      int inputHeight = mCHW.d[1];
      int inputWidth = mCHW.d[2];
      
      mOutputHeight = inputHeight * mScale;
      mOutputWidth = inputWidth * mScale;
  
      return 0;
  }
  
  void UpsampleLayerPlugin::configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize)
  {
      //std::cout << "type " << int(type) << "format " << (int)format <<std::endl;
      assert((type == DataType::kFLOAT || type == DataType::kHALF || type == DataType::kINT8) && format == PluginFormat::kNCHW);
      mDataType = type;
      
      //std::cout << "configureWithFormat:" <<inputDims[0].d[0]<< " " <<inputDims[0].d[1] << " "<<inputDims[0].d[2] <<std::endl;
  }
  
  //it is called prior to any call to initialize().
  Dims UpsampleLayerPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
  {
      //std::cout <<"Input:" << inputs[0].d[0] << " "<<inputs[0].d[1]<< " "<<inputs[0].d[2]<<std::endl;
      //std::cout <<"nbInputDims : "<<nbInputDims<< " input:" << inputs[0].nbDims << std::endl;
  
      mCHW = inputs[0];
      mOutputHeight = inputs[0].d[1]* mScale;
      mOutputWidth = inputs[0].d[2]* mScale;
      //std::cout << "ouputDims:" << mCHW.d[0] << " " << mOutputHeight << " " << mOutputWidth << std::endl;
      return Dims3(mCHW.d[0], mOutputHeight, mOutputWidth);
  }
}

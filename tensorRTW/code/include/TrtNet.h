#ifndef __TRT_NET_H_
#define __TRT_NET_H_

#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <numeric>
#include "NvInferPlugin.h"
#include "NvCaffeParser.h"
#include "PluginFactory.h"
#include "Utils.h"

namespace Tn
{
    enum class RUN_MODE
    {
        FLOAT32 = 0,
        FLOAT16 = 1,    
        INT8 = 2
    };

    class trtNet 
    {
        public:
            //Load from caffe model
			/* trtNet(const std::string& prototxt, const std::string& caffemodel, const std::vector<std::string>& outputNodesName,
				const std::vector<std::vector<float>>& calibratorData, RUN_MODE mode /*= RUN_MODE::FLOAT32) :mTrtContext(nullptr), mTrtEngine(nullptr), mTrtRunTime(nullptr), mTrtRunMode(mode), mTrtInputCount(0), mTrtIterationTime(0) {};
			 */
			trtNet() :mTrtContext(nullptr), mTrtEngine(nullptr), mTrtRunTime(nullptr), mTrtInputCount(0), mTrtIterationTime(0) {};
          

            ~trtNet()
            {
                // Release the stream and the buffers
                cudaStreamSynchronize(mTrtCudaStream);
                cudaStreamDestroy(mTrtCudaStream);
                for(auto& item : mTrtCudaBuffer)
                    cudaFree(item);

                mTrtPluginFactory.destroyPlugin();

                if(!mTrtRunTime)
                    mTrtRunTime->destroy();
                if(!mTrtContext)
                    mTrtContext->destroy();
                if(!mTrtEngine)
                    mTrtEngine->destroy();
            };

			void trtNet_engine(const std::string& prototxt, const std::string& caffeModel, const std::vector<std::string>& outputNodesName,
				const std::vector<std::vector<float>>& calibratorData, RUN_MODE mode = RUN_MODE::FLOAT32);

			void set_mode(RUN_MODE mode)
			{
				mTrtRunMode = mode;
			}
			//Load from engine file
			void trtNet_engine(const std::string& engineFile);

            void saveEngine(std::string fileName)
            {
                if(mTrtEngine)
                {
                    nvinfer1::IHostMemory* data = mTrtEngine->serialize();
                    std::ofstream file;
                    file.open(fileName,std::ios::binary | std::ios::out);
                    if(!file.is_open())
                    {
                        std::cout << "read create engine file" << fileName <<" failed" << std::endl;
                        return;
                    }

                    file.write((const char*)data->data(), data->size());
                    file.close();
                }
            };

            void doInference(const void* inputData, void* outputData);
            
            inline size_t getInputSize() {
                return std::accumulate(mTrtBindBufferSize.begin(), mTrtBindBufferSize.begin() + mTrtInputCount,0);
            };

            inline size_t getOutputSize() {
                return std::accumulate(mTrtBindBufferSize.begin() + mTrtInputCount, mTrtBindBufferSize.end(),0);
            };
            
            void printTime()
            {
                mTrtProfiler.printLayerTimes(mTrtIterationTime);
            }

        private:
                nvinfer1::ICudaEngine* loadModelAndCreateEngine(const char* deployFile, const char* modelFile,int maxBatchSize,
                                        nvcaffeparser1::ICaffeParser* parser, nvcaffeparser1::IPluginFactory* pluginFactory,
                                        nvinfer1::IInt8Calibrator* calibrator, nvinfer1::IHostMemory*& trtModelStream,const std::vector<std::string>& outputNodesName);

                void InitEngine();

                nvinfer1::IExecutionContext* mTrtContext;
                nvinfer1::ICudaEngine* mTrtEngine;
                nvinfer1::IRuntime* mTrtRunTime;
                PluginFactory mTrtPluginFactory;    
                cudaStream_t mTrtCudaStream;
                Profiler mTrtProfiler;
                RUN_MODE mTrtRunMode;

                std::vector<void*> mTrtCudaBuffer;
                std::vector<int64_t> mTrtBindBufferSize;
                int mTrtInputCount;
                int mTrtIterationTime;
    };
}

#endif //__TRT_NET_H_

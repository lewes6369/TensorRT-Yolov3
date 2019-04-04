#ifndef __PLUGIN_FACTORY_H_
#define __PLUGIN_FACTORY_H_

#include <vector>
#include <memory>
#include <regex>
#include "UpsampleLayer.h"
#include "YoloLayer.h"
#include "NvInferPlugin.h"
#include "NvCaffeParser.h"

namespace Tn
{
    static constexpr float NEG_SLOPE = 0.1;
    static constexpr float UPSAMPLE_SCALE = 2.0;
    static constexpr int CUDA_THREAD_NUM = 512;

    // Integration for serialization.
    using nvinfer1::plugin::INvPlugin;
    using nvinfer1::plugin::createPReLUPlugin;
    using nvinfer1::UpsampleLayerPlugin;
    using nvinfer1::YoloLayerPlugin;
    class PluginFactory : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactoryExt
    {
        public:
            inline bool isLeakyRelu(const char* layerName)
            {
                return std::regex_match(layerName , std::regex(R"(layer(\d*)-act)"));
            }

            inline bool isUpsample(const char* layerName)
            {
                return std::regex_match(layerName , std::regex(R"(layer(\d*)-upsample)"));
            }

            inline bool isYolo(const char* layerName)
            {
                return strcmp(layerName,"yolo-det") == 0;
            }

            virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights) override
            {
                assert(isPlugin(layerName));

                if(isLeakyRelu(layerName))
                {
                    assert(nbWeights == 0 && weights == nullptr);
                    mPluginLeakyRelu.emplace_back(std::unique_ptr<INvPlugin, void(*)(INvPlugin*)>(createPReLUPlugin(NEG_SLOPE), nvPluginDeleter));
                    return mPluginLeakyRelu.back().get();
                }
                else if (isUpsample(layerName))
                {
                    assert(nbWeights == 0 && weights == nullptr);
                    mPluginUpsample.emplace_back(std::unique_ptr<UpsampleLayerPlugin>(new UpsampleLayerPlugin(UPSAMPLE_SCALE,CUDA_THREAD_NUM)));
                    return mPluginUpsample.back().get();
                }
                else if (isYolo(layerName))
                {
                    assert(nbWeights == 0 && weights == nullptr && mPluginYolo.get() ==  nullptr);
                    mPluginYolo.reset(new YoloLayerPlugin(CUDA_THREAD_NUM));
                    return mPluginYolo.get();
                }
                else
                {
                assert(0);
                return nullptr;
                }
            }

        nvinfer1::IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override
        {
            assert(isPlugin(layerName));

            if (isLeakyRelu(layerName))
            {
                mPluginLeakyRelu.emplace_back(std::unique_ptr<INvPlugin, void (*)(INvPlugin*)>(createPReLUPlugin(serialData, serialLength), nvPluginDeleter));
                return mPluginLeakyRelu.back().get();
            }
            else if (isUpsample(layerName))
            {
                mPluginUpsample.emplace_back(std::unique_ptr<UpsampleLayerPlugin>(new UpsampleLayerPlugin(serialData, serialLength)));
                return mPluginUpsample.back().get();
            }
            else if (isYolo(layerName))
            {
                assert(mPluginYolo.get() ==  nullptr);
                mPluginYolo.reset(new YoloLayerPlugin(serialData, serialLength));
                return mPluginYolo.get();
            }
            else
            {
                assert(0);
                return nullptr;
            }
        }


        bool isPlugin(const char* name) override
        {
            return isPluginExt(name);
        }

        bool isPluginExt(const char* name) override
        {
            //std::cout << "check plugin " << name  << isYolo(name)<< std::endl;
            return isLeakyRelu(name) || isUpsample(name) || isYolo(name); 
        }

        // The application has to destroy the plugin when it knows it's safe to do so.
        void destroyPlugin()
        {
            for (auto& item : mPluginLeakyRelu)
                item.reset();

            for (auto& item : mPluginUpsample)
                item.reset();

            mPluginYolo.reset();
        }

        void (*nvPluginDeleter)(INvPlugin*){[](INvPlugin* ptr) { if(ptr) ptr->destroy(); }};

        std::vector<std::unique_ptr<INvPlugin,void (*)(INvPlugin*)>> mPluginLeakyRelu{};
        std::vector<std::unique_ptr<UpsampleLayerPlugin>> mPluginUpsample{};
        std::unique_ptr<YoloLayerPlugin> mPluginYolo {nullptr};
    };
}

#endif
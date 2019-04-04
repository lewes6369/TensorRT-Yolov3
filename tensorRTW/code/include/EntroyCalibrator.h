#ifndef _ENTROY_CALIBRATOR_H
#define _ENTROY_CALIBRATOR_H

#include <cudnn.h>
#include <string>
#include <vector>
#include "NvInfer.h"
#include "Utils.h"

namespace nvinfer1
{

class Int8EntropyCalibrator : public IInt8EntropyCalibrator
{
public:
	Int8EntropyCalibrator(int BatchSize,const std::vector<std::vector<float>>& data,const std::string& CalibDataName = "",bool readCache = true);

	virtual ~Int8EntropyCalibrator();

	int getBatchSize() const override { return mBatchSize; }

	bool getBatch(void* bindings[], const char* names[], int nbBindings) override;

	const void* readCalibrationCache(size_t& length) override;

	void writeCalibrationCache(const void* cache, size_t length) override;

private:
	std::string mCalibDataName;
	std::vector<std::vector<float>> mDatas;
	int mBatchSize;

	int mCurBatchIdx;
	float* mCurBatchData{ nullptr };
	
	size_t mInputCount;
	bool mReadCache;
	void* mDeviceInput{ nullptr };

	std::vector<char> mCalibrationCache;
};

}	//namespace

#endif //_ENTROY_CALIBRATOR_H

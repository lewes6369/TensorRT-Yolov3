#ifndef _EVAL_H_
#define _EVAL_H_

#include <vector>
#include <list>
#include <string>
#include "dataReader.h"

namespace Tn
{
    float evalTopResult(std::list<std::vector<float>>& result,std::list<int>& groundTruth,int* Tp = nullptr,int* FP = nullptr,int topK = 1);
    float evalMAPResult(const std::list<std::vector<Bbox>>& bboxesList,const std::list<std::vector<Bbox>> & truthboxesList,int classNum,float iouThresh);
}

#endif
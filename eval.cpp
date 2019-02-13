#include "eval.h"
#include <algorithm>    
#include <assert.h>
#include <iostream>
#include <iomanip>

using namespace std;

namespace Tn
{
    float evalTopResult(list<vector<float>>& result,list<int>& groundTruth,int* TP /*= nullptr*/,int* FP /*= nullptr*/,int topK /*= 1*/)
    {
        int _TP = TP ? *TP: 0;
        int _FP = FP ? *FP: 0; 

        assert(result.size() == groundTruth.size());

        auto pRe = result.begin();
        auto pGT = groundTruth.begin();
        for (; pRe != result.end() && pGT != groundTruth.end();
            ++pRe, ++pGT)
        {
            auto& labels = *pRe;
            int truthClass = *pGT;
            float gtProb = labels[truthClass];

            int biggerCount = 0;
            for (auto& prob : labels)
            {
                if (prob >= gtProb)
                    ++biggerCount;
            }

            biggerCount > topK ? ++_FP : ++_TP;
        }

        float accuracy=float(_TP)/(_TP+_FP);
        if(TP) *TP =_TP;
        if(FP) *FP =_FP;

        cout<<"top " << topK <<" accuracy :"<< setprecision(4) << accuracy << endl;

        return accuracy;
    }

    float iou_compute(const Bbox& a,const Bbox& b)
    {
        int and_right=min(a.right,b.right);
        int and_left =max(a.left,b.left);
        int and_top  =max(a.top,b.top);
        int and_bot  =min(a.bot,b.bot);

        if ((and_top>and_bot) || (and_left>and_right))
        {
            return 0.0f;
        }
        float sand=(and_right-and_left)*(and_bot-and_top)*1.0f;
        float sa=(a.right-a.left)*(a.bot-a.top)*1.0f;
        float sb=(b.right-b.left)*(b.bot-b.top)*1.0f;

        float iou=sand/(sa+sb-sand);
        return iou;
    }

    float evalMAPResult(const list<vector<Bbox>>& bboxesList,const list<vector<Bbox>>& truthboxesList,int classNum,float iouThresh)
    {
        assert(bboxesList.size() == truthboxesList.size());
        cout << "evalMAPResult:" << endl;

        float* precision = new float[classNum];
        float* recall = new float[classNum];
        float* AP = new float[classNum];

        vector<Bbox> **detBox = nullptr;
        vector<Bbox> **truthBox = nullptr;

        int sampleCount = bboxesList.size();
        detBox = new vector<Bbox>* [sampleCount];
        truthBox = new vector<Bbox>* [sampleCount];
        for (int i = 0 ;i < sampleCount ; ++ i)
        {
            detBox[i] = new vector<Bbox>[classNum]{};
            truthBox[i] = new vector<Bbox>[classNum]{};
        }

        auto pBoxIter = bboxesList.begin();
        auto pTrueIter = truthboxesList.begin();
        for (int i = 0;i< sampleCount;++i , ++pBoxIter , ++pTrueIter)
        {
            for (const auto& item : *pBoxIter)
                detBox[i][item.classId].push_back(item);

            for (const auto& item : *pTrueIter)
                truthBox[i][item.classId].push_back(item);
        }

        for (int i = 0;i < classNum; ++ i)
        {
            using CheckPair = pair<Bbox,bool>;
            vector< CheckPair > checkPRBoxs;
            int FN = 0;
            for (int j = 0;j< sampleCount;++j)
            {   
                auto& dboxes = detBox[j][i];
                auto& tboxes = truthBox[j][i];

                auto checkTBoxes = tboxes;
                for (const auto& item: dboxes)
                {
                    int maxIdx = -1;
                    float maxIou = 0;
                    
                    for (const auto& tItem: checkTBoxes)
                    {
                        float iou=iou_compute(item,tItem);
                        //std::cout << "iou" << iou << std::endl;
                        if(iou > maxIou)
                        {
                            maxIdx = &tItem - &checkTBoxes[0];
                            maxIou = iou;
                        }
                    }

                    if(maxIou > iouThresh)
                    {
                        checkPRBoxs.push_back({item,true});
                        checkTBoxes.erase(checkTBoxes.begin() + maxIdx);
                    }
                    else
                    {
                        //FP 
                        checkPRBoxs.push_back({item,false});
                    }
                }
                //FN
                FN += checkTBoxes.size();
            }

            float TP = count_if(checkPRBoxs.begin(), checkPRBoxs.end(), [](CheckPair& item){return item.second == true;} );

            int total = checkPRBoxs.size();
            if(total == 0)
            {
                AP[i] = 1;
                continue;
            }

            //recall:         
            recall[i] = (abs(TP + FN) < 1e-5) ? 1 : TP / (TP + FN);
            //precision
            precision[i] = TP / total;

            //compute AP:
            sort(checkPRBoxs.begin(),checkPRBoxs.end(),[](CheckPair& left,CheckPair& right){
                return left.first.score > right.first.score;
                }
            );

            int PR_TP = 0;
            int PR_FP = 0;
            vector< pair<float,float> >  PRValues;  //<P,R>
            for (const auto& item : checkPRBoxs)
            {
                item.second ? ++PR_TP : ++PR_FP;
                PRValues.emplace_back( make_pair(PR_TP/ float(PR_TP+PR_FP) , PR_TP / float(total)) );
            }
            
            float sum = PRValues[0].first * PRValues[0].second;

            for (unsigned int m = 0; m < PRValues.size()-1;++m)
            {
                float w = PRValues[m + 1].second - PRValues[m].second ;
                float h = PRValues[m + 1].first;
                sum += w*h;
            }
            
            AP[i] = sum;

            cout<< setprecision(4) << "class:" << std::setw(3) << i 
                << " iou thresh-" << iouThresh 
                << " AP:" << std::setw(7) << AP[i] 
                << " recall:" << std::setw(7) << recall[i] 
                << " precision:" << std::setw(7) << precision[i] << endl;
        }

        float sumAp = 0;
        for (int i = 0;i < classNum;++i)
            sumAp += AP[i];
        
        float MAP = sumAp / classNum;
        cout<< "MAP:" << MAP << endl;
        
        if (precision)
            delete[] precision;
        if (recall)
            delete[] recall;
        if (AP)
            delete[] AP;
  
        for (int i = 0;i < sampleCount; ++i)
        {
            delete[] detBox[i];
            delete[] truthBox[i];
        }

        delete[] detBox;
        delete[] truthBox;

        return MAP;
    }
}
//
//  LIH.h
//  SmileDetect
//
//  Created by Chien Chin-yu on 2015/1/1.
//  Copyright (c) 2015年 Chien Chin-yu. All rights reserved.
//

#ifndef __SmileDetect__LIH__
#define __SmileDetect__LIH__

#include "opencv2/core/core.hpp"
#include "descriptor.h"

class LIH : public Descriptor{
    
public:
    LIH(int gridSize, int bitSize);
    ~LIH(){}
    
    void getFeature(const cv::Mat image, float** feature);
    int getVecSize();
    
private:
    int _gridSize = 8;
    int _bitSize = 8;
};

#endif /* defined(__SmileDetect__LIH__) */

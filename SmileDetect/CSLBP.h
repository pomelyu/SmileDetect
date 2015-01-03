//
//  CSLBP.h
//  SmileDetect
//
//  Created by Chien Chin-yu on 2015/1/1.
//  Copyright (c) 2015年 Chien Chin-yu. All rights reserved.
//

#ifndef __SmileDetect__CSLBP__
#define __SmileDetect__CSLBP__

#include "opencv2/core/core.hpp"
#include "descriptor.h"

class CSLBP : public Descriptor{

public:
    CSLBP(int gridSize);
    
    void getFeature(const cv::Mat image, float** feature);
    int getVecSize();

private:
    const int BIT_SIZE = 16;
    int _gridSize = 5;
};

#endif /* defined(__SmileDetect__CSLBP__) */

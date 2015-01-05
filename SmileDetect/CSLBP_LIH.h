//
//  CSLBP_LIH.h
//  SmileDetect
//
//  Created by Chien Chin-yu on 2015/1/5.
//  Copyright (c) 2015年 Chien Chin-yu. All rights reserved.
//

#ifndef __SmileDetect__CSLBP_LIH__
#define __SmileDetect__CSLBP_LIH__

#include "opencv2/core/core.hpp"
#include "descriptor.h"
#include "CSLBP.h"
#include "LIH.h"

class CSLBP_LIH : public Descriptor{
    
public:
    CSLBP_LIH(int CS_gridSize, int L_gridSize, int L_bitSize);
    ~CSLBP_LIH();
    
    void getFeature(const cv::Mat image, float** feature);
    int getVecSize();
    
private:
    CSLBP* cslbp;
    LIH* lih;
};

#endif /* defined(__SmileDetect__CSLBP_LIH__) */

//
//  descriptor.h
//  SmileDetect
//
//  Created by Chien Chin-yu on 2015/1/1.
//  Copyright (c) 2015年 Chien Chin-yu. All rights reserved.
//

#ifndef SmileDetect_descriptor_h
#define SmileDetect_descriptor_h

enum Feature{
    LIH_DESCRIPTOR,     // Local intensity histogram (LIH)
    CSLBP_DESCRIPTOR,   // Center-Symmetric Local Binary Pattern (CS-LBP)
};

class Descriptor{
public:
    virtual void getFeature(const cv::Mat image, float** feature){};
    virtual int getVecSize(){return 1;};
};

#endif

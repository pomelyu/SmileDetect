//
//  RT.h
//  SmileDetect
//
//  Created by Chien Chin-yu on 2015/1/3.
//  Copyright (c) 2015年 Chien Chin-yu. All rights reserved.
//

#ifndef __SmileDetect__RT__
#define __SmileDetect__RT__

#include <stdio.h>

#include <stdio.h>
#include "classifier.h"
#include "opencv2/ml/ml.hpp"

class RT : public Classifier{
    
public:
    void train();
    void crossvalidation();
    float predict();
    float predict(cv::Mat feature, cv::Mat label);
    
private:
    CvRTrees tree;
    CvRTParams params;
};

#endif /* defined(__SmileDetect__RT__) */

//
//  Boost.h
//  SmileDetect
//
//  Created by Chien Chin-yu on 2015/1/3.
//  Copyright (c) 2015年 Chien Chin-yu. All rights reserved.
//

#ifndef __SmileDetect__Boost__
#define __SmileDetect__Boost__

#include <stdio.h>
#include "classifier.h"
#include "opencv2/ml/ml.hpp"

class Boost : public Classifier{
    
public:
    Boost(){ params = boost.get_params(); }
    ~Boost(){};
    
    void train();
    void crossvalidation(float* parameter);
    float predict();
    float predict(cv::Mat feature, cv::Mat label);
    
private:
    CvBoost boost;
    CvBoostParams params;
};

#endif /* defined(__SmileDetect__Boost__) */

//
//  NN.h
//  SmileDetect
//
//  Created by CMLab on 1/3/15.
//  Copyright (c) 2015 Chien Chin-yu. All rights reserved.
//

#ifndef __SmileDetect__NN__
#define __SmileDetect__NN__

#include <stdio.h>
#include "classifier.h"
#include "opencv2/ml/ml.hpp"

class NN : public Classifier{
    
public:
    NN(){}
    
    void train(float* parameter);
    void crossvalidation(float* parameter);
    float predict();
    float predict(cv::Mat feature, cv::Mat label);
    
private:
    CvANN_MLP mlp;
    CvANN_MLP_TrainParams params;
};
#endif /* defined(__SmileDetect__NN__) */

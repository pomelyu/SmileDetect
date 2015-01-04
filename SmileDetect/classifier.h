//
//  classifier.h
//  SmileDetect
//
//  Created by Chien Chin-yu on 2015/1/3.
//  Copyright (c) 2015年 Chien Chin-yu. All rights reserved.
//

#ifndef SmileDetect_classifier_h
#define SmileDetect_classifier_h

#include <string>
#include "opencv2/highgui/highgui.hpp"

using std::string;

enum Method{
    LINEAR_SVM,
    ADABOOST,
    RANDOM_TREE,
    NETURAL_NETWORK
};

class Classifier{
public:
    virtual void train(){}
    virtual void crossvalidation(float* parameter){}
    virtual float predict(){ return 0; }
    virtual float predict(cv::Mat feature, cv::Mat label){ return 0; }
    
    void dataFromFile(string filePath);
    
protected:
    cv::Mat shuffleRows(const cv::Mat &matrix);
    cv::Mat data;
};

#endif

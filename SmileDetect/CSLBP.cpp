//
//  CSLBP.cpp
//  SmileDetect
//
//  Created by Chien Chin-yu on 2015/1/1.
//  Copyright (c) 2015年 Chien Chin-yu. All rights reserved.
//

#include "CSLBP.h"
#include "opencv2/opencv.hpp"
#include "math.h"

using namespace cv;

int sign(const int a){
    int result = (a < 0)? 0: 1;
    return result;
}


CSLBP::CSLBP(int gridSize){
    _gridSize = gridSize;
}


void
CSLBP::getFeature(const Mat image, float** feature){
    
    // == Resize image
    int RESIZE = 120;
    Mat resized = Mat::zeros(RESIZE, RESIZE, CV_8U);
    cv::resize(image, resized, cv::Size(RESIZE, RESIZE), INTER_CUBIC);
    
    // == Padding the image
    Mat padded = Mat::zeros(RESIZE+2, RESIZE+2, CV_8U);
    cv::copyMakeBorder(resized, padded, 1, 1, 1, 1, cv::BORDER_REPLICATE);
    resized.release();
    
    // == Initialize feature vector
    float* vec = new float[_gridSize * _gridSize * BIT_SIZE];
    for (int i = 0; i < _gridSize * _gridSize * BIT_SIZE; i++){
        vec[i] = 0;
    }
    
    namedWindow("test", CV_WINDOW_AUTOSIZE);
    imshow("test", padded);

    // == Extract Center-Symmetric Local Binary Pattern
    //
    //   n5 n6 n7
    //   n4 nc n0
    //   n3 n2 n1
    // s(n0-n4)2^0 + s(n1-n5)2^1 + s(n2-n6)2^2 + s(n3-n7)2^3
    int gridSpan = ceil((float)RESIZE / _gridSize);
    
    for (int j = 1; j < RESIZE+1; j++){
        for (int i = 1; i < RESIZE+1; i++) {
            int jj = (j-1) / gridSpan;
            int ii = (i-1) / gridSpan;
            
            Mat local = padded.colRange(i-1, i+1).rowRange(j-1, j+1);
            int kk = sign(local.at<uint8_t>(2,3) - local.at<uint8_t>(2,1))
                + (sign(local.at<uint8_t>(3,3) - local.at<uint8_t>(1,1)) << 1)
                + (sign(local.at<uint8_t>(3,2) - local.at<uint8_t>(1,2)) << 2)
                + (sign(local.at<uint8_t>(3,1) - local.at<uint8_t>(1,3)) << 3);
            
            vec[jj * _gridSize * BIT_SIZE + ii * BIT_SIZE + kk] += 1;
        }
    }

    *feature = vec;
}


int
CSLBP::getVecSize(){
    return _gridSize * _gridSize * BIT_SIZE;
}
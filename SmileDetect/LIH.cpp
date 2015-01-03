//
//  LIH.cpp
//  SmileDetect
//
//  Created by Chien Chin-yu on 2015/1/1.
//  Copyright (c) 2015年 Chien Chin-yu. All rights reserved.
//

#include "LIH.h"
#include "opencv2/opencv.hpp"
#include "math.h"

using namespace cv;

LIH::LIH(int gridSize, int bitSize){
    _gridSize = gridSize;
    _bitSize = bitSize;
}


void
LIH::getFeature(const Mat image, float** feature){
    
    // == Resize image
    int RESIZE = 120;
    Mat resized = Mat::zeros(RESIZE, RESIZE, CV_8U);
    cv::resize(image, resized, cv::Size(RESIZE, RESIZE), INTER_CUBIC);
    
    // == Initialize feature vector
    float* vec = new float[_gridSize * _gridSize * _bitSize];
    for (int i = 0; i < _gridSize * _gridSize * _bitSize; i++){
        vec[i] = 0;
    }
    

    // == Extract Local intensity histogram
    int gridSpan = ceil((float)RESIZE / _gridSize);
    int bitSpan = ceil(256.0 / _bitSize);

    for (int j = 0; j < resized.rows; j++){
        for (int i = 0; i < resized.cols; i++) {
            int ii = i / gridSpan;
            int jj = j / gridSpan;
            int kk = resized.at<uint8_t>(j, i) / bitSpan;
            
            vec[jj * _gridSize * _bitSize + ii * _bitSize + kk] += 1;
        }
    }

    
    // == Normalize the LIH in the cell
    // Normalize range from 0 to 1
    for (int j = 0; j < _gridSize; j++) {
        for (int i = 0; i < _gridSize; i++){
            int base = j * _gridSize * _bitSize + i * _gridSize;
            int max = 0;
            for (int k = 0; k < _bitSize; k++){
                int tmp = vec[base + k];
                if (tmp > max){
                    max = tmp;
                }
            }
            
            for (int k = 0; k < _bitSize; k++){
                vec[base + k] = (float)(vec[base + k]) / max;
            }
        }
    }
    
    *feature = vec;
}


int
LIH::getVecSize(){
    return _gridSize * _gridSize * _bitSize;
}


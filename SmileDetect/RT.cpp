//
//  RT.cpp
//  SmileDetect
//
//  Created by Chien Chin-yu on 2015/1/3.
//  Copyright (c) 2015年 Chien Chin-yu. All rights reserved.
//

#include "RT.h"

void
RT::train(){
    if (data.rows == 0){
        std::cerr << "Error, no data\n";
        return ;
    }

    int cols = data.cols;
    tree.train(data.colRange(0, cols), CV_ROW_SAMPLE, data.col(cols),
               cv::Mat(), cv::Mat(), cv::Mat(), cv::Mat(), params);
}


void
RT::crossvalidation(){
    if (data.rows == 0){
        std::cerr << "Error, no data\n";
        return;
    }
    
    cv::Mat shuffle = shuffleRows(data);
    
    // == Set up the boost params
    CvRTParams tmpParams(20, 10, 0, false, 10, 0, false, false, 5, false, 0);
    params = tmpParams;
    
    // == Prepare data for crossvalidation
    const int fold = 5;
    int CELL_SIZE = floor((float)shuffle.rows / fold);
    cv::Mat Cells[fold];
    for (int i = 0; i < fold; i++){
        Cells[i] = shuffle.rowRange( i*CELL_SIZE, (i+1)*CELL_SIZE );
    }
    
    // == cross validation
    float total = 0;
    for (int i = 0; i < fold; i++){
        cv::Mat oneFold = cv::Mat::zeros(0, shuffle.cols, CV_32F);
        
        for (int j = 0; j < fold; j++){
            if (j != i) {
                oneFold.push_back(Cells[j]);
            }
        }
        
        int cols = oneFold.cols;
        tree.train(data.colRange(0, cols), CV_ROW_SAMPLE, data.col(cols),
                   cv::Mat(), cv::Mat(), cv::Mat(), cv::Mat(), params);
        float accuracy = predict(Cells[i].colRange(1, cols), Cells[i].col(0));
        
        std::cout << "Fold " << i << ". Accuracy = " << accuracy << "\n";
        total += accuracy;
    }
    std::cout << "Total: " << total/fold << "\n";
}


float
RT::predict(){
    if (data.rows == 0){
        std::cerr << "Error, no data\n";
        return 0;
    }
    int cols = data.cols;
    return predict(data.colRange(1, cols), data.col(0));
}


float
RT::predict(cv::Mat feature, cv::Mat label){
    float right = 0;
    for (int i = 0; i < feature.rows; i++) {
        cv::Mat tmp = feature.row(i);
        if (tree.predict(tmp) == label.at<float>(i, 0)){
            right++;
        }
    }
    return (right / feature.rows);
}
//
//  NN.cpp
//  SmileDetect
//
//  Created by CMLab on 1/3/15.
//  Copyright (c) 2015 Chien Chin-yu. All rights reserved.
//

#include "NN.h"

void
NN::train(){
    if (data.rows == 0){
        std::cerr << "Error, no data\n";
        return ;
    }
    
    int cols = data.cols;
    mlp.train(data.colRange(0, cols), data.col(cols), cv::Mat(),
               cv::Mat(),params,0);
}


void
NN::crossvalidation(float* parameter){
    if (data.rows == 0){
        std::cerr << "Error, no data\n";
        return;
    }
    
    cv::Mat shuffle = shuffleRows(data);
    
    // == Set up the boost params
    CvANN_MLP_TrainParams tmpParams(
        cvTermCriteria( CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 0.01 ),
        CvANN_MLP_TrainParams::RPROP,
        0.1,0.5);
    
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
        
        cv::Mat layers = cv::Mat(3,1,CV_32SC1);
        layers.row(0) = cv::Scalar(oneFold.cols-1);
        layers.row(1) = cv::Scalar(20);
        layers.row(2) = cv::Scalar(1);
        mlp.create(layers);
        mlp.train(oneFold.colRange(1, cols),oneFold.col(0), cv::Mat(), cv::Mat(),params);
        
        float accuracy = predict(Cells[i].colRange(1, cols), Cells[i].col(0));
        
        std::cout << "Fold " << i << ". Accuracy = " << accuracy << "\n";
        total += accuracy;
    }
    std::cout << "Total: " << total/fold << "\n";
}


float
NN::predict(){
    if (data.rows == 0){
        std::cerr << "Error, no data\n";
        return 0;
    }
    int cols = data.cols;
    return predict(data.colRange(1, cols), data.col(0));
}


float
NN::predict(cv::Mat feature, cv::Mat label){
    float right = 0;
    cv::Mat output;
    mlp.predict(feature, output);
    for (int i = 0; i < label.rows; ++i) {
//        std::cout << label.at<float>(i,0) << " " << output.at<float>(i,0) << std::endl;
        if (label.at<float>(i,0) == 0 && output.at<float>(i,0) <= 0) {
            ++right;
        }
        else if(label.at<float>(i,0) == 1 && output.at<float>(i,0) > 0){
            ++right;
        }
    }
    return (right / feature.rows);
}
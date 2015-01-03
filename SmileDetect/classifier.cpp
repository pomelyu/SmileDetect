//
//  classifier.cpp
//  SmileDetect
//
//  Created by Chien Chin-yu on 2015/1/3.
//  Copyright (c) 2015年 Chien Chin-yu. All rights reserved.
//

#include "classifier.h"
#include <fstream>
#include <vector>

using std::vector;

void
Classifier::dataFromFile(string filePath){
    std::fstream file;
    file.open(filePath, std::ios::in);
    
    data = cv::Mat::zeros(0, 1, CV_32F);
    char line[1000000];
    
    // get first line to get the dims
    if (file.getline(line, sizeof(line))){
        char tmpline[1000000];
        memcpy(tmpline, line, strlen(line));
        char* seg;
        seg = strtok(tmpline, " ");
        cv::Mat tmp = cv::Mat::zeros(1, 1, CV_32F);
        tmp.at<float>(0, 0) = atof(seg);
        data.push_back(tmp);
        
        seg = strtok(NULL, ":");
        seg = strtok(NULL, " ");
        while (seg != NULL) {
            tmp = cv::Mat::zeros(1, 1, CV_32F);
            tmp.at<float>(0, 0) = atof(seg);
            data.push_back(tmp);
            
            seg = strtok(NULL, ":");
            seg = strtok(NULL, " ");
        }
    }
    
    cv::transpose(data, data);
    int dims = data.cols;

    
    // the other line
    while (file.getline(line, sizeof(line))) {
        char tmpline[1000000];
        memcpy(tmpline, line, strlen(line));
        char* seg = strtok(tmpline, " ");
        
        cv::Mat oneLine = cv::Mat::zeros(1, dims, CV_32F);
        oneLine.at<float>(0, 0) = atof(seg);
        
        int idx = 1;
        seg = strtok(NULL, ":");
        seg = strtok(NULL, " ");
        while (seg != NULL) {
            oneLine.at<float>(0, idx) = atof(seg);
            idx++;
            seg = strtok(NULL, ":");
            seg = strtok(NULL, " ");
        }
        data.push_back(oneLine);
    }
}

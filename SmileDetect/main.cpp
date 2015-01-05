//
//  main.cpp
//  SmaileDetect
//
//  Created by Chien Chin-yu on 2015/1/1.
//  Copyright (c) 2015年 Chien Chin-yu. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <string>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "descriptor.h"
#include "CSLBP.h"
#include "CSLBP_LIH.h"
#include "LIH.h"
#include "classifier.h"
#include "Boost.h"
#include "RT.h"
#include "NN.h"

using std::cout;
using std::cerr;
using std::endl;
using std::string;

enum Mode{
    EXTRACT,
    TRAIN,
    TEST
};

int exist(string name);
void parseArg(Mode& mode,
              Descriptor** descriptor,
              Classifier** classifier,
              string& imagePath,
              string& filePath,
              string& dataPath,
              string& label,
              int argc,
              const char* argv[]
              );

void showArg(){
    cout
    << " == Parameter of the Smile Destect == "                         << "\n"
    << " [-m] : mode"                                                   << "\n"
    << "        [0] Extract feature (need -d, -l, -f, -i)"              << "\n"
    << "        [1] Training (need -c, -F)"                             << "\n"
    << "        [2] Classify (need -c, -f, -F)"                         << "\n"
    << " [-d] : Choose the descriptor"                                  << "\n"
    << "        [0] : [LIH]   Local intensity histogram"                << "\n"
    << "        [1] : [CSLBP] Center-Symmetric Local Binary Pattern"    << "\n"
    << "        [2] : [CSLBP + LIH]"                                    << "\n"
    << " [-l] : Label of the data"                                      << "\n"
    << "        [0] : For non-smile"                                    << "\n"
    << "        [1] : For smile"                                        << "\n"
    << " [-f] : The file to record the descriptor"                      << "\n"
    << "          ex. Data/LIH.txt"                                     << "\n"
    << " [-i] : The image folder"                                       << "\n"
    << "          ex. Data/Smile/%1d.jpg"                               << "\n"
    << " [-c] : Choose the classifier"                                  << "\n"
    << "      : [0] LinearSVM"                                          << "\n"
    << "      : [1] AdaBoost"                                           << "\n"
    << "      : [2] Random Forest"                                      << "\n"
    << "      : [3] Netural Network"                                    << "\n"
    << " [-F] : The data file to classify"                              << "\n"
    << "          ex. Data/LIH.txt"                                     << "\n"
    ;
}


int main(int argc, const char * argv[]) {
    
    Mode mode;
    Descriptor* descriptor = NULL;
    Classifier* classifier = NULL;
    string imagePath = "";
    string filePath = "";
    string dataPath = "";
    string label = "";
    
    parseArg(mode, &descriptor, & classifier,
             imagePath, filePath, dataPath, label, argc, argv);
    
    switch (mode) {
        case EXTRACT:
        {
            // == Open the data file
            std::fstream file;
            if (!filePath.empty()){
                if (exist(filePath)){
                    cout << "File existed!. Append new data to the file.\n";
                    file.open(filePath, std::ios::app | std::ios::out);
                }
                else{
                    cout << "File not existed!. Create new file\n";
                    file.open(filePath, std::ios::out);
                }
                if (!file.is_open()){
                    cerr << "Fail to open the file\n" << endl;
                }
            }
            
            // == Read image sequency
            if (!imagePath.empty()){
                cv::VideoCapture imgSeq(imagePath);
                cv::Mat img;
                
                if (!imgSeq.isOpened())
                    cerr << "Image could not be opened\n";
                
                int idx = 0;
                while (true) {
                    imgSeq >> img;
                    if (img.empty()){
                        cout << "Image extraction done\n";
                        break;
                    }
                    else{
                        idx++;
                        cout << "Process Image: " << idx << endl;
                        cv::Mat grayImg;
                        cv::cvtColor(img, grayImg, CV_BGR2GRAY);
                        
                        // == Extract feature and save to File
                        float* feature;
                        descriptor->getFeature(grayImg, &feature);
                        
                        file << label;
                        for (int i = 0; i < descriptor->getVecSize(); i++){
                            file << " " << i+1 << ":" << feature[i];
                        }
                        file << endl;
                        
                        delete[] feature;
                    }
                }
                imgSeq.release();
            }
            break;
        }
            
        case TRAIN:
        {
            classifier->dataFromFile(dataPath);
            float params[2] = {10, 5};
            classifier->crossvalidation(params);
//            for (int i = 1; i <= 20; i++) {
//                params[1] = i;
//                classifier->crossvalidation(params);
//            }
            break;
        }
            
        case TEST:
        {
            classifier->dataFromFile(filePath);
            float params[2] = {9, 10};
            classifier->train(params);
            cout << "Training Fin." << endl;
            classifier->dataFromFile(dataPath);
            float result = classifier->predict();
            cout << result << endl;
            break;
        }
            
        default:
            break;
    }
    
    delete descriptor;
    delete classifier;
    return 0;
}


int exist(string name){
    struct stat buffer;
    char* tmp = new char[name.length()+1];
    std::strcpy(tmp, name.c_str());
    delete[] tmp;
    return (stat(tmp, &buffer) == 0);
}


void parseArg(Mode& mode,
              Descriptor** descriptor,
              Classifier** classifier,
              string& imagePath,
              string& filePath,
              string& dataPath,
              string& label,
              int argc,
              const char* argv[]
              ){
    
    if (argc < 3){
        showArg();
        return;
    }
    
    // == parse the argument
    int argIdx = 1;
    for (; ; ) {
        if (argIdx >= argc)
            break;
        if (argv[argIdx][0] == '-'){
            char c = argv[argIdx][1];
            argIdx++;
            switch (c) {
                // == Set up mode
                case 'm':
                    mode = (Mode)atoi(argv[argIdx]);
                    break;
                    
                // == Set up descriptor
                case 'd':
                    switch (atoi(argv[argIdx])) {
                        case LIH_DESCRIPTOR:
                            *descriptor = new LIH(8, 8);
                            break;
                        case CSLBP_DESCRIPTOR:
                            *descriptor = new CSLBP(5);
                            break;
                        case CSLBP_LIH_DESCRIPTOR:
                            *descriptor = new CSLBP_LIH(5, 8, 8);
                        default:
                            break;
                    }
                    break;
                    
                // == Set up label
                case 'l':
                    label = argv[argIdx];
                    break;
                    
                // == Set up File path
                case 'f':
                    filePath = argv[argIdx];
                    break;
                    
                // == Set up Data path
                case 'F':
                    dataPath = argv[argIdx];
                    break;
                    
                // == Set up Image path
                case 'i':
                    imagePath = argv[argIdx];
                    break;
                
                // == Set up classifier
                case 'c':
                    switch (atoi(argv[argIdx])) {
                        case LINEAR_SVM:
                            *classifier = NULL;
                            break;
                        case ADABOOST:
                            *classifier = new Boost();
                            break;
                        case RANDOM_TREE:
                            *classifier = new RT();
                            break;
                        case NETURAL_NETWORK:
                            *classifier = new NN();
                            break;
                        default:
                            break;
                    }
                    break;
                default:
                    cerr << "Unkonwn argument: "<< c << "\n";
                    break;
            }
        }
        argIdx++;
    }
}
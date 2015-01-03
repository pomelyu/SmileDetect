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
#include "LIH.h"

using std::cout;
using std::cerr;
using std::endl;
using std::string;

int exist(string name);
void parseArg(Descriptor** descriptor,
              string& imagePath,
              string& filePath,
              string& label,
              int argc,
              char* argv[]
              );

int main(int argc, const char * argv[]) {
    
    bool isTest = true;
    
    Descriptor* descriptor = NULL;
    string imagePath = "";
    string filePath = "";
    string label = "";
    
    if (isTest) {
        descriptor = new CSLBP(5);
        imagePath = "Data/NonSmile/%1d.jpg";
        filePath = "Data/CSLBP.txt";
        label = "0";
    }
    else{
        void parseArg(Descriptor** descriptor,
                      string& imagePath,
                      string& filePath,
                      string& label,
                      int argc,
                      char* argv[]
                      );
    }
    
    
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
                    file << " " << i << ":" << feature[i];
                }
                file << endl;
                
                delete feature;
            }
        }
    }
    delete descriptor;
    return 0;
}


int exist(string name){
    struct stat buffer;
    char* tmp = new char[name.length()+1];
    std::strcpy(tmp, name.c_str());
    delete tmp;
    return (stat(tmp, &buffer) == 0);
}


void parseArg(Descriptor** descriptor,
              string& imagePath,
              string& filePath,
              string& label,
              int argc,
              char* argv[]
              ){
    
    if (argc < 3){
        // == Display the parameter
        cout << " == Parameter of the Smile Destect == \n"
        << " [-d] : Choose the descriptor\n"
        << "        [0] : [LIH]   Local intensity histogram\n"
        << "        [1] : [CSLBP] Center-Symmetric Local Binary Pattern\n"
        << " [-l] : Label of the data\n"
        << "        [0] : For non-smile\n"
        << "        [1] : For smile\n"
        << " [-f] : The file to record the descriptor\n"
        << "          ex. Data/LIH.txt\n"
        << " [-i] : The image folder\n"
        << "          ex. Data/Smile/%1d.jpg";
        return ;
    }
    
    // == parse the argument
    int argIdx = 1;
    for (; ; ) {
        if (argIdx >= argc)
            break;
        if (argv[argIdx][0] != '-'){
            argIdx++;
            continue;
        }
        else{
            char c = argv[argIdx][1];
            switch (c) {
                    // == Set up descriptor
                case 'd':
                    argIdx++;
                    switch (atoi(argv[argIdx])) {
                        case LIH_DESCRIPTOR:
                            *descriptor = new LIH(8, 8);
                            break;
                        case CSLBP_DESCRIPTOR:
                            *descriptor = new CSLBP(5);
                            break;
                        default:
                            break;
                    }
                    break;
                    
                    // == Set up label
                case 'l':
                    argIdx++;
                    label = argv[argIdx];
                    break;
                    
                    // == Set up File path
                case 'f':
                    argIdx++;
                    filePath = argv[argIdx];
                    break;
                    
                    // == Set up Image path
                case 'i':
                    argIdx++;
                    imagePath = argv[argIdx];
                    break;
                    
                default:
                    cerr << "Unkonwn argument\n";
                    break;
            }
            argIdx++;
        }
    }
}
//
//  CSLBP_LIH.cpp
//  SmileDetect
//
//  Created by Chien Chin-yu on 2015/1/5.
//  Copyright (c) 2015年 Chien Chin-yu. All rights reserved.
//

#include "CSLBP_LIH.h"

CSLBP_LIH::CSLBP_LIH(int CS_gridSize, int L_gridSize, int L_bitSize){
    cslbp = new CSLBP(CS_gridSize);
    lih = new LIH(L_gridSize, L_bitSize);
}

CSLBP_LIH::~CSLBP_LIH(){
    cslbp->~CSLBP();
    lih->~LIH();
    delete cslbp;
    delete lih;
}

void
CSLBP_LIH::getFeature(const cv::Mat image, float** feature){
    int size_cslbp = cslbp->getVecSize();
    int size_lih = lih->getVecSize();
    
    float* vec = new float[size_cslbp + size_lih];
    
    for (int i = 0; i < size_lih + size_cslbp; i++){
        vec[i] = 0;
    }
    
    float* cslbp_feature;
    cslbp->getFeature(image, &cslbp_feature);
    memcpy(vec, cslbp_feature, sizeof(float) * size_cslbp);
    delete[] cslbp_feature;
    
    float* lih_feature;
    lih->getFeature(image, &lih_feature);
    memcpy(&vec[size_cslbp], lih_feature, sizeof(float) * size_lih);
    delete[] lih_feature;
    
    *feature = vec;
}


int
CSLBP_LIH::getVecSize(){
    return (cslbp->getVecSize() + lih->getVecSize());
}
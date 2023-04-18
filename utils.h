//==================================================================================
// BSD 2-Clause License
//
// Copyright (c) 2023, Duality Technologies Inc.
//
// All rights reserved.
//
// Author TPOC: contact@openfhe.org
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//==================================================================================

#ifndef DPRIVE_ML__UTILS_H_
#define DPRIVE_ML__UTILS_H_

#include <climits>
#include "openfhe.h"
#include <limits>
#include "lr_types.h"
#include "lr_train_funcs.h"
#include "utils.h"
#include "pt_matrix.h"
#include "enc_matrix.h"
#include "parameters.h"


////////// Misc support Function declarations related to logistic regression training on encrypted data ///////////////////////////////


///////////////////////////////////////////////////////////////////
// support functions added by DBC


//////////////////////////////////////////////////
// returns next power of 2 >= x
usint NextPow2(const usint x);

//////////////////////////////////////////////////
// returns true if x == power of 2
bool IsPow2(usint x);

// function to generate power of two columnSize and rowSize to fit into the numSlots of the ciphertext
// for matrix MAT_ROW_MAJOR MAT_COL_MAJOR packing.
// returns std::pair<columnSize, rowSize>
std::pair<usint, usint> ComputePaddedDimensions(const usint numRows, const usint numCols, const usint numSlots);

////////////////////////////////////////////////////////
//not sure where these functions will end up
//
//!todo clean up do we need single vector in Mat_row_major or mat_col_major? arent they the same thing for a single vector?
// converts Mat input (vector of vectors representation) to single vector MAT_ROW_MAJOR representation.
Vec Mat2MatRowMajorVec(const Mat &inMat);

// converts Mat input (vector of vectors representation) of a column or row vector to single vector
Vec OneDMat2Vec(const Mat &inMat);

///////////////////////////////////////////////////////////
// encode and encrypt a Mat into Ciphertext in MAT_ROW_MAJOR format with zero padding
// note these functions DO apply zero padding
CT Mat2CtMRM(CC &cc, const Mat &inMat, const int rowSize, const int numSlots, const KeyPair &keys);

///////////////////////////////////////////////////////////
//  encode and encrypt a One Dimensional Mat into Ciphertext in VEC_COL_CLONED format
// zero padded out to rowSize, the power of 2 dimension, then cloned to
// fill out numSlots
CT OneDMat2CtVCC(CC &cc, const Mat &inMat, const int rowSize, const int numSlots, const KeyPair &keys);

///////////////////////////////////////////////////////////

CT collateOneDMats2CtVRC(CC &cc, const Mat &inMat, const Mat &inMat2, const int colSize, const int numSlots, const KeyPair &keys);

///////////////////////////////////////////////////////////////
// Prints out Vector VectorRowCloned
void PrintVecRowCloned(const Vec &x, const int rowSize);

///////////////////////////////////////////////////////////////
// Prints out Vector VectorColCloned
void PrintVecColCloned(const Vec &x, const int rowSize);

///////////////////////////////////////////////////////////////
// Prints out a vector stored in MatRowMajor
void PrintMatRowMajor(const Vec &z, const int rowSize);

template<typename T>
void SimplePrintVec(const std::string prefixMsg, const T &vec) {
  std::cout << prefixMsg;
  for (auto &el : vec) {
    std::cout << el << ",";
  }
  std::cout << std::endl;
}

void populateData(
    Parameters &params,
    CC &cc,
    KeyPair &keys,
    Mat &NegXt,
    Mat &beta,
    Mat &X,
    Mat &y,
    Mat &testX,
    Mat &testY,
    PT &ptExtractThetaMask,
    PT &ptExtractPhiMask,
    float lrGamma
);

#endif //DPRIVE_ML__UTILS_H_

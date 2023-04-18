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

#ifndef DPRIVE_ML__LR_TRAIN_FUNCS_H_
#define DPRIVE_ML__LR_TRAIN_FUNCS_H_

#include "lr_types.h"
#include "openfhe.h"

////////// Function declarations related to logistic regression training on encrypted data ///////////////////////////////

/* Takes plaintext training data matrix X, label vector y and initial vector b input
// and initializes logistic regression training values and parameters.
 *
 * In particular:
 * - It initializes the model weight vector beta to 0, which is going to be the vector we learn.
 *
 * X, y,  beta need to be allocated outside the function.
 * @param TBD
 * @return
 */

Mat InitializeLogReg(Mat &X, Mat &y, float scalingFactor = 1.0);

/**
 * Calculate the lr-scaled gradient. Based on the log-likelihood
 * @param cc                Cryptocontext
 * @param ctX               Features
 * @param ctNegXt           -features transposed
 * @param ctLabels          features
 * @param ctThetas           weights
 * @param ctGradStoreInto        gradients
 * @param lr                learning rate
 * @param colSize           num cols for gradient scaling
 * @param rowSize           length of row of fppe matrix
 * @param origNumSamples    Number of samples
 * @param rowKeys           keys for row operations
 * @param colKeys           keys for col operations
 * @param keys              keys for enc/dec
 * @param withBT            whether to run bootstrapping
 */
void EncLogRegCalculateGradient(
    CC &cc,
    const CT &ctX,
    const CT &ctNegXt,
    const CT &ctLabels,
    CT &ctThetas,
    CT &ctGradStoreInto,
    usint rowSize,
    const MatKeys &rowKeys,
    const MatKeys &colKeys,
    const KeyPair &keys,
    bool debug=false,
    int chebRangeStart = -64,
    int chebRangeEnd = 64,
    int chebPolyDegree = 128,
    int debugPlaintextLength=32
    );

///////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////
// checks to see if all elements of the Mat inMat are within abs(x)< bound --> throws otherwise
void BoundCheckMat(const Mat &inMat, const double bound);

///////////////////////////////////////////////////////////////
// Re-encrcypt cipher text ctx
PT ReEncrypt(CC &cc, CT &ctx, const KeyPair &keys);

///////////////////////////////////////////////////////////////
// Returns the current Depth of CT
int ReturnDepth(const CT &ct);

///////////////////////////////////////////////////////////////
// compute loss function
// Formulation based off of: https://stackoverflow.com/a/47798689/18031872
double ComputeLoss(const Mat &betas, const Mat &X, const Mat &y);

#endif //DPRIVE_ML__LR_TRAIN_FUNCS_H_

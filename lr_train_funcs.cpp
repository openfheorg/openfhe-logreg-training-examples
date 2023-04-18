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


#include "lr_train_funcs.h"
#include "pt_matrix.h"
#include "utils/debug.h"
#include "enc_matrix.h"
#include "math.h"

////////////////////////////////////////////////////////////////////////////
// Observe that if we pass in the scalingFactor (e.g lr / numRows) we can save on a multiplication
Mat InitializeLogReg(Mat &X, Mat &y, float scalingFactor) {
  /////////////////////////////////////////
  // update this for our problem
  /////////////////////////////////////////

  if (X.size() <= 0) {
    std::cerr << "Please provide a data matrix with positive number of rows." << std::endl;
    exit(0);
  }

#ifdef ENABLE_DEBUG
  std::cerr << "Initialization - Input data X (showing only 5 rows): " << std::endl;
  Mat Xsub = Mat(X.begin(), X.begin()+5);
  PrintMatrix(Xsub);
  std::cerr << std::endl;
#endif // ENABLE_DEBUG

  // Compute X transpose
  //note X tranpose is the same CT packing as x Just labeled differntly since
  // X mat_col_major == X' mat_row_major
  //copy XT = X
  Mat XT = Mat(X.begin(), X.end());

  // take negative of XT
  MatrixScalarMult(XT, -1.0 * scalingFactor);

#ifdef ENABLE_DEBUG
  std::cerr << "Initialization - X transpose (showing only 5 rows, 5 columns): " << std::endl;
  Mat XTsub = Mat(5);
  for (usint i=0; i<XTsub.size(); i++)
    XTsub[i] = Vec(XT[i].begin(), XT[i].begin()+5);
  PrintMatrix(XTsub);
  std::cerr << std::endl;
#endif // ENABLE_DEBUG
  return (XT);
}

///////////////////////////////////////////////////////////////////////////////////////
void EncLogRegCalculateGradient(
    CC &cc,
    const CT &ctX,
    const CT &ctNegXt,
    const CT &ctLabels,
    CT &ctThetas,
    CT &ctGradStoreInto,
    const usint rowSize,
    const MatKeys &rowKeys,
    const MatKeys &colKeys,
    const KeyPair &keys,
    bool debug,
    int chebRangeStart,
    int chebRangeEnd,
    int chebPolyDegree,
    int debugPlaintextLength
) {
  OPENFHE_DEBUG_FLAG(false);
  // We use the same notation as in
  //    https://eprint.iacr.org/2018/662.pdf
  //    It seems like their labels are {-1, 1} which we do not use. Change accordingly
  CT ctLogits;
  PT dbg;

  if (debug) {
    cc->Decrypt(keys.secretKey, ctThetas, &dbg);
    dbg->SetLength(debugPlaintextLength);
    std::cout << "\tDEBUG: Thetas: " << dbg;
    cc->Decrypt(keys.secretKey, ctX, &dbg);
    dbg->SetLength(debugPlaintextLength);
    std::cout << "\tDEBUG: Xs: " << dbg;
  }

  // Line 4
  MatrixVectorProductRow(cc, keys, colKeys, ctX, ctThetas, rowSize, ctLogits);
  if (debug) {
    cc->Decrypt(keys.secretKey, ctLogits, &dbg);
    dbg->SetLength(debugPlaintextLength);
    std::cout << "\tLogits: " << dbg;
    std::cout << "\tLogits level: " << ctLogits->GetLevel() << "\n" << std::endl;
  }

  // Line 5/6
  auto preds = cc->EvalLogistic(ctLogits, chebRangeStart, chebRangeEnd, chebPolyDegree);
  if (debug) {
    cc->Decrypt(keys.secretKey, preds, &dbg);
    dbg->SetLength(debugPlaintextLength);
    std::cout << "\tPreds " << dbg;
    std::cout << "\tPreds level (post sigmoid): " << preds->GetLevel() << "\n" << std::endl;
  }

  // Line 8 - see Page 9 for their notation
  OPENFHE_DEBUG("\tPre-Residual");
  auto residual = cc->EvalSub(ctLabels, preds);

  if (debug) {
    cc->Decrypt(keys.secretKey, residual, &dbg);
    dbg->SetLength(debugPlaintextLength);
    std::cout << "\tResiduals " << dbg;
    std::cout << "\tResidual level: " << residual->GetLevel() << "\n" << std::endl;
  }

  MatrixVectorProductCol(cc, rowKeys, ctNegXt, residual, rowSize, ctGradStoreInto);

  if (debug) {
    cc->Decrypt(keys.secretKey, ctGradStoreInto, &dbg);
    dbg->SetLength(debugPlaintextLength);
    std::cout << "\tScaled gradients: " << dbg;
      std::cout << "\tctGrad store into level: " << ctGradStoreInto->GetLevel() << "\n" << std::endl;
  }

}

///////////////////////////////////////////////////////////////
void BoundCheckMat(const Mat &inMat, const double bound) {

  usint numRows = inMat.size();
  usint numCols = inMat[0].size();

  //yes this is slow...
  for (usint i = 0; i < numRows; i++) {
    for (usint j = 0; j < numCols; j++) {
      if (abs((int) inMat[i][j]) >= (int) bound) {
        std::cout << "element at [" << i << "," << j << "] is " << inMat[i][j] << " bounds " << bound << std::endl;
      }
    }
  }
}

////////////////////////////////const//////////////////////////////
PT ReEncrypt(CC &cc, CT &ctx, const KeyPair &keys) {

  OPENFHE_DEBUG_FLAG(false);
  OPENFHE_DEBUG("In ReEncrypt");
  // reencrypt x
  PT xPT;
  OPENFHE_DEBUG("Decrypt");
  cc->Decrypt(keys.secretKey, ctx, &xPT);

  Vec x = xPT->GetRealPackedValue();

  xPT = cc->MakeCKKSPackedPlaintext(x);

  OPENFHE_DEBUG("Encrypt() ");
  ctx = cc->Encrypt(keys.publicKey, xPT);
  return xPT; //return this for debug purposes...
}

int ReturnDepth(const CT &ct) {
  auto mulDepth = ct->GetElements()[0].GetNumOfElements() - 1;
  auto scaling = ct->GetScalingFactor();
  std::cout << "mult Depth: " << mulDepth << " Scaling: " << scaling << std::endl;
  return (mulDepth);
}

double ComputeLoss(const Mat &b, const Mat &X, const Mat &y) {
  // Based off of https://stackoverflow.com/a/47798689/18031872
  OPENFHE_DEBUG_FLAG(false);
  OPENFHE_DEBUG("In ComputeLoss");
  usint numSamp = X.size();     //n_samp

  /////////////////////////////////////////////////////////////////
  //Calculate t1: matmul(-y.T, log(yHat)
  /////////////////////////////////////////////////////////////////
  //yHat = sigmoid(X * beta);
  Mat yHat = Mat(numSamp, Vec(1, 0.0));
  MatrixMult(X, b, yHat);
  MatrixSigmoid(yHat);
  // log(yHat)
  Mat logYHat = Mat(numSamp, Vec(1, 0.0));
  MatrixLog(yHat, logYHat);

  Mat yT = Mat(y[0].size(), Vec(y.size(), 0.0));
  MatrixTransp(y, yT);
  MatrixScalarMult(yT, -1);
  Mat t1Mat = Mat(1, Vec(1, 0.0));
  MatrixMult(yT, logYHat, t1Mat);
  //PrintMatrix(t1Mat);

  /////////////////////////////////////////////////////////////////
  //t2: matmult(
  //    t2_a,
  //    t2_b
  //    )
  // t2_a = 1 - y.T
  // t2_b = log(1 - yHat)
  /////////////////////////////////////////////////////////////////
  // from earlier it exists as -yT. We change it back here
  // so we can do a sub. Less confusing for newer readers
  Mat t2Mat_a = Mat(yT.size(), Vec(yT[0].size(), 0.0));
  MatrixScalarMult(yT, -1);
  // Getting t2_a
  ScalarSubMat(1, yT, t2Mat_a);
  OPENFHE_DEBUG("Got t2_a: 1-yT");

  Mat t2Mat_b = Mat(y.size(), Vec(1, 0.0));
  ScalarSubMat(1, yHat, t2Mat_b);
  MatrixLog(t2Mat_b, t2Mat_b);
  OPENFHE_DEBUG("Got t2_b: log(1-yHat)");

  Mat t2Mat = Mat(1, Vec(1, 0.0));
  MatrixMult(t2Mat_a, t2Mat_b, t2Mat);

  // Should now have a Mat Scalar that we add up
  Mat loglikelihood = Mat(1, Vec(1, 0.0));
  MatrixMatrixSub(t1Mat, t2Mat, loglikelihood);
  return loglikelihood[0][0] / double(numSamp);
}

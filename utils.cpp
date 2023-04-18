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

///////////////////////////////////////////////////////////////////
// support functions added by DBC
#include "utils.h"
#include "utils/debug.h"
#include "parameters.h"

//////////////////////////////////////////////////
usint NextPow2(const usint x) {
  return pow(2, ceil(log(double(x)) / log(2.0)));
};

//////////////////////////////////////////////////
bool IsPow2(usint x) {
  if (x > 0) {
    while (x % 2 == 0) {
      x /= 2;
    }
    if (x == 1) {
      return true;
    }
  }
  if (x == 0 || x != 1) {
    return false;
  }
  return false;
}

/////////////////////////////////
std::pair<usint, usint> ComputePaddedDimensions(const usint numRows, const usint numCols, const usint numSlots) {
    auto rowSize = NextPow2(numCols);
    auto colSize = numSlots/rowSize;
    return std::make_pair(colSize, rowSize);
}

/////////////////////////////////
Vec Mat2MatRowMajorVec(const Mat &inMat) {
  //matrix row major { row 0, row 1, etc}
  //verified
  usint numRows = inMat.size();
  usint numCols = inMat[0].size();
  OPENFHE_DEBUG_FLAG(false);
  OPENFHE_DEBUGEXP(numRows);
  OPENFHE_DEBUGEXP(numCols);
  Vec outVec;
  //yes this is slow...
  for (usint i = 0; i < numRows; i++) {
    for (usint j = 0; j < numCols; j++) {
      outVec.push_back(inMat[i][j]);
      OPENFHE_DEBUGEXP(inMat[i][j]);
    }
  }
  return outVec;
}

/////////////////////////////////
Vec OneDMat2Vec(const Mat &inMat) {
  //matrix row major { row 0, row 1, etc}

  OPENFHE_DEBUG_FLAG(false);
  usint numRows = inMat.size();
  usint numCols = inMat[0].size();

  OPENFHE_DEBUGEXP(numRows);
  OPENFHE_DEBUGEXP(numCols);
  OPENFHE_DEBUG("in OneDMat2Vec");
  if (dbg_flag) {
    PrintMatrix(inMat);
  }

  if ((numCols != 1) && (numRows != 1))
    OPENFHE_THROW(lbcrypto::config_error,
                   __FILE__ + std::string(" ") +
                       __FUNCTION__ + std::string(":") +
                       std::to_string(__LINE__) +
                       std::string("Error: input Mat is not a row or column vector"));

  //This function will unravel a Mat with a single row or column into a single vector
  //no zero padding
  Vec outVec = Mat2MatRowMajorVec(inMat);
  OPENFHE_DEBUGEXP(outVec.size());
  return outVec;
}

///////////////////////////////////////////////////////////
CT OneDMat2CtVCC(CC &cc, const Mat &inMat, const int rowSize, const int numSlots, const KeyPair &keys) {
  //verifired
  OPENFHE_DEBUG_FLAG(false);
  OPENFHE_DEBUG("in OneDMat2CtVCC");
  // input is currently a Mat (row vector) use Mat2Vec to make it a vector
  auto inVec = OneDMat2Vec(inMat);

  int origLargestSize = inVec.size();
  OPENFHE_DEBUGEXP(origLargestSize);
  auto colSize = numSlots / rowSize;

  OPENFHE_DEBUGEXP(rowSize);
  OPENFHE_DEBUGEXP(colSize);
  if (origLargestSize > colSize) {
    OPENFHE_THROW(lbcrypto::config_error, __FILE__ + std::string(" ") + __FUNCTION__ + std::string(":") +
        std::to_string(__LINE__) +
        std::string("Error: input vector largest dimension exceeds colSize"));
  }
  // zero pad out to colSize
  for (auto i = origLargestSize; i < colSize; i++) {
    inVec.push_back(0.0);
  }
  OPENFHE_DEBUG("after zero pad");
  if (!IsPow2(inVec.size())) {
    OPENFHE_THROW(lbcrypto::config_error, __FILE__ + std::string(" ") + __FUNCTION__ + std::string(":") +
        std::to_string(__LINE__) +
        std::string("Error: input vector non power of two"));
  }

  // inVecCC is VEC_COL_CLONED (and zero padded)
  Vec inVecCC;
  GetVecColCloned<double>(inVec, numSlots, 0.0, inVecCC);
  OPENFHE_DEBUG("after GetVecColCloned");
  if (dbg_flag) {
    PrintVecColCloned(inVecCC, colSize);
  }
  // make plaintext
  PT inVecCCPT = cc->MakeCKKSPackedPlaintext(inVecCC); // encode cloned vector
  //encrypt
  CT ctin = cc->Encrypt(keys.publicKey, inVecCCPT);
  return ctin;
}

//CT OneDMat2CtVRC(CC &cc, const Mat &inMat, const int rowSize, const int numSlots, const KeyPair &keys) {
Vec cloneVecRc(const Mat &inMat, const int rowSize, const int numSlots){

  //verifired
  OPENFHE_DEBUG_FLAG(false);

  OPENFHE_DEBUG("in OneDMat2CtVRC");
  // input is currently a Mat s use Mat2Vec to make it a vector
  auto inVec = OneDMat2Vec(inMat);
  int origNumRow = inVec.size();
  OPENFHE_DEBUGEXP(origNumRow);
  auto colSize = numSlots / rowSize;
  OPENFHE_DEBUGEXP(rowSize);
  OPENFHE_DEBUGEXP(colSize);

  if (origNumRow > rowSize) {
    OPENFHE_THROW(lbcrypto::config_error, __FILE__ + std::string(" ") + __FUNCTION__ + std::string(":") +
        std::to_string(__LINE__) +
        std::string("Error: input vector # rows exceeds rowSize"));
  }
//   zero pad out to rowSize
  for (auto i = origNumRow; i < rowSize; i++) {
    inVec.push_back(0.0);
  }
  OPENFHE_DEBUG("after zero pad");
  OPENFHE_DEBUGEXP(inVec.size());

  if (!IsPow2(inVec.size())) {
    OPENFHE_THROW(lbcrypto::config_error, __FILE__ + std::string(" ") + __FUNCTION__ + std::string(":") +
        std::to_string(__LINE__) +
        std::string("Error: input vector non power of two"));
  }
//   inVecRC is VEC_ROW_CLONED (and Zeropadded)
  Vec inVecRC;

  GetVecRowCloned<double>(inVec, numSlots, 0.0, inVecRC);
  if (dbg_flag) {
    PrintVecRowCloned(inVecRC, rowSize); // note number of rows needed here.
  }
  return inVecRC;
}

CT collateOneDMats2CtVRC(CC &cc, const Mat &inMat, const Mat &inMat2, const int rowSize, const int numSlots, const KeyPair &keys) {
  if (inMat2.size() != inMat.size() || inMat2[0].size() != inMat[0].size()){
    OPENFHE_THROW(lbcrypto::config_error, __FILE__ + std::string(" ") + __FUNCTION__ + std::string(":") +
        std::to_string(__LINE__) +
        std::string("Error: 1D-Matrices to collate are not of the same size!"));
  }

  Vec inVecRc = cloneVecRc(inMat, rowSize, numSlots);
  Vec inVecRc2 = cloneVecRc(inMat2, rowSize, numSlots);
  Vec collated(numSlots);
  for (auto i=0; i < numSlots; i++){
    if ((i / rowSize) % 2 == 0){
      collated[i] = inVecRc[i];
    } else {
      collated[i] = inVecRc2[i];
    }
  }
  // make plaintext
  PT inVecRCPT = cc->MakeCKKSPackedPlaintext(collated);
  //encrypt
  CT ctin = cc->Encrypt(keys.publicKey, inVecRCPT);
  return ctin;
}

///////////////////////////////////////////////////////////
CT Mat2CtMRM(CC &cc, const Mat &inMat, const int rowSize, const int numSlots, const KeyPair &keys) {
  // inMat is to be used in a MatrixVectorProductRow so needs to be encrypted as MAT_ROW_MAJOR nfp x nsp
  // inMat is currently a Mat: vector nrows long of vectors (ncol long)
  // so this storage requirement is differnt, instead of rowSize as a limit this packed with colSize as the width limit.

  OPENFHE_DEBUG_FLAG(false);
  OPENFHE_DEBUG("in Mat2CtMRM");
  int origNumRows = inMat.size();     //n_samp (note transposed)
  int origNumCols = inMat[0].size();  //n_feat (including the intecept column)

  auto numCols = rowSize; //note this is the bigger dimension
  auto numRows = numSlots / numCols;

  OPENFHE_DEBUGEXP(origNumRows);
  OPENFHE_DEBUGEXP(origNumCols);
  OPENFHE_DEBUGEXP(numRows);
  OPENFHE_DEBUGEXP(numCols);

  if (origNumRows > numRows) {
    OPENFHE_THROW(lbcrypto::config_error, __FILE__ + std::string(" ") + __FUNCTION__ + std::string(":") +
        std::to_string(__LINE__) +
        std::string("Error: input matrix # rows exceeds numRows"));
  }
  if (origNumCols > numCols) {
    OPENFHE_THROW(lbcrypto::config_error, __FILE__ + std::string(" ") + __FUNCTION__ + std::string(":") +
        std::to_string(__LINE__) +
        std::string("Error: input matrix # cols exceeds numCols"));
  }

  //  copy matrix to a new array, zero padding rows and columns out to rowSize and columnSize
  Vec inRMZP(numSlots, 0.0); //row major zero padded Note full vector created set to zeros

  auto k = 0; //index into vector to write
  auto i = 0;
  for (; i < origNumRows; i++) {
    auto j = 0;
    for (; j < origNumCols; j++) {
      inRMZP[k] = inMat[i][j]; //fill in a row with data
      k++;
    }
    for (; j < numCols; j++) { // fill in rest of row with zeros
      k++;
    }
  }

  for (; i < numRows; i++) { //fill in zero rows
    for (auto j = 0; j < numCols; j++) {
      inRMZP[k] = 0.0; //fill in a row with data
      k++;
    }
  }
  OPENFHE_DEBUGEXP(inRMZP.size());

  //convert that  that out into inRMZP
  OPENFHE_DEBUG("inRMZP in row major zero pad");

  if (dbg_flag) {
    PrintMatRowMajor(inRMZP, numCols);  //need to verify
  }
  PT inPT = cc->MakeCKKSPackedPlaintext(inRMZP); // encode inPT plaintext matrix
  auto ctin = cc->Encrypt(keys.publicKey, inPT); //ciphertext in
  return ctin;
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
    ){

  usint numSlots = cc->GetEncodingParams()->GetBatchSize();
  /////////////////////////////////////////////////////////
  // Load inputs and set up the problem
  /////////////////////////////////////////////////////////

  // Read training data and labels from CSV file
  // - X, our Data matrix   size numSamp x n_features (nrow x ncol)
  // - y, our result vector size numSamp x 1 (col vector)
  // Note all plaintext matricies and vectors are of type Mat for simlicity
  // i.e. vector is Mat with one singleton dimension

  std::vector<std::string> featureNames;
  std::vector<std::string> labelNames;

  bool normalizeFlag(false); //should this be a command line parameter?
  LoadDataFile(params.trainXFile, X, featureNames, params.rowsToRead, normalizeFlag);
  LoadDataFile(params.testXFile, testX, featureNames, params.rowsToRead, normalizeFlag);
  // We never normalize the labels.
  LoadDataFile(params.trainYFile, y, labelNames, params.rowsToRead, false);
  LoadDataFile(params.testYFile, testY, labelNames, params.rowsToRead, false);

  //determine dimensions for matrix encryptions
  usint originalNumSamp = X.size();     //n_samp
  usint originalNumFeat = X[0].size();  //n_feat (including the intecept column

  if (X.size() != y.size() || testX.size() != testY.size()) {
    std::cerr << " X and y dimension mismatch!" << std::endl;
    exit(EXIT_FAILURE);
  }

#ifdef ENABLE_DEBUG
  std::cout << "Original Training data set size (r x c): "
            << originalNumSamp << " x " << originalNumFeat << std::endl;
  std::cout << "First row of training set (for sanity check): " << std::endl;
  SimplePrintVec("SMALL_SCALE: First X: ", X[0]);
  std::cout << std::endl;
  std::cout << "Training labels size " << y.size() << " x 1" << std::endl;
  std::cout << "First label of training set (for sanity check): " << std::endl;
  SimplePrintVec("SMALL_SCALE: First y: ", y[0]);
  std::cout << std::endl;
#endif // ENABLE_DEBUG

//////////////////////////////////////////////////////
  // Encode and encrypt input data
  // note encrypted matrix code uses rowSize and colSize instead of numRows and numCols
  /*
        rowSize == num features O(10) rounded up to power of two
        colSize = num samples O(1024) or O(64) depending on encoding

        beta: nf x 1 column vector
        X = ns x nf matrix
        -X' = nf x ns matrix
        y = ns x 1 column vector
        mu, lr constant scalars
  */

  //Encoding notes
  // numSlots came from encryption scheme paramters.

  auto dims = ComputePaddedDimensions(originalNumSamp, originalNumFeat, numSlots);
  usint colSize = dims.first;
  usint rowSize = dims.second;
  int signedRowSize = (int) rowSize;

  std::vector<int> rotationIndices = {-signedRowSize, signedRowSize};
  std::cout << "\tEvalRotate keys" << std::endl;
  cc->EvalRotateKeyGen(keys.secretKey, rotationIndices);

  std::cout << "colSize x rowSize = " << colSize << " * " << rowSize << " = " << colSize * rowSize << std::endl;

  if (colSize * rowSize != numSlots) {
    std::cerr << "numSlots exceeded " << numSlots << std::endl;
    exit(EXIT_FAILURE);
  }


  // generate -X' and r (starts as zeros)
  // generate CT for X

#ifdef ENABLE_DEBUG
  std::cout << "beta:" << std::endl;
  PrintMatrix(beta);

  usint nrow2print(4); //print 4 rows or columns for sanity
  std::cout << "Initialized data: " << std::endl;
  std::cout << "X (showing only " << nrow2print << " rows): " << std::endl;
  PrintSubmatrix(X, nrow2print, X[0].size());
  std::cout << "NegXt (showing only " << nrow2print << " col): " << std::endl;
  PrintSubmatrix(NegXt, X.size(), nrow2print);
  std::cout << "beta: " << std::endl;
  PrintMatrix(beta);
  std::cout << std::endl;
#endif // ENABLE_DEBUG

  /////////////////////////////////////////////////////////////////
  //Setup dataset and learning parameters
  /////////////////////////////////////////////////////////////////

  // X will be used in a MatrixVectorProductRow so needs to be encrypted as MAT_ROW_MAJOR rowSize x colSize
  // negXt will be used in MatrixVectorPRoductCol, but since it is transpose of X we can use
  // negX in MAT_ROW_MAJOR because that is the same as negX' in MAT_COL_MAJOR
  // both use the same packing.

  // - Weight vector beta n_features x 1 (col vector)
  beta = Mat(originalNumFeat, Vec(1, 0.0));

  {
    Vec thetaMask = Vec(numSlots, 0);
    Vec phiMask = Vec(numSlots, 0);
    for (uint i = 0; i < numSlots; i++) {
      if ((i / rowSize) % 2 == 0) {
        thetaMask[i] = 1;
      } else {
        phiMask[i] = 1;
      }
    }
    ptExtractThetaMask = cc->MakeCKKSPackedPlaintext(thetaMask);
    ptExtractPhiMask = cc->MakeCKKSPackedPlaintext(phiMask);
  }
  NegXt = InitializeLogReg(X, y, lrGamma / y.size());

}

////////////////////////////////////////////////////////////////////
// Utility print functinons
void PrintVecRowCloned(const Vec &z, const int rowSize) {
  Vec firstSet;
  for (auto j = 0; j < rowSize; j++) {
    firstSet.push_back(z[j]);
  }
  SimplePrintVec("First clone: ", firstSet);

  bool good = true;
  for (auto i = 0U; i < z.size(); i += rowSize) {
    for (auto j = 0; j < rowSize; j++) {
      if (firstSet[j] != z[i + j]) {
        std::cout << "!";
        good &= false;
      }
    }
    if (!good) break;
  }
  std::cout << std::endl;
  if (good) {
    std::cout << " all clones match" << std::endl;
  } else {
    std::cout << " some clones do not match" << std::endl;
    SimplePrintVec("Input vector: ", z);
  }

}

void PrintVecColCloned(const Vec &z, const int rowSize) {
  OPENFHE_DEBUG_FLAG(false);
  OPENFHE_DEBUG("in PrintVecColCloned");
  OPENFHE_DEBUGEXP(rowSize);
  Vec firstSet;
  for (auto i = 0U; i < z.size(); i += rowSize) {
    firstSet.push_back(z[i]);
  }

  OPENFHE_DEBUGEXP(z.size());
  OPENFHE_DEBUGEXP(firstSet.size());
  SimplePrintVec("First entry of each clone: ", firstSet);

  bool good = true;
  unsigned int k(0);
  for (auto i = 0U; i < z.size(); i += rowSize) {
    for (auto j = 0; j < rowSize; j++) {
      if (firstSet[k] != z[i + j]) {
        std::cout << "! " << i << ", " << j << ", " << k << ": " << firstSet[k] << " != " << z[i + j] << std::endl;
        good &= false;
      }
    }
    if (!good) break;
    k++;
  }
  std::cout << std::endl;
  if (good) {
    std::cout << " all clones match" << std::endl;
  } else {
    std::cout << " some clones do not match" << std::endl;
    SimplePrintVec("Input vector: ", z);
  }
}

void PrintMatRowMajor(const Vec &z, const int rowSize) {
  for (auto i = 0U; i < z.size(); i += rowSize) {
    std::cout << "row " << i << ": [";
    for (auto j = 0; j < rowSize; j++) {
      std::cout << z[i + j] << ",";
    }
    std::cout << "]" << std::endl;
  }
}

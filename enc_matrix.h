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

#ifndef ENC_MATRIX_H
#define ENC_MATRIX_H

#include "openfhe.h"
#include "lr_types.h"

// convert 1d vector to row cloned (input must be zero padded to power of two
// output is a VEC_ROW_CLONED
template<typename type>
void GetVecRowCloned(
    std::vector<type> &inVec, uint32_t numSlots, type paddingVal, std::vector<type> &outVec
);



// convert 1d vector to  col cloned (input must be zero padded to power of two)
// input vector must be power of two. output is a VEC_COL_CLONED

template<typename type>
void GetVecColCloned(
    std::vector<type> &inVec, uint32_t numSlots, type paddingVal, std::vector<type> &outVec
);

template<typename Element>
void MatrixVectorProductRow(
    CC &context,
    KeyPair kp,
    std::shared_ptr<std::map<usint, lbcrypto::EvalKey<Element>>> evalSumCols,
    const CT &cMat,
    const CT &cVecRowCloned,
    uint32_t rowSize,
    lbcrypto::Ciphertext<Element> &cProduct
) {
  OPENFHE_DEBUG_FLAG(false);
  auto cMult = context->EvalMult(cMat, cVecRowCloned);
  OPENFHE_DEBUG(cMult->GetLevel());
  cProduct = context->EvalSumCols(cMult, rowSize, *evalSumCols);
  OPENFHE_DEBUG(cProduct->GetLevel());
}

// Product with Col inputs

template<typename Element>
void MatrixVectorProductCol(
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> &context,
    const std::shared_ptr<std::map<usint, lbcrypto::EvalKey<Element>>> evalSumRows,
    const lbcrypto::Ciphertext<Element> &cMat,
    const lbcrypto::Ciphertext<Element> &cVecColCloned,
    const uint32_t rowSize,
    lbcrypto::Ciphertext<Element> &cProduct
) {
  auto cMult = context->EvalMult(cMat, cVecColCloned);
  cProduct = context->EvalSumRows(cMult, rowSize, *evalSumRows);
}

template<typename type>
void GetVecRowCloned(
    std::vector<type> &inVec, uint32_t numSlots, type paddingVal, std::vector<type> &outVec
) {
  size_t n = inVec.size();

  if (numSlots < n)
    OPENFHE_THROW(__FILE__ + std::string(" ") +
                      __FUNCTION__ + std::string(":") +
                      std::to_string(__LINE__) +
                      std::string("Error: numClones x vecSize > numSlots!"));

  if (numSlots == n) {
    outVec = inVec;
    return;
  }

  outVec.clear();
  uint32_t numClones = numSlots / n;

  outVec.reserve(n * numClones);

  for (uint32_t i = 0; i < numClones; i++) {
    std::copy(inVec.begin(), inVec.end(), back_inserter(outVec));
  }

  // padd the remaining values with paddingVal
  outVec.resize(numSlots, paddingVal);

}
// takes inVec as input vector
// deterines's its length and then clones it out to fill the number of slots.
// @returns outVec
template<typename type>
void GetVecColCloned(
    std::vector<type> &inVec, uint32_t numSlots, type paddingVal, std::vector<type> &outVec
) {
  size_t n = inVec.size();
  if (numSlots < n)
    OPENFHE_THROW(__FILE__ + std::string(" ") +
                      __FUNCTION__ + std::string(":") +
                      std::to_string(__LINE__) +
                      std::string("Error: numClones x vecSize > numSlots!"));

  if (numSlots == n) {
    outVec = inVec;
    return;
  }

  outVec.clear();
  uint32_t numClones = numSlots / n;

  for (uint32_t i = 0; i < n; i++)
    for (uint32_t j = 0; j < numClones; j++)
      outVec.emplace_back(inVec[i]);
}

#endif //ENC_MATRIX_H

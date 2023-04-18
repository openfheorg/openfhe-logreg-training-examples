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

#include "openfhe.h"
#include "utils.h"
#include <iostream>

double LOWER_BOUND = -16;
double UPPER_BOUND = -LOWER_BOUND;
long double STEP = 0.001;
uint32_t POLY_DEGREE = 59;
std::string FILE_NAME =
    "../py_scripts/sigmoidResults_" + std::to_string((int) UPPER_BOUND) + "_" + std::to_string(POLY_DEGREE) + ".txt";



// In this example, we evaluate the logistic function 1 / (1 + exp(-x)) on an input of doubles
int main() {

  // The multiplicative depth depends on the polynomial degree.
  // See the FUNCTION_EVALUATION.md file for a table mapping polynomial degrees to multiplicative depths.
  //https://github.com/openfheorg/openfhe-development/blob/main/src/pke/examples/FUNCTION_EVALUATION.md

  uint32_t multDepth = 0;

  // NOTE: Please refer to: https://github.com/openfheorg/openfhe-development/blob/main/src/pke/examples/FUNCTION_EVALUATION.md
  // to select the correct multDepth for your problem.
  if (multDepth == 0) {
    std::cerr << "Please set the multiplicative depth based on: https://github.com/openfheorg/openfhe-development/blob/main/src/pke/examples/FUNCTION_EVALUATION.md" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::cout << "Mult Depth set to: " << multDepth << std::endl;

  std::cout << "--------------------------------- EVAL LOGISTIC FUNCTION ---------------------------------"
            << std::endl;

  lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS> parameters;
  // We set a smaller ring dimension to improve performance for this example.
  // In production environments, the security level should be set to
  // HEStd_128_classic, HEStd_192_classic, or HEStd_256_classic for 128-bit, 192-bit,
  // or 256-bit security, respectively.

  std::ofstream myFile;
  myFile.open(FILE_NAME);
  if (!myFile.is_open()) {
    std::cerr << "File to store in could not be opened" << std::endl;
    exit(EXIT_FAILURE);
  }

  std::vector<std::double_t> input;

  for (long double i = LOWER_BOUND + STEP; i < UPPER_BOUND - STEP; i = i + STEP) {
    input.emplace_back(i);
  }

  std::cout << "\tNumber of elements in the input vector: " << input.size() <<
            "\n\tFully packed batch size: " << NextPow2(input.size()) <<
            "\n\tRequired Ring Size: " << NextPow2(input.size()) * 2 << std::endl;

  uint32_t ringDimension = 1 << 18;
  parameters.SetSecurityLevel(lbcrypto::HEStd_NotSet);
  parameters.SetRingDim(ringDimension);
  std::cout << "Batch size: " << ringDimension / 2 << std::endl;
  parameters.SetBatchSize(ringDimension / 2);
#if NATIVEINT == 128
  usint scalingModSize = 85;
  usint firstModSize = 89;
#else
  usint scalingModSize = 59;
    usint firstModSize   = 60;
#endif
  parameters.SetScalingModSize(scalingModSize);
  parameters.SetFirstModSize(firstModSize);

  parameters.SetMultiplicativeDepth(multDepth);
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc = GenCryptoContext(parameters);
  cc->Enable(PKE);
  cc->Enable(KEYSWITCH);
  cc->Enable(LEVELEDSHE);
  // We need to enable Advanced SHE to use the Chebyshev approximation.
  cc->Enable(ADVANCEDSHE);

  auto keyPair = cc->KeyGen();
  // We need to generate mult keys to run Chebyshev approximations.
  cc->EvalMultKeyGen(keyPair.secretKey);
  size_t encodedLength = input.size();
  std::cout << encodedLength << std::endl;
  lbcrypto::Plaintext plaintext = cc->MakeCKKSPackedPlaintext(input);
  auto ciphertext = cc->Encrypt(keyPair.publicKey, plaintext);

  auto result = cc->EvalLogistic(ciphertext, LOWER_BOUND, UPPER_BOUND, POLY_DEGREE);

  lbcrypto::Plaintext plaintextDec;
  cc->Decrypt(keyPair.secretKey, result, &plaintextDec);
  plaintextDec->SetLength(encodedLength);
  auto resp = plaintextDec->GetRealPackedValue();

  std::string strToIO;

  for (uint32_t i = 0; i < encodedLength; i++) {
    strToIO = std::to_string(input[i]) + "," + std::to_string(resp[i]) + '\n';
    myFile << strToIO;
  }
  myFile.close();

}

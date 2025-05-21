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

#ifndef DPRIVE_ML__PARAMETERS_H_
#define DPRIVE_ML__PARAMETERS_H_

#include <string>

#include "openfhe.h"
#include <getopt.h>
#include <iostream>
#include "data_io.h"
#include "lr_train_funcs.h"
#include "lr_types.h"

class Parameters {

 public:
  Parameters() = default;

  void populateParams(
      int argc,
      char *argv[],
      usint numIters_def,
      bool withBT_def,
      int rowsToRead_def,
      const std::string &trainXFile_def,
      const std::string &trainYFile_def,
      const std::string &testXFile_def,
      const std::string &testYFile_def,
      uint32_t ringDimension_def,
      usint writeEvery_def,
      int btPrecision_def=0,
      bool verbose = false,
      bool withCompositeScaling = false,
      bool doublePrecisionCS = false,
      bool highPrecisionCS = false
  ) {

    std::string outFilePrefix_def = "../results/nag_";
    int outputPrecision_def = dbl::max_digits10;

    numIters = numIters_def;
    withBT = withBT_def;
    rowsToRead = rowsToRead_def;
    trainXFile = trainXFile_def;
    trainYFile = trainYFile_def;
    testXFile = testXFile_def;
    testYFile = testYFile_def;
    std::string outFilePrefix = outFilePrefix_def;
    ringDimension = ringDimension_def;
    btPrecision = btPrecision_def;

    outputPrecision = outputPrecision_def;

    withCS = withCompositeScaling;
    dbPrecisionCS = doublePrecisionCS;
    hPrecisionCS = highPrecisionCS;

    int opt;
    while ((opt = getopt(argc, argv, "bmn:r:x:y:j:k:d:w:p:e:cmn:fmn:tmn:h")) != -1) {
      switch (opt) {
        case 'b':withBT = true;
          std::cout << "bootstrapping enabled" << std::endl;
          break;
        case 'n':numIters = atoi(optarg);
          std::cout << "numIters: " << numIters << std::endl;
          break;
        case 'r':rowsToRead = atoi(optarg);
          std::cout << "rowsToRead: " << rowsToRead << std::endl;
          break;

        case 'e':btPrecision = atoi(optarg);
          std::cout << "Bootstrapping Precision (only valid in 64-bit setups): " << btPrecision << std::endl;
          break;
          /**
           * Train-Test files
           */
        case 'x':trainXFile = optarg;
          std::cout << "trainXFile: " << trainXFile << std::endl;
          break;
        case 'y':trainYFile = optarg;
          std::cout << "trainYFile: " << trainYFile << std::endl;
          break;
        case 'j':testXFile = optarg;
          std::cout << "testXFile: " << testXFile << std::endl;
          break;
        case 'k':testYFile = optarg;
          std::cout << "testYFile: " << testYFile << std::endl;
          break;
        case 'd': ringDimension = atoi(optarg);
          std::cout << "ringDimension: " << ringDimension << std::endl;
          break;
        case 'w':outFilePrefix = optarg;
          std::cout << "output File(s) prefix: " << outFilePrefix << std::endl;
          break;
        case 'p':outputPrecision = atoi(optarg);
          std::cout << "output precision: " << outputPrecision << std::endl;
          break;
        case 'c':withCS = true;
          std::cout << "composite scaling technique enabled" << std::endl; 
          break;
        case 'f':dbPrecisionCS = true;
          std::cout << "composite scaling register size is 64 bits" << std::endl;
          break;
        case 't':hPrecisionCS = true;
          std::cout << "using (non-secure) high precision composite scaling" << std::endl;
          break;
        case 'h':
        default: /* '?' */
          std::cerr << "Usage: " << std::endl
                    << "arguments:" << std::endl
                    << "  -b do bootstraping (emulate otherwise) [" << (withBT_def ? "true" : "false") << "]"
                    << std::endl
                    << "  -e <bootstrapping precision in 64-bit scenario> [" << btPrecision_def << "]" << std::endl
                    << "  -n <number of iterations to perform> [" << numIters_def << "]" << std::endl
                    << "  -r <number of rows to read> [" << rowsToRead_def << "]" << std::endl
                    << "  -x <training X file name> [" << trainXFile_def << "]" << std::endl
                    << "  -y <training y file name> [" << trainYFile_def << "]" << std::endl
                    << "  -j <testing X file name> [" << testXFile_def << "]" << std::endl
                    << "  -k <testing y file name> [" << testYFile_def << "]" << std::endl
                    << "  -d <ring dimension> [" << ringDimension_def << "]" << std::endl
                    << "  -w <output file name prefix> [" << outFilePrefix_def << "]" << std::endl
                    << "  -p <outputPrecision> [" << outputPrecision_def << "]" << std::endl
                    << "  -c enable and run with composite scaling technique [" << (withCompositeScaling ? "true" : "false") << std::endl
                    << "  -t use high precision composite scaling [" << (highPrecisionCS ? "true" : "false") << std::endl
                    << "  -f register word size for composite scaling" << (doublePrecisionCS ? 64 : 32) << std::endl
                    << "  -h prints this message" << std::endl;
          std::exit(EXIT_FAILURE);
      }
    }

    rowsToRead = (rowsToRead == 0) ? -1 : rowsToRead;
    if (withBT) {
      outFilePrefix = outFilePrefix_def + "bootstrap_";
    } else {
      outFilePrefix = outFilePrefix_def + "interactive_";
    }

    weightsOutFile = outFilePrefix + "weights.csv";
    trainOutFile = outFilePrefix + "train.csv";
    testLossOutFile = outFilePrefix + "test.csv";
    lossOutFile = outFilePrefix + "loss.csv";

    std::cerr.precision(outputPrecision); //set output precision.
    if (verbose) {

      std::cout << "Command line arguments: " << std::endl;
      std::cout << "\tIterations: " << numIters << std::endl;
      std::cout << "\tWrite Every: " << writeEvery << std::endl;
      std::cout << "\tUse Bootstrapping? " << withBT << std::endl;
      std::cout << "\tUse Composite Scaling Tech? " << withCS << std::endl;
      std::cout << "\tUse High Precision Composite Scaling? " << hPrecisionCS << std::endl;
      std::cout << "\tComposite Scaling HW Precision: " << ((dbPrecisionCS) ? 64 : 32) << std::endl;
      std::cout << "\tTraining samples to read: " << rowsToRead << std::endl;
      std::cout << "\tTraining X CSV file: " << trainXFile << std::endl;
      std::cout << "\tTraining y CSV file: " << trainYFile << std::endl;
      std::cout << "\tTest X CSV file: " << testXFile << std::endl;
      std::cout << "\tTest y CSV file: " << testYFile << std::endl;
      std::cout << "\tRing Dimension: " << ringDimension << std::endl << std::endl;
      std::cout << "\tOutput precision: " << outputPrecision << std::endl << std::endl;
      std::cout << "\tOutput model weights CSV file: " << weightsOutFile << std::endl;
      std::cout << "\tOutput train prediction CSV file: " << trainOutFile << std::endl;
      std::cout << "\tOutput test loss CSV file: " << testLossOutFile << std::endl;
      std::cout << "\tOutput train loss CSV file: " << lossOutFile << std::endl;
      std::cout << std::endl;
    }
  }

  /////////////////////////////////////////////////////////////////
  //Various params declared
  /////////////////////////////////////////////////////////////////
  usint numIters;
  bool withBT;
  int rowsToRead;
  usint writeEvery;
  uint32_t ringDimension;
  int outputPrecision;
  std::string trainXFile;
  std::string trainYFile;
  std::string testXFile;
  std::string testYFile;
  std::string weightsOutFile;
  std::string trainOutFile;
  std::string testLossOutFile;
  std::string lossOutFile;
  int btPrecision;
  bool withCS;
  bool dbPrecisionCS;
  bool hPrecisionCS;
};

#endif //DPRIVE_ML__PARAMETERS_H_

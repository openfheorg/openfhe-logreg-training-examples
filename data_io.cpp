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

#include "limits"
#include "data_io.h"

void ReadHeader(
    std::istream &is,
    std::vector<std::string> &featureNames) {

  std::string line;
  getline(is, line);
  Vec fields;
  std::string tok;
  std::stringstream ss(line);
  while (getline(ss, tok, ',')) {
    featureNames.push_back(tok);
  }
}

void ReadData(
    std::istream &is,
    Mat &data,
    int maxLines) {

  if (maxLines < 0)
    maxLines = std::numeric_limits<int>::max();

  if (maxLines == 0) {
    std::cerr << "Please specify a non-zero number of rows to read." << std::endl;
    exit(0);
  }

  std::string line;
  while (getline(is, line) && (maxLines-- > 0)) {
    //fields contains the intercept as well
    Vec fields;
    std::string tok;
    std::stringstream ss(line);
    while (getline(ss, tok, ',')) {
      fields.push_back(stof(tok));
    }
    Vec dataRow(fields.begin(), fields.end());
    data.push_back(dataRow);
  }
}

void LoadDataFile(
    std::string filename,
    Mat &data,
    std::vector<std::string> &featureNames,
    int numRowsToRead,
    bool normalize_flag) {

  std::filebuf fb;
  if (fb.open(filename, std::ios::in)) {

    std::istream is(&fb);

    ReadHeader(is, featureNames);
    ReadData(is, data, numRowsToRead);

    fb.close();
  } else {
    std::cerr << "Error reading in file " << filename << std::endl;
    exit(EXIT_FAILURE);
  }

  int numCols = data[0].size();
  int numRows = data.size();
  Vec accume(numCols, 0.0);
  Vec maxVals(numCols, -1e10);
  Vec minVals(numCols, 1e10);

  // Summary stats for each col
  for (auto i = 0; i < numRows; i++) {
    for (auto j = 0; j < numCols; j++) {
      auto val = data[i][j];
      accume[j] += val;
      maxVals[j] = std::max(maxVals[j], val);
      minVals[j] = std::min(minVals[j], val);
    }
  }
  for (auto j = 0; j < numCols; j++) {
    accume[j] /= double(numRows);
  }

  std::cout << "Feature Analysis:    min     ave    max" << std::endl;
  for (auto j = 0; j < numCols; j++) {
    std::cout << "\t" << featureNames[j] << ": " << minVals[j] << " " << accume[j] << " " << maxVals[j] << std::endl;
  }

  if (normalize_flag) {
    std::cout << "Normalizing all input data to +-0.5" << std::endl;
    Vec normVals(numCols, 0.0);

    for (auto j = 0; j < numCols; j++) {
      normVals[j] = std::max(abs(maxVals[j]), abs(minVals[j])) * 2.0;
    }

    for (auto i = 0; i < numRows; i++) {
      for (auto j = 0; j < numCols - 1; j++) { //do not adjust intercept
        data[i][j] /= normVals[j];

      }
    }

    {
      Vec normalizedAccumE(numCols, 0.0);
      Vec normalizedMaxVals(numCols, -1e10);
      Vec normalizedMinVals(numCols, 1e10);

      for (auto i = 0; i < numRows; i++) {
        for (auto j = 0; j < numCols; j++) {
          auto val = data[i][j];
          normalizedAccumE[j] += val;
          normalizedMaxVals[j] = std::max(normalizedMaxVals[j], val);
          normalizedMinVals[j] = std::min(normalizedMinVals[j], val);
        }
      }
      for (auto j = 0; j < numCols; j++) {
        normalizedAccumE[j] /= double(numRows);
      }

      std::cout << "Normalized:" << std::endl;
      std::cout << "feature:    min     ave    max" << std::endl;
      for (auto j = 0; j < numCols; j++) {
        std::cout << featureNames[j] << ": " << normalizedMinVals[j] << " " << normalizedAccumE[j] << " "
                  << normalizedMaxVals[j] << std::endl;
      }
    }

  }

}

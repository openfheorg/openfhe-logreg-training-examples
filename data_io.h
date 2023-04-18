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

#ifndef DPRIVE_ML__DATA_IO_H_
#define DPRIVE_ML__DATA_IO_H_

#include <iostream>
#include <vector>
#include <string>
#include "lr_types.h"

/* reads in the feature names from the header of the file
 */
void ReadHeader(std::istream &is, std::vector<std::string> &featureNames);

/* Reads up to maxLines records from file and parses them into data and labels.
 * If maxLines is negative, it will read as many lines are in the file.
 * note intercept is already in this data.
 */
void ReadData(std::istream &is, Mat &data, int maxLines = -1);

/* Creates an input file stream and uses ReadData to read rowsToRead rows from the file.
 * If rowsToRead is negative, it will read all rows in the file.
 */
void LoadDataFile(std::string filename, Mat &data, std::vector<std::string> &featureNames, int rowsToRead, bool normalize_flag);

#endif //DPRIVE_ML__DATA_IO_H_

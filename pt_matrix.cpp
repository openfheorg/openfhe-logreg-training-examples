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

#include "pt_matrix.h"

/* Multiplies a matrix A with a scalar value t, in place.
 */
void MatrixScalarMult(Mat &A, prim_type t) {
  for (usint i = 0; i < A.size(); i++) {
    for (usint j = 0; j < A[i].size(); j++) {
      A[i][j] = t * A[i][j];
    }
  }
}

void ScalarSubMat(prim_type t, Mat &A, Mat &B){
  for (usint i = 0; i < A.size(); i++) {
    for (usint j = 0; j < A[i].size(); j++) {
      B[i][j] = t - A[i][j];
    }
  }
}

void MatrixMatrixAdd(Mat &A, Mat &B, Mat &C){
  if (A.size() != B.size()|| B.size() != C.size()){
    throw std::invalid_argument("MatrixMatrixAdd A B and C must all have same leading dimension");
  }

  if (A[0].size() != B[0].size() || B[0].size() != C[0].size()) {
    throw std::invalid_argument("MatrixMatrixAdd A B and C must all have same trailing dimension");
  }
  for (usint i = 0; i < A.size(); i++) {
    for (usint j = 0; j < A[i].size(); j++) {
      C[i][j] = A[i][j] + B[i][j];
    }
  }
}

void MatrixMatrixSub(Mat &A, Mat &B, Mat &C) {
  if (A.size() != B.size() || B.size() != C.size()) {
    throw std::invalid_argument("MatrixMatrixAdd A B and C must all have same leading dimension");
  }

  if (A[0].size() != B[0].size() || B[0].size() != C[0].size()) {
    throw std::invalid_argument("MatrixMatrixAdd A B and C must all have same trailing dimension");
  }
  for (usint i = 0; i < A.size(); i++) {
    for (usint j = 0; j < A[i].size(); j++) {
      C[i][j] = A[i][j] - B[i][j];
    }
  }
}

/* Applies the sigmoid function on a matrix A, in place.
 */
void MatrixSigmoid(Mat &A) {
  for (usint i = 0; i < A.size(); i++) {
    for (usint j = 0; j < A[i].size(); j++) {
      A[i][j] = prim_type(1) / (prim_type(1.0) + exp(-A[i][j]));
    }
  }
}
void MatrixLog(Mat &A, Mat &B) {
  for (usint i = 0; i < A.size(); i++) {
    for (usint j = 0; j < A[i].size(); j++) {
      B[i][j] = std::log(A[i][j]);
    }
  }
}

/* Prints matrix A.
 */
void PrintMatrix(const Mat &A) {
  //set the output precision to double max digits.
  std::cerr.precision(dbl::max_digits10);

  for (usint i = 0; i < A.size(); i++) {
    std::cerr << "[ ";
    for (usint j = 0; j < A[i].size(); j++) {
      std::cerr << A[i][j] << ", ";
    }
    std::cerr << " ]" << std::endl;
  }
}

/* Prints Submatrix of A. 0..nrow-1 x 0..ncol-1
 */

void PrintSubmatrix(const Mat &A, const unsigned int nrow, const unsigned int ncol) {
  //set the output precision to double max digits.
  std::cerr.precision(dbl::max_digits10);

  usint nr = std::min(nrow, usint(A.size()));
  usint nc = std::min(ncol, usint(A[0].size()));

  for (usint i = 0; i < nr; i++) {
    std::cerr << "[ ";

    for (usint j = 0; j < nc; j++) {
      std::cerr << A[i][j] << ", ";
    }
    std::cerr << " ]" << std::endl;
  }
}

void MatrixMult(const Mat &A, const Mat &B, Mat &C) {

  auto numRows = A.size();
  auto numCols = B[0].size();
  auto middleDim = A[0].size();

  if (middleDim != B.size()) {
    throw std::invalid_argument(" Matrixmult: Input Dimension mismatch");
  }

  if ((numRows != C.size()) || (numCols != C[0].size())) {
    throw std::invalid_argument(" Matrixmult: Output Dimension mismatch");
  }

  for (auto i = 0U; i < numRows; i++) {
    for (auto j = 0U; j < numCols; j++) {
      for (auto k = 0U; k < middleDim; k++) {
        //        C[i][j] += A[i][k] * B[k][j];
        C.at(i).at(j) += A.at(i).at(k) * B.at(k).at(j);
      }
    }
  }
}

void MatrixTransp(const Mat &A, Mat &AT) {
  auto numRowsA = A.size();
  auto numColsA = A[0].size();

  if ((numRowsA != AT[0].size()) || (numColsA != AT.size())) {
    throw std::invalid_argument(" MatrixTransp: Output Dimension mismatch");
  }

  for (auto i = 0U; i < numRowsA; i++) {
    for (auto j = 0U; j < numColsA; j++) {
      //      AT[j][i] += A[i][j];
      AT.at(j).at(i) += A.at(i).at(j);
    }
  }
}

void InvertMatrix(Mat &x) {
  int dim = x.size();
  if (dim <= 0) return;  // sanity check
  if (dim == 1) return;  // must be of dimension >= 2

  for (int i = 1; i < dim; i++) x[0][i] /= x[0][0];  // normalize row 0

  for (int i = 1; i < dim; i++) {
    for (int j = i; j < dim; j++) {  // do a column of L
      prim_type sum = 0.0;
      for (int k = 0; k < i; k++) {
        sum += x[j][k] * x[k][i];
      }
      x[j][i] -= sum;
    }

    if (i == dim - 1) continue;

    for (int j = i + 1; j < dim; j++) {  // do a row of U
      prim_type sum = 0.0;
      for (int k = 0; k < i; k++) {
        sum += x[i][k] * x[k][j];
      }
      x[i][j] = (x[i][j] - sum) / x[i][i];
    }
  }

  for (int i = 0; i < dim; i++) { // invert L
    for (int j = i; j < dim; j++) {
      prim_type tmp = 1.0;
      if (i != j) {
        tmp = 0.0;
        for (int k = i; k < j; k++)
          tmp -= x[j][k] * x[k][i];
      }
      x[j][i] = tmp / x[j][j];
    }
  }

  for (int i = 0; i < dim; i++) { // invert U
    for (int j = i; j < dim; j++) {
      if (i == j) continue;
      prim_type sum = 0.0;
      for (int k = i; k < j; k++) {
        sum += x[k][j] * ((i == k) ? 1.0 : x[i][k]);
      }
      x[i][j] = -sum;
    }
  }

  for (int i = 0; i < dim; i++) { // final inversion
    for (int j = 0; j < dim; j++) {
      prim_type sum = 0.0;
      for (int k = ((i > j) ? i : j); k < dim; k++) {
        sum += ((j == k) ? 1.0 : x[j][k]) * x[k][i];
      }
      x[j][i] = sum;
    }
  }
}

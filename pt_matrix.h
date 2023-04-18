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

#ifndef DPRIVE_ML__PT_MATRIX_H_
#define DPRIVE_ML__PT_MATRIX_H_
#include "lr_types.h"

///////// Function declarations related to plaintext matrix arithmetic  ///////////////////////////////
// note for simplicity vectors are also represented by Matricies (with a singleton dimension

/* Performs matrix multiplication in the clear using the textbook O(n^3) algorithm.
 * Matrix dimensions: A(numRows, middleDim) x B(middleDim, numCols) = C(numRows, numCols)
 * Matrix C has to be allocated outside the function.
 */
void MatrixMult(const Mat &A, const Mat &B, Mat &C);

/* Transposes matrix A of dimensions (numRows, numCols) and puts the result in
 * matrix AT of dimensions (numCols, numRows).
 * Matrix AT has to be allocated outside the function.
 */
void MatrixTransp(const Mat &A, Mat &AT);

/* Inverts square matrix A of dimensions (dim, dim) in place.
 * More info about the algorithm in:
* Multiplies a matrix A with a scalar value t, in place.
 */
void MatrixScalarMult(Mat &A, prim_type t);

void MatrixMatrixSub(Mat &A, Mat &B, Mat &C);

void ScalarSubMat(prim_type t, Mat &A, Mat &B);

/* Applies the sigmoid function on a matrix A, in place.
 */
void MatrixSigmoid(Mat &A);
void MatrixLog(Mat &A, Mat &B);

/* Prints matrix A.
 */
void PrintMatrix(const Mat &A);

/* Prints Submatrix of A. 0..nrow-1 x 0..ncol-1
 */
void PrintSubmatrix(const Mat &A, const unsigned int nrow, const unsigned int ncol);

#endif //DPRIVE_ML__PT_MATRIX_H_

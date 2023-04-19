# Logistic Regression Training Examples using CKKS

This repository provides examples of how to perform the training of logistic regression models on data encrypted by FHE using the OpenFHE library. This implementation is intended for demonstrations of how to use OpenFHE for model training.  The examples are intended to be used for illustrative purposes only, and not for benchmarking.  There are many much more efficient approaches to logistic regression training that either use [proprietary designs](https://dualitytech.com) or aren't as good for illustrative examples.  The specific approach we use here is based on [Nesterov-accelerated gradient descent](https://jlmelville.github.io/mize/nesterov.html).

**Note** These examples were developed as part of the DARPA DPRIVE program.  They are solely for research purposes and should not be used for benchmarking or production purposes where performance is critical. Although the sample code was contributed by [Duality Technologies](https://www.dualitytech.com), this sample code is not related to the privacy-preserving logistic regression training capabilities provided in past, present or future Duality Technology products.

# Table of Contents

1. [Logistic Regression](#Logistic-Regression)
2. [Building the Code](#Building-this-repository)
3. [Implementation Notes](#implementation-notes-)
   1. [Iterative Bootstrapping](#multi-iteration-bootstrap)
   2. [Sparse Packing](#sparse-packing)
4. [Contents](#Repository-Contents)
   1. [C++ Files](#c-code)
   2. [PyScripts](#pyscripts-folder)
   3. [Results](#results-folder)
   4. [Sigmoid Approx Results](#sigmoidapproxresults)
   5. [Train Data](#traindata)
5. [Acknowledgments](#acknowledgments)

# Logistic Regression

In this repository we have implemented logistic-regression model training and model inference on the 2014 US Infant
Mortality Dataset (as provided in the `train_data` directory).

## Inference:

Given our design matrix, **X**, and our vector of weights, $\bar{w}$, our model prediction is expressed as:

```math
\hat{y} = sigmoid(\mathbf{X}\bar{w})  
```

**Note**: the sigmoid function can also be called the logistic function, and gives logistic regression its name

**Note**: one limitation of FHE is that we cannot directly calculate non-linear functions. As such, we use polynomial
approximations of our non-linear functions. In OpenFHE, we provide the utility `EvalChebyshev` to do such
approximations. See
our [official documentation](https://github.com/openfheorg/openfhe-development/blob/main/src/pke/examples/FUNCTION_EVALUATION.md)
and
our [working example](https://github.com/openfheorg/openfhe-development/blob/main/src/pke/examples/function-evaluation.cpp)

## Loss function

Our loss function is the cross-entropy loss function

```math
\mathcal{l} = -(y * log(\hat{y}) + (1 - y) * log(1 - \hat{y}))
```

## Gradient Descent

We optimize our model via the gradient descent method so our objective is:

```math
\frac{\partial l}{\partial \bar{w}} = -X^T(y - \hat{y})
```

## Optimization method

### Nesterov Accelerated Gradient

Nesterov Accelerated Gradient can be thought of as a classical gradient descent, with a second "phase" that involves a special momentum parameter. 

Given our parameters, $\theta$ and $\phi$ such that:

- $\theta$ is our $\bar{w}$, our "weights"
- $\phi$ is our momentum tracker

our update can be expressed as follows:

```math
\text{Stage 1, momentum update:} \phi_{t+1} = \theta_t + \eta \nabla f(\theta_t)
\text{Stage 2, weight update:} \theta_{t+1} = \phi_{t+1} + \mu (\phi_{t+1} - \phi_t)
```

## FHE Logistic Regression Notes

Although logistic regression in-the-clear and in FHE are similar, we leverage a number of optimizations. For example, we pre-compute various multiplications in-the-clear
to reduce the number of required ciphertext multiplications.

### Pre-computation

- pre-computing the $-X^T$ prior to the gradient descent step
- pre-scaling the $-X^T$ by dividing by the number of samples

# Building this repository

1) Install OpenFHE as per
   the [instructions](https://openfhe-development.readthedocs.io/en/latest/sphinx_rsts/intro/installation/installation.html)

2) Build this repo

```
mkdir build
cd build
cmake ..
make -j N
```

where `N` is the number of cores you want to use.

3) Go into your build directory and run `./lr_nag`.

### 1.2) Options

#### CLI Arguments + Defaults

```
-b flag: whether to run in bootstrap or not. DEFAULT: true
-n int: number of iterations for training. DEFAULT: 200
-r int: rows to read from the dataset. DEFAULT: -1 (read all)
-x string: train features (CSV). DEFAULT: train_data/X_norm_1024.csv
-y string: train labels (CSV). DEFAULT: train_data/y_1024.csv
-j string: test features (CSV). DEFAULT: train_data/X_norm.csv
-k string: test labels (CSV). DEFAULT: train_data/y.csv
-d int: ring dimension. DEFAULT: 1 << 17
-w string: Outpuit file prefix. DEFAULT: See below
-p int: Output precision. DEFAULT: 0. If non-0 we run 2-iteration bootstrap. See below for more information
```

`-w` default: depends on the formulation (sgd/ nag) but amounts to either `../results/nag_` or `../results/sgd_`

# Implementation Notes:

## Multi-iteration bootstrap

In the case of 64-bit precision (specified when installing OpenFHE), we **can** run bootstrap twice, which leads to improved precision, and should rival
that of bootstrapping in 128-bit. If you specify a non-zero precision, we run in 2-iteration mode, else just single iteration. See 
[iterative-ckks-bootstrapping](https://github.com/openfheorg/openfhe-development/blob/main/src/pke/examples/iterative-ckks-bootstrapping.cpp) for more information.

## Sparse Packing

Note how we pack the `Theta` and the `Phi` into a single ciphertext. This is to allow us to run only a single bootstrap as opposed to two, one for each parameter. See [advanced-ckks-bootstrapping](https://github.com/openfheorg/openfhe-development/blob/main/src/pke/examples/advanced-ckks-bootstrapping.cpp) for more information.

# Repository Contents

## C++ Code

- `cheb_analysis.cpp`: for a given polynomial degree and range to estimate over, this generates the estimations and
  outputs the contents to a file in the `py_scripts/` folder. The file can then be analyzed to study the estimated error
  between the estimated value and the actual value at various points.

- `data_io`: header and source file for reading in a CSV file.
- `enc_matrix`: header and source file for various encrypted matrix operations, primarily encrypted matrix
  multiplications
- `lr_nag.cpp`: the "main" file to kick off the logistic regression training.
- `lr_train_funcs`: header and source file for handling training.
- `lr_types.h`: Type aliases
- `parameters.h`: code for crypto-parameter setting and parsing from command-line arguments.
- `pt_matrix`: code for plaintext matrix operations e.g. matrix multiplication, transpose, addition
- `utils`: printing and packing plaintext matrices

## py_scripts folder

Contains various misc. python scripts.

**Note:** the code for training in `parameter_search.ipynb` and `step_by_step_training_debugger.ipynb` is very similar.
However,
the code is separated because we use `numba` to accelerate our hyperparameter search, and numba is not amenable to
debugging via `print`.

`parameter_search.ipynb`:

- Code to run our hyperparameter search
- Code to analyze the
    - convergence: according to criteria set in original R scripts
    - loss
    - AUC-ROC
- Outputs the plots into `plots` subfolder
- Note: we ran the search process for a maximum of `1_000` epochs before terminating. At which point we mark the run
  as "not converged"

`step_by_step_training_debugger.ipynb`:

- when prototyping the code, I found it useful to have a python implementation that I could compare against to ensure
  that my implementation was not diverging. This file implements the NAG.

## results folder

Where we store the results of our encrypted runs.

Note: our encrypted run produces four artifacts:

- train predictions
- test predictions
- train loss
- test loss

## sigmoidApproxResults

Investigates what happens as we modify the sigmoidApprox parameters. The goal of this is to explore:

- the amount of error at various approximation degrees
- the effect that changing the degrees has on the training process

`chebyshev_approx_analysis.ipynb`:

- reads in data from `raw_data/sigmoidResults_X_Y.txt` where `X` describes the range of estimation `(-X, X)` and `Y`
  describes the polynomial degree estimation.

- plots the
    - values across a specified range
    - estimation error (exact - approximation)
    - stacked plot of the above two

- note: the data that is read in is generated from `cheb_analysis.cpp`

`train_analysis.ipynb`:

- studies the amount to which the approximation degree impacts the training process.
- read in data from `raw_data/X_Y_loss.csv` where `X` is the optimization method and `Y` is the approximation degree

- note: we explored two optimization methods
    - nesterov-accelerated-gradient descent

## train_data

Contains the data files. We prototyped on `(X_norm_64, y_64)` and then validated on `(X_norm_1024, y_1024)` before
moving to the full-scale `(X_norm_32764, y_32764)`

`reduceDataset.py` subsamples the dataset to make the number of true cases and false cases to be equal.

# Contributors

These examples were mainly developed by Ian Quah, with some contributions/suggestions from Ahmad Al Badawi, David Bruce Cousins, and Yuriy Polyakov.

# Acknowledgments

Distribution Statement "A" (Approved for Public Release, Distribution Unlimited). This work is supported in part by
DARPA through HR0011-21-9-0003 and HR0011-20-9-0102. The views, opinions, and/or findings expressed are those of the
author(s) and should not be interpreted as representing the official views or policies of the Department of Defense or
the U.S. Government.

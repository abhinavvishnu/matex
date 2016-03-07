// Copyright 2009 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef _OPENSOURCE_PSC_EVD_H__
#define _OPENSOURCE_PSC_EVD_H__

#include "common.h"

namespace learning_psc {
// EigenSolverSymmetric is a wrapper of PARPACK which is easier to use.
// To use it, you need to provide matrix size and callback function y=Ax.
class EigenSolverSymmetric {
 public:
  // num_local_row: how many matrix rows belong to me
  // eigen_type:    "LM": Largest magnitude
  //                "SM": Smallest magnitude
  //                "LA": Largest algebraic
  //                "SA": Smallest algebraic
  //                "BE": Half from each end of spectrum
  // eigen_desire: how many eigenvalues do we need
  // eigen_space:  how many eigenspace do we give it. The more the better.
  //               Emperically set to 2* eigen_desire
  // av:           Callback of y=Ax
  EigenSolverSymmetric(int num_local_row, string eigen_type,
              int eigen_desire, int eigen_space);
  ~EigenSolverSymmetric();

  // T av: the callback object to compute y=Ax
  // For the other parameters, return the pointer the result.
  // eigen_got: How many eigenvalues do we calculate out. The number may be
  //            less than eigen_desire.
  // eigen_value: Real part of the eigenvalues we have got.
  // eigen_value_vector: Eigenvators. Size: eigen_got*num_local_row. We have to
  //                     merge after this function to get full eigenvectors of
  //                     size eigen_got*num_row
  template <class T>
  void Solve(T& av, int* eigen_got, double** eigen_value,
             double ** eigen_vector);

  void set_max_iterations(int max_iters) { max_iterations_ = max_iters; }
  void set_tolerance(double tolerance) { tolerance_ = tolerance; }
 private:
  // Solver parameters
  int num_local_row_;
  int eigen_space_, eigen_desire_;
  int max_iterations_;
  double tolerance_;
  string eigen_type_;

  // Workspace for arpack
  long int* select_;
  double* resid_;
  double* v_;
  double* workd_;
  double* workl_;
  double* d_;
};

class Evd {
 public:
  Evd();
  ~Evd();
  // Read the data and store the sparse matrix into private members.
  // For parallel version, only store part of the matrix.
  void Read(const string& filename);
  // Compute eigenvalues of the sparse matrix. Based on arpack, we wish to
  // compute num_eigen largest eigenvalues, but we must give an eigen_space
  // which should be much larger than num_eigen.
  // Suggest eigen_space > 2*num_eigen
  void Compute(int num_eigen, int eigen_space, int max_iterations,
               double tolerance);
  // Write eigenvalues.
  // Write eigenvectors.
  void Write(const string& eigenvaluefile,
             const string& eigenvectorfile);
  // Get evd result.
  int num_converged() { return num_converged_; }
  const double* eigen_values() { return eigen_values_; }
  const double* eigen_vectors() { return eigen_vectors_; }
  // Given x, compute y=Ax
  void operator()(double *x, double *y) const;
 private:
  // Internal use.
  void AppendItem(int local_row_index, int column, double value);
  // While reading matrix, the matrix size is num_total_row_ * num_total_row_.
  // For paralle version, the number of rows stored on me is num_local_row_.
  int num_total_rows_;
  int num_local_rows_;
  int my_start_row_index_;
  // How many rows each task has. row_count_of_each_proc_.size() = num_procs
  vector<int> row_count_of_each_proc_;
  int row_count_max_;
  // The matrix is sparsely stored using Compressed Row Storage
  // column_index_ contains all elements' column index.
  // value_ contains all elements' values.
  // row_start_index_ contains element index of the first element of each row.
  // column_index_.size() and value_.size() = number of nonzeroes.
  // row_start_index_.size() = num_local_row_;
  vector<int> column_index_;
  vector<double> value_;
  vector<int> row_start_index_;

  // MPI info
  int myid_;
  int pnum_;
  // Eigenvalue solver
  EigenSolverSymmetric* solver_;
  // EVD result
  int num_converged_;
  double* eigen_values_;
  double* eigen_vectors_;
};


}  // namespace learning_psc

#endif  // _OPENSOURCE_PSC_EVD_H__

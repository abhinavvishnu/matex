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

#ifndef _OPENSOURCE_PSC_KMEANS_H__
#define _OPENSOURCE_PSC_KMENAS_H__

#include "common.h"

namespace learning_psc {
class KMeans {
 public:
  KMeans();
  // Read the file and store the eigenvectors(reduced features) into
  // private members.
  void Read(const string& filename);
  // Do kmeans based on parameters.
  void DoKMeans(int num_clusters,
                const string& kmeans_initialize_method,
                int kmeans_max_loop,
                double kmeans_threshold);
  // Write cluster results
  void Write(const string& output_path);
  // For unitest, get cluster result.
  const vector<int>& local_memberships() { return local_memberships_; }
 private:
  // Initialize kmeans centers.
  void InitializeCenters(const string& kmeans_initialize_method);
  void GeneratePermutation(int length, int buf_size,
                           int seed, vector<int> *perm);
  void KMeansClustering(int kmeans_max_loop, double kmeans_threshold);
  void FindNearestCluster(int data_point_index,
                          int *center_index,
                          double *min_distance);
  double DotProduct(const vector<double>& v1, const double* v2);
  double DotProduct(const vector<double>& v1, const vector<double>& v2);
  double ComputeDistance(const vector<double> &data_point_1,
                         const vector<double> &data_point_2);
  double ComputeDistance(const vector<double> &data_point_1,
                         const double* data_point_2);
  // Local data
  // The eigenvalues and eigenvectors computed by eigenvalue decomposition.
  vector<vector<double> > local_rows_;
  int num_total_rows_;
  int my_start_row_index_;
  vector<int> row_count_of_each_proc_;
  int num_clusters_;
  int num_columns_;
  // Kmeans cluster center: num_clusters * num_columns_
  vector<double> cluster_centers_storage_;
  vector<double*> cluster_centers_;
  vector<int> cluster_sizes_;
  // Data samples memberships. Final kmeans result.
  vector<int> local_memberships_;

  // MPI related
  int myid_;
  int pnum_;
};

}  // namespace learning_psc

#endif  // _OPENSOURCE_PSC_KMEANS_H__

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

#include <cfloat>
#include "common.h"
#include "kmeans.h"

namespace learning_psc {
KMeans::KMeans()
    : num_total_rows_(0),
      my_start_row_index_(0) {
  MPI_Comm_rank(MPI_COMM_WORLD, &myid_);
  MPI_Comm_size(MPI_COMM_WORLD, &pnum_);
}

void KMeans::Read(const string& filename) {
  num_total_rows_ = 0;
  if (myid_ == 0) {
    ifstream fin2(filename.c_str());
    string line;
    while (getline(fin2, line)) {  // Each line is a training document.
      if (line.size() > 0 &&      // Skip empty lines.
        line[0] != '\r' &&      // Skip empty lines.
        line[0] != '\n' &&      // Skip empty lines.
        line[0] != '#') {
        ++num_total_rows_;
      }
    }
  }
  MPI_Bcast(&num_total_rows_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Compute how many rows each computer has.
  row_count_of_each_proc_.resize(pnum_, 0);
  for (int i = 0; i < pnum_; ++i) {
    row_count_of_each_proc_[i] = num_total_rows_ / pnum_ + (num_total_rows_ % pnum_ > i ? 1 : 0);
  }

  my_start_row_index_ = 0;
  for (int i = 0; i < myid_; ++i) {
    my_start_row_index_ += row_count_of_each_proc_[i];
  }
  // Read my part
  ifstream fin2(filename.c_str());
  string line;
  int i = 0;
  while (getline(fin2, line)) {  // Each line is a training document.
    if (line.size() > 0 &&      // Skip empty lines.
        line[0] != '\r' &&      // Skip empty lines.
        line[0] != '\n' &&      // Skip empty lines.
        line[0] != '#') {       // Skip comment lines.
      if (i < my_start_row_index_ || i >= my_start_row_index_ + row_count_of_each_proc_[myid_]) {
        ++i;
        continue;
      }
      istringstream ss(line);
      vector<double> row;
      double value;
      while (ss >> value) {  // Load and init a document.
        row.push_back(value);
      }
      local_rows_.push_back(row);
      ++i;
    }
  }
  // Check all instances(rows) are of the same dimension.
  for (int i = 1; i < local_rows_.size(); ++i) {
    CHECK_EQ(local_rows_[0].size(), local_rows_[i].size());
  }
  num_columns_ = local_rows_[0].size();
}

void KMeans::DoKMeans(int num_clusters,
                      const string& kmeans_initialize_method,
                      int kmeans_max_loop,
                      double kmeans_threshold) {
  if (num_clusters == 0) {
    num_clusters_ = local_rows_[0].size();
  } else {
    num_clusters_ = num_clusters;
  }
  InitializeCenters(kmeans_initialize_method);
  KMeansClustering(kmeans_max_loop, kmeans_threshold);
}
void KMeans::Write(const string& filename) {
  for (int p = 0; p < pnum_; ++p) {
    if (p == myid_) {
      std::ofstream fout;
      if (p == 0) {
        fout.open(filename.c_str());
      } else {
        fout.open(filename.c_str(), std::ios::app);
      }
      for (int i = 0; i < local_memberships_.size(); ++i) {
        fout << local_memberships_[i] << std::endl;
      }
      fout.close();
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

void KMeans::InitializeCenters(const string& kmeans_initialize_method) {
  // Normalize eigen vectors.
  for (int i = 0; i < local_rows_.size(); ++i) {
    double norm_i = DotProduct(local_rows_[i], local_rows_[i]);
    if (norm_i != 0) {
      norm_i = sqrt(norm_i);
      for (int j = 0; j < local_rows_[i].size(); ++j) {
        local_rows_[i][j] /= norm_i;
      }
    }
  }

  cluster_centers_storage_.resize(num_clusters_ * num_columns_, 0);
  cluster_centers_.resize(num_clusters_, 0);
  for (int i = 0; i < num_clusters_; ++i) {
    cluster_centers_[i] = &(cluster_centers_storage_[i * num_columns_]);
  }

  if (kmeans_initialize_method == "orthogonal_centers") {
    // Initialize the first cluster center by randomly selecting a row.
    int rand_index = 0;
    if (myid_ == 0) {
      rand_index =
          static_cast<int>(RandDouble()
                           * local_rows_.size());
      for (int i = 0; i < num_columns_; ++i) {
        cluster_centers_[0][i] = local_rows_[rand_index][i];
      }
    }

    MPI_Bcast(&cluster_centers_[0][0],
              num_columns_,
              MPI_DOUBLE,
              0, MPI_COMM_WORLD);
    // Use most orthogonal (smallest product) points to initialize
    // other centers.
    // additional_vector stores the comparisons between data points
    // and centers. We choose the one in additional_vector with the smallest
    // value (most orthogonal) as the next center.
    vector<double> additional_vector(local_rows_.size(), 0);
    for (int k = 1; k < num_clusters_; ++k) {
      int orthogonal_angle_index = -1;
      double orthogonal_angle = FLT_MAX;
      for (int i = 0; i < local_rows_.size(); ++i) {
        additional_vector[i] +=
            fabs(DotProduct(local_rows_[i],
                            cluster_centers_[k - 1]));
        if (additional_vector[i] < orthogonal_angle) {
          orthogonal_angle = additional_vector[i];
          orthogonal_angle_index = i;
        }
      }
      // Collect orthogonal points from all machines, then
      // pick the most orthogonal one from those points.
      IndexValue local_id_angle, global_id_angle;
      local_id_angle.index = myid_;
      local_id_angle.value = orthogonal_angle;
      MPI_Allreduce(&local_id_angle,
                    &global_id_angle,
                    1,
                    MPI_DOUBLE_INT,
                    MPI_MINLOC, MPI_COMM_WORLD);
      int global_orthogonal_proc_index = global_id_angle.index;

      if (myid_ == global_orthogonal_proc_index) {
        for (int i = 0; i < num_columns_; ++i) {
          cluster_centers_[k][i] =
              local_rows_[orthogonal_angle_index][i];
        }
      }
      MPI_Bcast(&cluster_centers_[k][0],
                num_columns_,
                MPI_DOUBLE,
                global_orthogonal_proc_index, MPI_COMM_WORLD);
    }
    for (int i = 0; i < num_clusters_; ++i) {
      cluster_sizes_.push_back(0);
    }
  } else if (kmeans_initialize_method == "random") {
    // Initialize cluster centers by randomly using examples.
    vector<int> random_permutation_index(num_total_rows_);
    if (myid_ == 0) {
      GeneratePermutation(num_total_rows_,
                          50,
                          0,
                          &random_permutation_index);
    }
    MPI_Bcast(&random_permutation_index[0],
              num_total_rows_,
              MPI_INT,
              0, MPI_COMM_WORLD);
    for (int i = 0; i < num_clusters_; ++i) {
      for (int j = 0; j < num_columns_; ++j) {
        double value;
        if (random_permutation_index[i] >= my_start_row_index_ &&
            random_permutation_index[i] < my_start_row_index_ +
                                          local_rows_.size()) {
          value = local_rows_[random_permutation_index[i] -
                  my_start_row_index_][j];
        } else {
          value = 0;
        }
        cluster_centers_[i][j] = value;
      }
      cluster_sizes_.push_back(0);
    }
    vector<double> cluster_centers_storage_backup(num_clusters_ * num_columns_);
    MPI_Allreduce(&cluster_centers_storage_[0],
                  &cluster_centers_storage_backup[0],
                  num_clusters_ * num_columns_,
                  MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);
    memcpy(&cluster_centers_storage_[0],
           &cluster_centers_storage_backup[0], num_clusters_ * num_columns_);

  } else {
    LOG(FATAL) << "No such an initialization method: "
               << kmeans_initialize_method << std::endl;
  }
}
void KMeans::GeneratePermutation(int length,
                                 int buf_size,
                                 int seed,
                                 vector<int> *perm) {
  // Initialize num_max;
  int num_max = buf_size;
  // A buffer for getting random numbers.
  vector<double> random_vector(num_max, 0);
  // Total number of random numbers needed.
  int len = length - 1;
  // How many to generate when we call Eval.
  int num = 0;
  // The array index into R.
  int num_i = 0;
  // First initialize the array "perm" to the identity permutation.
  for (int j = 0; j < length; ++j) {
    (*perm)[j] = j;
  }

  // Swap a random element in the front with the i'th element.
  for (int i = length - 1; i > 0; --i) {
    // Generate more random numbers.
    if (num_i == num) {
      num = (len < num_max) ? len : num_max;
      random_vector.clear();
      for (int j = 0; j < num; ++j) {
        double rand_value = RandDouble();
        random_vector.push_back(rand_value);
      }
      len -= num;
      num_i = 0;
    }
    // Pick a float in [0,i+1]. Truncate random_i to an integer.
    double random_i = (i + 1) * random_vector[num_i++];
    int k_i = static_cast<int>(random_i);
    if (k_i < i) {
      // Swap elements i and k.
      int temp = (*perm)[i];
      (*perm)[i] = (*perm)[k_i];
      (*perm)[k_i] = temp;
    }
  }
}
void KMeans::KMeansClustering(int kmeans_max_loop, double kmeans_threshold) {
  // Initialize new_cluster_size and new_cluster_centers.
  // They are also replicately stored.
  vector<double> new_cluster_centers_storage;
  new_cluster_centers_storage.resize(num_clusters_ * num_columns_, 0);
  vector<double*> new_cluster_centers(num_clusters_);
  vector<int> new_cluster_sizes(num_clusters_, 0);
  for (int i = 0; i < num_clusters_; ++i) {
    new_cluster_centers[i] = &(new_cluster_centers_storage[i * num_columns_]);
  }

  // Kmeans loss value.
  double sum_local_loss = 0.0;
  double sum_total_loss = 0.0;
  double sum_total_loss_old = 0.0;

  // delta is the loss difference and loop is the iteration number.
  // They are used to control the convergence of kmeans.
  double delta = 0.0;

  // The nearest cluster of a data point based on some distance measure.
  int nearest_cluster_index = 0;

  for (int i = 0; i < local_rows_.size(); ++i) {
    int value = 0;
    local_memberships_.push_back(value);
  }
  int loop = 1;
  do {
    if (myid_ == 0) {
      LOG(INFO) << "Iteration: " << loop << std::endl;
    }
    // Reset new clusters and sizes.
    for (int i = 0; i < num_clusters_; ++i) {
      for (int j = 0; j < num_columns_; ++j) {
        new_cluster_centers[i][j] = 0;
      }
      new_cluster_sizes[i] = 0;
    }
    sum_local_loss = 0.0;
    double min_distance = 0.0;
    for (int i = 0; i < local_rows_.size(); ++i) {
      // Find the array index of nestest cluster center.
      FindNearestCluster(i, &nearest_cluster_index, &min_distance);
      sum_local_loss += min_distance;
      // Update new cluster centers: sum of data points located within.
      new_cluster_sizes[nearest_cluster_index]++;
      for (int j = 0; j < num_columns_; ++j) {
        new_cluster_centers[nearest_cluster_index][j] +=
            local_rows_[i][j];
      }
      local_memberships_[i] = nearest_cluster_index;
    }
    // Sum all data points.
    MPI_Allreduce(&new_cluster_centers_storage[0],
                  &cluster_centers_storage_[0],
                  num_clusters_ * num_columns_,
                  MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(&new_cluster_sizes[0],
                  &cluster_sizes_[0],
                  num_clusters_,
                  MPI_INT,
                  MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&sum_local_loss,
                  &sum_total_loss,
                  1,
                  MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);

    // Compute the loss change.
    delta = fabs(sum_total_loss - sum_total_loss_old)
        / (sum_total_loss_old + 1e-10);
    sum_total_loss_old = sum_total_loss;

    // Average the sum to obtain new centers.
    int count_not_zero = 0;
    for (int i = 0; i < num_clusters_; ++i) {
      if (cluster_sizes_[i] > 0) {
        for (int j = 0; j < num_columns_; ++j) {
          cluster_centers_[i][j] /= cluster_sizes_[i];
        }
        count_not_zero++;
      }
    }
    if (myid_ == 0) {
      LOG(INFO) << "Number of non-empty clusters is " << count_not_zero << std::endl;
      LOG(INFO) << "Loss value is " << sum_total_loss << " and "
                << "Loss difference is " << delta << std::endl;
    }
  } while (delta > kmeans_threshold && loop++ < kmeans_max_loop);
}

void KMeans::FindNearestCluster(int data_point_index,
                                int *center_index,
                                double *min_distance) {
  double distance;
  *min_distance = FLT_MAX;
  *center_index = -1;
  // Find the cluster that has the minimal distance to the data point.
  for (int i = 0; i < cluster_centers_.size(); ++i) {
    distance = ComputeDistance(local_rows_[data_point_index],
                                      cluster_centers_[i]);
    if (distance < *min_distance) {
      *min_distance = distance;
      *center_index = i;
    }
  }
}

double KMeans::DotProduct(const vector<double>& v1, const double* v2) {
  double norm_sq = 0.0;
  for (int i = 0; i < v1.size(); ++i) {
    norm_sq += v1[i] * v2[i];
  }
  return norm_sq;
}
double KMeans::DotProduct(const vector<double>& v1, const vector<double>& v2) {
  double norm_sq = 0.0;
  for (int i = 0; i < v1.size(); ++i) {
    norm_sq += v1[i] * v2[i];
  }
  return norm_sq;
}
double KMeans::ComputeDistance(
    const vector<double> &data_point_1,
    const vector<double> &data_point_2) {
  CHECK(data_point_1.size() == data_point_2.size());
  double distance = 0;

  // Ignore sqrt, it's the same for k-means.
  for (int i = 0; i < data_point_1.size(); ++i) {
    distance += (data_point_1[i] - data_point_2[i])
      * (data_point_1[i] - data_point_2[i]);
  }
  return distance;
}
double KMeans::ComputeDistance(
    const vector<double> &data_point_1,
    const double* data_point_2) {
  double distance = 0;

  // Ignore sqrt, it's the same for k-means.
  for (int i = 0; i < data_point_1.size(); ++i) {
    distance += (data_point_1[i] - data_point_2[i])
      * (data_point_1[i] - data_point_2[i]);
  }
  return distance;
}

}  // namespace learning_psc
using namespace learning_psc;
using namespace std;
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int FLAGS_num_clusters = 0;
  int FLAGS_kmeans_loop  = 100;
  double FLAGS_kmeans_threshold = 1e-3;
  string FLAGS_initialization_method = "orthogonal_centers";
  string FLAGS_input = "";
  string FLAGS_output = "";
  for (int i = 1; i < argc; ++i) {
    if (0 == strcmp(argv[i], "--num_clusters")) {
      std::istringstream(argv[i+1]) >> FLAGS_num_clusters;
      ++i;
    } else if (0 == strcmp(argv[i], "--kmeans_loop")) {
      std::istringstream(argv[i+1]) >> FLAGS_kmeans_loop;
      ++i;
    } else if (0 == strcmp(argv[i], "--kmeans_threshold")) {
      std::istringstream(argv[i+1]) >> FLAGS_kmeans_threshold;
      ++i;
    } else if (0 == strcmp(argv[i], "--input")) {
      FLAGS_input = argv[i+1];
      ++i;
    } else if (0 == strcmp(argv[i], "--output")) {
      FLAGS_output = argv[i+1];
      ++i;
    } else if (0 == strcmp(argv[i], "--initialization_method")) {
      FLAGS_initialization_method = argv[i+1];
      ++i;
    }

  }
  if (FLAGS_num_clusters <= 0) {
    cerr << "--num_clusters must > 0" << endl;
    MPI_Finalize();
    return 1;
  }
  if (FLAGS_input == "") {
    cerr << "--input must not be empty" << endl;
    MPI_Finalize();
    return 1;
  }
  if (FLAGS_output == "") {
    cerr << "--output must not be empty" << endl;
    MPI_Finalize();
    return 1;
  }
  if (FLAGS_initialization_method != "orthogonal_centers" &&
      FLAGS_initialization_method != "random") {
    cerr << "--initialization_methond can only be 'orthogonal_centers' or 'random'" << endl;
    MPI_Finalize();
    return 1;
  }
  KMeans kmeans;
  kmeans.Read(FLAGS_input);
  kmeans.DoKMeans(FLAGS_num_clusters, FLAGS_initialization_method,
                  FLAGS_kmeans_loop, FLAGS_kmeans_threshold);
  kmeans.Write(FLAGS_output);
  MPI_Finalize();
  return 0;
}

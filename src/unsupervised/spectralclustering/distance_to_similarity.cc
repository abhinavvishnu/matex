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

#include <algorithm>
#include "common.h"
#include "distance_to_similarity.h"

namespace learning_psc {
DistanceToSimilarity::DistanceToSimilarity() {
  MPI_Comm_rank(MPI_COMM_WORLD, &myid_);
  MPI_Comm_size(MPI_COMM_WORLD, &pnum_);
}

void DistanceToSimilarity::ReadAndSetSymmetric(const string& filename) {
  ifstream fin2(filename.c_str());
  string line;
  int i = 0;
  while (getline(fin2, line)) {  // Each line is a training document.
    if (line.size() > 0 &&      // Skip empty lines.
        line[0] != '\r' &&      // Skip empty lines.
        line[0] != '\n' &&      // Skip empty lines.
        line[0] != '#') {       // Skip comment lines.
      istringstream ss(line);
      int index;
      double value;
      char colon;
      while (ss >> index >> colon >> value) {  // Load and init a document.
        if (index % pnum_ == myid_) {
          int local_index = index / pnum_;
          if (documents_.size() <= local_index) {
            documents_.resize(local_index + 1);
          }
          documents_[local_index][i] = value;
        }
        if (i % pnum_ == myid_) {
          int local_index = i / pnum_;
          if (documents_.size() <= local_index) {
            documents_.resize(local_index + 1);
          }
          documents_[local_index][index] = value;
        }
      }
      ++i;
    }
  }
  num_total_docs_ = i;
}

void DistanceToSimilarity::Compute() {
  // Parallely compute distance distance's row average.
  vector<double> row_avg_local(num_total_docs_, 0);
  vector<double> row_avg_global(num_total_docs_, 0);
  int global_index = myid_;
  for (int i = 0; i < documents_.size(); ++i, global_index += pnum_) {
    double sum = 0;
    for (map<int, double>::const_iterator iter = documents_[i].begin();
         iter != documents_[i].end(); ++iter) {
      sum += iter->second;
    }
    row_avg_local[global_index] = sum / documents_[i].size();
  }
  MPI_Allreduce(&row_avg_local[0], &row_avg_global[0], num_total_docs_,
                MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  // Convert distance to similarity.
  // similarity  = exp(- distance * distance / (2 * gamma1 * gamma2))
  // Here, we use row average for gamma1 and gamma2 (selftune gamma)
  global_index = myid_;
  for (int i = 0; i < documents_.size(); ++i, global_index += pnum_) {
    for (map<int, double>::iterator iter = documents_[i].begin();
         iter != documents_[i].end(); ++iter) {
      iter->second = exp(-iter->second * iter->second / 2 / row_avg_global[global_index] / row_avg_global[iter->first]);
    }
  }
  // Normalize to laplacian matrix
  // Firstly, parallely compute similarity matrix's row average.
  vector<double> row_sum_local(num_total_docs_, 0);
  vector<double> row_sum_global(num_total_docs_, 0);
  global_index = myid_;
  for (int i = 0; i < documents_.size(); ++i, global_index += pnum_) {
    double sum = 0;
    for (map<int, double>::const_iterator iter = documents_[i].begin();
         iter != documents_[i].end(); ++iter) {
      sum += iter->second;
    }
    row_sum_local[global_index] = sum;
  }
  MPI_Allreduce(&row_sum_local[0], &row_sum_global[0], num_total_docs_,
                MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  // S = D^{-1/2} S D^{-1/2}
  global_index = myid_;
  for (int i = 0; i < documents_.size(); ++i, global_index += pnum_) {
    for (map<int, double>::iterator iter = documents_[i].begin();
         iter != documents_[i].end(); ++iter) {
      iter->second = iter->second / sqrt(row_sum_global[global_index] * row_sum_global[iter->first]);
    }
  }
}

void DistanceToSimilarity::Write(const string& file) {
  for (int i = 0; i < num_total_docs_; ++i) {
    if (i % pnum_ == myid_) {
      int local_index = i / pnum_;
      std::ofstream fout;
      if (i == 0) {
        fout.open(file.c_str());
      } else {
        fout.open(file.c_str(), std::ios::app);
      }
      const map<int, double>& m = documents_[local_index];
      vector<int> sorted_index;
      for (map<int, double>::const_iterator iter = m.begin();
           iter != m.end(); ++iter) {
        sorted_index.push_back(iter->first);
      }
      ::std::sort(sorted_index.begin(), sorted_index.end());
      for (int j = 0; j < sorted_index.size(); ++j) {
        if (j != 0) {
          fout << " ";
        }
        fout << (*m.find(sorted_index[j])).first << ":" << (*m.find(sorted_index[j])).second;
      }
      fout << std::endl;
      fout.close();
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
}
}  // namespace learning_psc

using namespace learning_psc;
using namespace std;
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  string FLAGS_input = "";
  string FLAGS_output = "";
  for (int i = 1; i < argc; ++i) {
    if (0 == strcmp(argv[i], "--input")) {
      FLAGS_input = argv[i+1];
      ++i;
    } else if (0 == strcmp(argv[i], "--output")) {
      FLAGS_output = argv[i+1];
      ++i;
    }
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
  DistanceToSimilarity compute;
  compute.ReadAndSetSymmetric(FLAGS_input);
  compute.Compute();
  compute.Write(FLAGS_output);
  MPI_Finalize();
  return 0;
}

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

#ifndef _OPENSOURCE_PSC_COMPUTE_DISTANCE_H__
#define _OPENSOURCE_PSC_COMPUTE_DISTANCE_H__

#include "common.h"

namespace learning_psc {
// Document uses sparse representation.
// Suppose the feature vector is (0,2,5,0)
// iv: <<1,2>, <2,5>>
struct Document {
  vector<IndexValue> iv;
  double two_norm_sq;
  Document();
  void Encode(string* s);
  void Decode(const string& s);
  void ComputeTwoNorm();
};

struct CmpIndexValueByValueAsc {
  bool operator()(const IndexValue& p1,
                  const IndexValue& p2) {
    return p1.value < p2.value;
  }
};

struct CmpIndexValueByIndexAsc{
  bool operator()(const IndexValue& p1,
                  const IndexValue& p2) {
    return p1.index < p2.index;
  }
};

class ComputeDistance {
 public:
  ComputeDistance(int t_nearest_neighbor);
  // Read documents from a file. Parallely distributed.
  // Suppose there are 5 processes, my id is 3.
  // The documents with index 3, 8, 13 ... are stored on me.
  void Read(const string& file);
  // Compute the distance from my documents to all other documents.
  void ParallelCompute();
  // Write the sparse distance matrix to disk.
  void Write(const string& file);
 private:
  void BroadcastDocument(Document* doc, int root);
  double InnerProduct(const Document& doc1, const Document& doc2);
  vector<Document> docs_; // the documents stored on me
  int num_total_docs_; // the total number of documents
  int myid_; // my process id
  int pnum_; // the total no of processes
  int t_nearest_neighbor_;
  vector<vector<IndexValue> > distance_;
};
}
#endif  // _OPENSOURCE_PSC_COMPUTE_DISTANCE_H__

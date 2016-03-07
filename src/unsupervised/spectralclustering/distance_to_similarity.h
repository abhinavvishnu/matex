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

#ifndef _OPENSOURCE_PSC_DISTANCE_TO_SIMILARITY_H__
#define _OPENSOURCE_PSC_DISTANCE_TO_SIMILARITY_H__

#include "common.h"

namespace learning_psc {

class DistanceToSimilarity {
 public:
  DistanceToSimilarity();
  void ReadAndSetSymmetric(const string& filename);
  void Compute();
  void Write(const string& file);
 private:
  vector<map<int, double> > documents_;
  int num_total_docs_;
  int myid_; // my process id
  int pnum_; // the total no of processes
};
}

#endif  // _OPENSOURCE_PSC_DISTANCE_TO_SIMILARITY_H__

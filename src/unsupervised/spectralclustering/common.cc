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

#include "common.h"

char kSegmentFaultCauser[] = "Used to cause artificial segmentation fault";

namespace learning_psc {

int GetAccumulativeSample(const vector<double>& distribution) {
  double distribution_sum = 0.0;
  for (int i = 0; i < distribution.size(); ++i) {
    distribution_sum += distribution[i];
  }

  double choice = RandDouble() * distribution_sum;
  double sum_so_far = 0.0;
  for (int i = 0; i < distribution.size(); ++i) {
    sum_so_far += distribution[i];
    if (sum_so_far >= choice) {
      return i;
    }
  }

  LOG(FATAL) << "Failed to choose element from distribution of size "
             << distribution.size() << " and sum " << distribution_sum;

  return -1;
}

std::ostream& operator << (std::ostream& out, vector<double>& v) {
  for (size_t i = 0; i < v.size(); ++i) {
    out << v[i] << " ";
  }
  return out;
}

void IntToString(int i32, string* key) {
  int original_size = key->size();
  key->resize(original_size + sizeof(i32));
  for (int i = sizeof(i32) - 1; i >= 0; --i) {
    (*key)[original_size + i] = i32 & 0xff;
    i32  = (i32 >> 8);
  }
}
int StringToInt(const char* key, int size) {
  CHECK(size == sizeof(int32));
  int i32 = 0;
  for (int i = 0; i < sizeof(i32); ++i) {
    i32 = (i32 << 8);
    i32 = i32 | static_cast<unsigned char>(key[i]);
  }
  return i32;
}
void Int64ToString(int64 i64, string* key) {
  int original_size = key->size();
  key->resize(original_size + sizeof(i64));
  for (int i = sizeof(i64) - 1; i >= 0; --i) {
    (*key)[original_size + i] = i64 & 0xff;
    i64  = (i64 >> 8);
  }
}
int64 StringToInt64(const char* key, int size) {
  CHECK(size == sizeof(int64));
  int64 i64 = 0;
  for (int i = 0; i < sizeof(i64); ++i) {
    i64 = (i64 << 8);
    i64 = i64 | static_cast<unsigned char>(key[i]);
  }
  return i64;
}
void DoubleToString(double d, string* key) {
  int64 l;
  memcpy(&l, &d, sizeof(d));
  Int64ToString(l, key);
}
double StringToDouble(const char* key, int size) {
  int64 l = StringToInt64(key, size);
  double d;
  memcpy(&d, &l, sizeof(d));
  return d;
}
}  // namespace learning_psc

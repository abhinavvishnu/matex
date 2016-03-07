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

#ifndef _OPENSOURCE_PSC_COMMON_H__
#define _OPENSOURCE_PSC_COMMON_H__

#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <list>
#include <map>
#include <string>
#include <vector>
#include <cmath>


// The CHECK_xxxx facilities, which generates a segmentation fault
// when a check is failed.  If the program is run within a debugger,
// the segmentation fault makes the debugger keeps track of the stack,
// which provides the context of the fail.
//
extern char kSegmentFaultCauser[];

#define CHECK(a) if (!(a)) {                                            \
    std::cerr << "CHECK failed "                                        \
              << __FILE__ << ":" << __LINE__ << "\n"                    \
              << #a << " = " << (a) << "\n";                            \
    *kSegmentFaultCauser = '\0';                                        \
  }                                                                     \

#define CHECK_EQ(a, b) if (!((a) == (b))) {                             \
    std::cerr << "CHECK_EQ failed "                                     \
              << __FILE__ << ":" << __LINE__ << "\n"                    \
              << #a << " = " << (a) << "\n"                             \
              << #b << " = " << (b) << "\n";                            \
    *kSegmentFaultCauser = '\0';                                        \
  }                                                                     \

#define CHECK_GT(a, b) if (!((a) > (b))) {                              \
    std::cerr << "CHECK_GT failed "                                     \
              << __FILE__ << ":" << __LINE__ << "\n"                    \
              << #a << " = " << (a) << "\n"                             \
              << #b << " = " << (b) << "\n";                            \
    *kSegmentFaultCauser = '\0';                                        \
  }                                                                     \

#define CHECK_LT(a, b) if (!((a) < (b))) {                              \
    std::cerr << "CHECK_LT failed "                                     \
              << __FILE__ << ":" << __LINE__ << "\n"                    \
              << #a << " = " << (a) << "\n"                             \
              << #b << " = " << (b) << "\n";                            \
    *kSegmentFaultCauser = '\0';                                        \
  }                                                                     \

#define CHECK_GE(a, b) if (!((a) >= (b))) {                             \
    std::cerr << "CHECK_GE failed "                                     \
              << __FILE__ << ":" << __LINE__ << "\n"                    \
              << #a << " = " << (a) << "\n"                             \
              << #b << " = " << (b) << "\n";                            \
    *kSegmentFaultCauser = '\0';                                        \
  }                                                                     \

#define CHECK_LE(a, b) if (!((a) <= (b))) {             \
    std::cerr << "CHECK_LE failed "                     \
              << __FILE__ << ":" << __LINE__ << "\n"    \
              << #a << " = " << (a) << "\n"             \
              << #b << " = " << (b) << "\n";            \
    *kSegmentFaultCauser = '\0';                        \
  }                                                     \
                                                        \


// The log facility, which makes it easy to leave of trace of your
// program.  The logs are classified according to their severity
// levels.  Logs of the level FATAL will cause a segmentation fault,
// which makes the debugger to keep track of the stack.
//
// Examples:
//   LOG(INFO) << iteration << "-th iteration ...";
//   LOG(FATAL) << "Probability value < 0 " << prob_value;
//
enum LogSeverity { INFO, WARNING, ERROR, FATAL };

class Logger {
 public:
  Logger(LogSeverity ls, const std::string& file, int line)
      : ls_(ls), file_(file), line_(line)
  {}
  std::ostream& stream() const {
    return std::cerr << file_ << " (" << line_ << ") : ";
  }
  ~Logger() {
    if (ls_ == FATAL) {
      *::kSegmentFaultCauser = '\0';
    }
  }
 private:
  LogSeverity ls_;
  std::string file_;
  int line_;
};

#define LOG(ls) Logger(ls, __FILE__, __LINE__).stream()

// Basis POD types.
typedef int                 int32;
#ifdef COMPILER_MSVC
typedef __int64             int64;
#else
typedef long long           int64;
#endif

// Frequently-used STL containers.
using std::list;
using std::map;
using std::string;
using std::vector;
using std::string;
using std::istringstream;
using std::ifstream;
using std::pair;
using std::sqrt;

namespace learning_psc {

struct IndexValue {
  int index;
  double value;
  IndexValue() {}
  IndexValue(int i, double v) : index(i), value(v) {
  }
};
// Generate a random float value in the range of [0,1) from the
// uniform distribution.
inline double RandDouble() {
  return rand() / static_cast<double>(RAND_MAX);
}

// Generate a random integer value in the range of [0,bound) from the
// uniform distribution.
inline int RandInt(int bound) {
  // NOTE: Do NOT use rand() % bound, which does not approximate a
  // discrete uniform distribution will.
  return static_cast<int>(RandDouble() * bound);
}

// Steaming output facilities for GSL matrix, GSL vector and STL
// vector.
std::ostream& operator << (std::ostream& out, vector<double>& v);


// Convert a int/double to its string form 4/8 bytes.
// The string form is not readable and could be converted back to int/double
// without losing any information.
void IntToString(int i32, string* key);
int StringToInt(const char* key, int size);
void Int64ToString(int64 i64, string* key);
int64 StringToInt64(const char* key, int size);

void DoubleToString(double d, string* key);
double StringToDouble(const char* key, int size);

// A maximum heap to store only Top N maximum elements of inserted elements.
// struct Cmp {bool operator()(double a, double b) {return a > b;}}
// TopN n(2); n.Insert(5); n.Insert(3); n.Insert(2);
// vector<double> v; n.Extract(&v);
// CHECK_EQ(3, v[0]);
// CHECK_EQ(5, v[1]);
template <class T, class Cmp>
class TopN {
 public:
  TopN(int n);
  void Insert(const T& element);
  void Extract(vector<T>* result);
 private:
  int n_;
  vector<T> elements_;
  Cmp cmp_;
};
template <class T, class Cmp>
TopN<T, Cmp>::TopN(int n) : n_(n), cmp_(Cmp()) {
}

template <class T, class Cmp>
void TopN<T, Cmp>::Insert(const T& element) {
  if (elements_.size() == n_ && !cmp_(element, elements_.front())) {
    return;
  }
  elements_.push_back(element);
  push_heap(elements_.begin(), elements_.end(), cmp_);
  if (elements_.size() > n_) {
    pop_heap(elements_.begin(), elements_.end(), cmp_);
    elements_.pop_back();
  }
}
template <class T, class Cmp>
void TopN<T, Cmp>::Extract(vector<T>* result) {
  int size = elements_.size();
  for (int i = 0; i < size; ++i) {
    pop_heap(elements_.begin(), elements_.end(), cmp_);
    result->push_back(elements_.back());
    elements_.pop_back();
  }
}

}  // namespace learning_psc

#endif  // _OPENSOURCE_PSC_COMMON_H__

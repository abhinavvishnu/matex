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
#include "evd.h"
//#include "f2c/f2c.h"

#if 0
struct a {
  long int mpi_bottom__, mpi_integer__, mpi_real__, mpi_double_precision__,
           mpi_complex__, mpi_double_complex__, mpi_logical__,
           mpi_character__, mpi_byte__, mpi_2integer__, mpi_2real__,
           mpi_2double_precision__, mpi_2complex__, mpi_2double_complex__,
           mpi_integer1__, mpi_integer2__, mpi_integer4__, mpi_real2__,
           mpi_real4__, mpi_real8__, mpi_ub__, mpi_lb__, mpi_comm_world__,
           mpi_comm_self__, mpi_group_empty__, mpi_sum__, mpi_max__,
           mpi_min__, mpi_prod__, mpi_land__, mpi_band__, mpi_lor__,
           mpi_bor__, mpi_lxor__, mpi_bxor__, mpi_minloc__, mpi_maxloc__,
           mpi_op_null__, mpi_tag_ub__, mpi_host__, mpi_io__,
           mpi_errors_are_fatal__, mpi_errors_return__, mpi_packed__;
} mpipriv_ = {
  0, MPI_INT, MPI_FLOAT, MPI_DOUBLE,
  0, 0, 0,
  MPI_BYTE, MPI_BYTE, 0, 0,
  0, 0, 0,
  0, 0, 0, 0,
  0, 0, 0, 0, 0,
  0, 0, MPI_SUM, MPI_MAX,
  MPI_MIN, 0, 0, 0, 0,
  0, 0, 0, 0, 0,
  0, 0, 0, 0,
  0, 0, 0
};
#endif


extern "C" {
#if 0
  extern int pdsaupd_(integer *comm, integer *ido, char *bmat,
      integer *n, char *which, integer *nev, doublereal *tol, doublereal *
      resid, integer *ncv, doublereal *v, integer *ldv, integer *iparam,
      integer *ipntr, doublereal *workd, doublereal *workl, integer *lworkl,
      integer *info, ftnlen bmat_len, ftnlen which_len);
  extern int pdseupd_(integer *comm, logical *rvec, char *howmny,
      logical *select, doublereal *d__, doublereal *z__, integer *ldz,
      doublereal *sigma, char *bmat, integer *n, char *which, integer *nev,
      doublereal *tol, doublereal *resid, integer *ncv, doublereal *v,
      integer *ldv, integer *iparam, integer *ipntr, doublereal *workd,
      doublereal *workl, integer *lworkl, integer *info, ftnlen howmny_len,
      ftnlen bmat_len, ftnlen which_len);
#endif

  // For logging configuration
  extern struct {
    long int logfil, ndigit, mgetv0, msaupd, msaup2, msaitr, mseigt, msapps,
             msgets, mseupd, mnaupd, mnaup2, mnaitr, mneigh, mnapps, mngets,
             mneupd, mcaupd, mcaup2, mcaitr, mceigh, mcapps, mcgets, mceupd;
  } debug_;
}
#if 0
extern "C" int mpi_allreduce__(doublereal* s, doublereal* r,
integer* c, integer* type, integer* op, integer* comm, integer* err) {
  return MPI_Allreduce(s, r, *c, *type, *op, MPI_COMM_WORLD);
}
#endif

#if 0
extern "C" int mpi_comm_rank__(integer* comm, integer* myid,
                               integer* error) {
  int id;
  int result = MPI_Comm_rank(MPI_COMM_WORLD, &id);
  *myid = id;
  return result;
}
#endif
namespace learning_psc {
EigenSolverSymmetric::EigenSolverSymmetric(int num_local_row, string eigen_type,
                         int eigen_desire, int eigen_space)
    : num_local_row_(num_local_row),
      eigen_space_(eigen_space),
      eigen_desire_(eigen_desire),
      max_iterations_(300),
      tolerance_(0),
      eigen_type_(eigen_type) {
  // alocate space
  int ncv = eigen_space;
  int mloc = num_local_row;
  int lworkl = ncv * (ncv + 8);
  select_ = new long int[ncv];
  resid_ = new double[mloc];
  memset(resid_, 1, sizeof(*resid_) * mloc);
  v_ = new double[mloc * ncv];
  workd_ = new double[3 * mloc];
  workl_ = new double[lworkl];
  d_ = new double[2 * ncv];
  // logging:voluminous.
  debug_.logfil = 6;
  debug_.ndigit = -3;
  debug_.mgetv0 = 0;
  debug_.msaupd = 1;
  debug_.msaup2 = 0;
  debug_.msaitr = 0;
  debug_.mseigt = 0;
  debug_.msapps = 0;
  debug_.msgets = 0;
  debug_.mseupd = 0;
}

EigenSolverSymmetric::~EigenSolverSymmetric() {
  delete[] select_;
  delete[] resid_;
  delete[] v_;
  delete[] workd_;
  delete[] workl_;
  delete[] d_;
}

// Please refer to ARPACK user manual for the following flow.
// You must be quite familiar with ARPACK to understand it.
template<class T>
void EigenSolverSymmetric::Solve(T& av,
                                 int* eigen_got, double** eigen_value,
                                 double** eigen_vector) {
  long int mloc = num_local_row_;
  long int ncv = eigen_space_;
  long int nev = eigen_desire_;
  long int lworkl = ncv * (ncv + 8);
  double sigma = 0;
  long int ipntr[11];
  long int iparam[11];
  char bmat = 'I';
  char which[5];
  strncpy(which, eigen_type_.c_str(), eigen_type_.size() + 1);
  long int ido = 0;
  double tol = tolerance_;
  long int info = 0;
  long int ishfts = 1;
  long int maxitr = max_iterations_;
  long int mode   = 1;
  long int kTrue = 1;
  char kA[] = "A";
  iparam[0] = ishfts;
  iparam[2] = maxitr;
  iparam[6] = mode;
  // The counts of computation: y = Ax.
  int opx_count = 0;
  int myid;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  // The StaticLocalVariable are originally pdsaupd and pdsaup2's static local
  // variables. They preverve the computing state. To enable checkpoint,
  // we now make them parameters.
  if (myid == 0) {
    LOG(INFO) << "Calling pdsaupd (computing eigenvalues)..." << std::endl;
  }
  while (true) {
#if 0
    pdsaupd_((long int*)MPI_COMM_WORLD,
               &ido, &bmat, &mloc, which, &nev, &tol, resid_,
               &ncv, v_, &mloc, iparam, ipntr, workd_, workl_, &lworkl,
               &info, 0, 0);
#endif
    // An error code less than zero indicates that the input parameters were
    // malformed or some unexpected error occurs.
    if (info < 0) {
      switch (info) {
        case -1:
          LOG(FATAL) << "N must be positive.";
          break;
        case -2:
          LOG(FATAL) << "NEV must be positive.";
          break;
        case -3:
          LOG(FATAL) << "NCV must be greater than NEV and less than or equal "
                     << "to N.";
          break;
        case -4:
          LOG(FATAL) << "The maximum number of Arnoldi update iterations "
                     << "allowed must be greater than zero.";
          break;
        case -5:
          LOG(FATAL) << "WHICH must be one of 'LM', 'SM', 'LA', 'SA' or 'BE'.";
          break;
        case -6:
          LOG(FATAL) << "BMAT must be one of 'I' or 'G'.";
          break;
        case -7:
          LOG(FATAL) << "Length of private work array WORKL is not sufficient.";
          break;
        case -8:
          LOG(FATAL) << "Error return from trid. eigenvalue calculation."
                     << "Informatinal error from LAPACK routine dsteqr.";
          break;
        case -9:
          LOG(FATAL) << "Starting vector is zero.";
          break;
        case -10:
          LOG(FATAL) << "IPARAM(7) must be 1,2,3,4,5.";
          break;
        case -11:
          LOG(FATAL) << "IPARAM(7) = 1 and BMAT = 'G' are incompatable.";
          break;
        case -12:
          LOG(FATAL) << "IPARAM(1) must be equal to 0 or 1.";
          break;
        case -13:
          LOG(FATAL) << "NEV and WHICH = 'BE' are incompatable.";
          break;
        case -9999:
          LOG(FATAL) << "Could not build an Arnoldi factorization."
                     << "IPARAM(5) returns the size of the current Arnoldi"
                     << "factorization. The user is advised to check that"
                     << "enough workspace and array storage has been "
                     << "allocated.";
          break;
        default:
          LOG(FATAL) << "unrecognised return value";
      }
    }
    if (info > 0) {
      switch (info) {
        case 1:
          LOG(ERROR) << "Maximum number of iterations taken.";
          break;
        case 3:
          LOG(FATAL) << "No shifts could be applied during a cycle of the "
                     << "Implicitly restarted Arnoldi iteration. "
                     << "One possibility is to increase the size of NCV "
                     << "relative to NEV.";
          break;
      }
    }
    if (ido == -1 || ido == 1) {
      av(workd_ + ipntr[0] - 1, workd_ + ipntr[1] - 1);
      ++opx_count;
      if (opx_count % 100 == 0 && myid == 0) {
        LOG(INFO) << "Number of OP*x operations:" << opx_count << std::endl;
      }
    } else {
      break;
    }
  }
  if (myid == 0) {
    LOG(INFO) << "Calling pdsaupd done.(Eigenvalues are ready)" << std::endl;
    LOG(INFO) << "Calling pdseupd (computing eigenvectors)..." << std::endl;
  }
#if 0
  pdseupd_((long int*)MPI_COMM_WORLD,
          &kTrue, kA, select_, d_, v_, &mloc, &sigma,
          &bmat, &mloc, which, &nev,
          &tol, resid_, &ncv, v_, &mloc, iparam, ipntr, workd_,
          workl_, &lworkl, &info, 0, 0, 0);
#endif
  if (myid == 0) {
    LOG(INFO) << "Calling pdseupd done.(Eigenvectors are ready)" << std::endl;
  }
  // An error code less than zero indicates that the input parameters were
  // malformed.
  CHECK_GE(info, 0);
  if (info > 0) {
    LOG(ERROR) << "Error code in pdseupd: " << info << std::endl;
  }
  long int nconv = iparam[4];
  /*
   * The following is to tell how accurate each eigenvalue is. This evaluation
   * cost too much time so we remove it.
  for (int j = 1; j <= nconv; ++j) {
    av_->Run(v_ + (j - 1) * mloc, ax_);
    daxpy(mloc, -d_[j - 1], v_ + (j - 1) * mloc, 1, ax_, 1);
    d_[j - 1 + ncv] = pdnorm2(mloc, ax_, 1);
  }
  */
  *eigen_value = d_;
  *eigen_vector = v_;
  *eigen_got = nconv;
}

Evd::Evd()
    : num_total_rows_(0), num_local_rows_(0),
      my_start_row_index_(0), solver_(NULL),
      num_converged_(0),
      eigen_values_(NULL), eigen_vectors_(NULL) {
  MPI_Comm_rank(MPI_COMM_WORLD, &myid_);
  MPI_Comm_size(MPI_COMM_WORLD, &pnum_);
}
Evd::~Evd() {
  delete solver_;
}

void Evd::Read(const string& filename) {
  // Master read the total number of rows.
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
  row_count_max_ = -1;
  for (int i = 0; i < pnum_; ++i) {
    row_count_of_each_proc_[i] = num_total_rows_ / pnum_ + (num_total_rows_ % pnum_ > i ? 1 : 0);
    if (row_count_of_each_proc_[i] > row_count_max_) {
      row_count_max_ = row_count_of_each_proc_[i];
    }
  }
  num_local_rows_ = row_count_of_each_proc_[myid_];
  my_start_row_index_ = 0;
  for (int i = 0; i < myid_; ++i) {
    my_start_row_index_ += row_count_of_each_proc_[i];
  }
  // Read my part
  ifstream fin2(filename.c_str());
  string line;
  int i = 0;
  int j = 0;
  while (getline(fin2, line)) {  // Each line is a training document.
    if (line.size() > 0 &&      // Skip empty lines.
        line[0] != '\r' &&      // Skip empty lines.
        line[0] != '\n' &&      // Skip empty lines.
        line[0] != '#') {       // Skip comment lines.
      if (i < my_start_row_index_ || i >= my_start_row_index_ + num_local_rows_) {
        ++i;
        continue;
      }
      istringstream ss(line);
      int index;
      double value;
      char colon;
      while (ss >> index >> colon >> value) {  // Load and init a document.
        AppendItem(j, index,
                   value);
      }
      ++j;
      ++i;
    }
  }
}
void Evd::Compute(int num_eigen, int eigen_space,
                  int max_iterations, double tolerance) {
  CHECK_GT(num_total_rows_, num_eigen);
  CHECK_GE(num_total_rows_, eigen_space);
  solver_ =
      new EigenSolverSymmetric(num_local_rows_,
                               "LA",
                               num_eigen,
                               eigen_space);

  // Do EVD.
  solver_->set_max_iterations(max_iterations);
  solver_->set_tolerance(tolerance);
  solver_->Solve(*this,
                 &num_converged_,
                 &eigen_values_,
                 &eigen_vectors_);
}
void Evd::operator()(double *x, double *y) const {
  // buff is the buffer to be gathered
  // buff2 is the buffer to store the allgather result.
  // completex is the complete x.
  double* buff = new double[row_count_max_];
  memset(buff, 0, sizeof(buff[0]) * row_count_max_);
  double* buff2 = new double[pnum_ * row_count_max_];
  double* completex = new double[num_total_rows_];
  for (int i = 0; i < num_local_rows_; ++i) {
    buff[i] = x[i];
  }
  MPI_Allgather(buff, row_count_max_, MPI_DOUBLE,
                buff2, row_count_max_, MPI_DOUBLE, MPI_COMM_WORLD);
  int index = 0;
  for (int i = 0; i < pnum_; ++i) {
    for (int j = 0; j < row_count_of_each_proc_[i]; ++j) {
      completex[index++] = buff2[i * row_count_max_ + j];
    }
  }
  for (int i = 0; i < row_start_index_.size(); ++i) {
    double sum = 0;
    int upper_bound = (i == row_start_index_.size() - 1 ?
                       column_index_.size(): row_start_index_[i+1]);
    for (int j = row_start_index_[i]; j < upper_bound; ++j) {
      sum += value_[j] * completex[column_index_[j]];
    }
    y[i] = sum;
  }
  delete[] buff;
  delete[] buff2;
  delete[] completex;
  return;
}

void Evd::AppendItem(int local_row_index, int column, double value) {
  column_index_.push_back(column);
  value_.push_back(value);
  row_start_index_.resize(local_row_index + 1, -1);
  if (row_start_index_[local_row_index] < 0) {
    row_start_index_[local_row_index] = value_.size() - 1;
  }
  int x = local_row_index - 1;
  while (x >= 0 && row_start_index_[x] < 0) {
    row_start_index_[x] = value_.size() - 1;
    --x;
  }
}

void Evd::Write(const string& eigenvaluefile,
                const string& eigenvectorfile) {
  // Writing eigen values.
  if (myid_ == 0) {
    std::ofstream fout(eigenvaluefile.c_str());
    for (int j = 0; j < num_converged_; ++j) {
      fout << eigen_values_[j] << std::endl;
    }
    fout.close();
  }
  for (int p = 0; p < pnum_; ++p) {
    if (p == myid_) {
      std::ofstream fout;
      if (p == 0) {
        fout.open(eigenvectorfile.c_str());
      } else {
        fout.open(eigenvectorfile.c_str(), std::ios::app);
      }
      for (int i = 0; i < num_local_rows_; ++i) {
        for (int j = 0; j < num_converged_; ++j) {
          if (j != 0) fout << " ";
          fout << eigen_vectors_[j * num_local_rows_ + i];
        }
        fout << std::endl;
      }
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
  int FLAGS_eigenvalue = 0;
  int FLAGS_eigenspace = 0;
  int FLAGS_arpack_iterations = 300;
  double FLAGS_arpack_tolerance = 0.0;
  string FLAGS_input = "";
  string FLAGS_eigenvalues_output = "";
  string FLAGS_eigenvectors_output = "";
  for (int i = 1; i < argc; ++i) {
    if (0 == strcmp(argv[i], "--eigenvalue")) {
      std::istringstream(argv[i+1]) >> FLAGS_eigenvalue;
      ++i;
    } else if (0 == strcmp(argv[i], "--eigenspace")) {
      std::istringstream(argv[i+1]) >> FLAGS_eigenspace;
      ++i;
    } else if (0 == strcmp(argv[i], "--arpack_iterations")) {
      std::istringstream(argv[i+1]) >> FLAGS_arpack_iterations;
      ++i;
    } else if (0 == strcmp(argv[i], "--arpack_tolerance")) {
      std::istringstream(argv[i+1]) >> FLAGS_arpack_tolerance;
      ++i;
    } else if (0 == strcmp(argv[i], "--input")) {
      FLAGS_input = argv[i+1];
      ++i;
    } else if (0 == strcmp(argv[i], "--eigenvalues_output")) {
      FLAGS_eigenvalues_output = argv[i+1];
      ++i;
    } else if (0 == strcmp(argv[i], "--eigenvectors_output")) {
      FLAGS_eigenvectors_output = argv[i+1];
      ++i;
    }

  }
  if (FLAGS_eigenvalue <= 0) {
    cerr << "--eigenvalue must > 0" << endl;
    MPI_Finalize();
    return 1;
  }
  if (FLAGS_eigenspace <= FLAGS_eigenvalue) {
    cerr << "--eigenspace must > eigenvalue" << endl;
    MPI_Finalize();
    return 1;
  }
  if (FLAGS_input == "") {
    cerr << "--input must not be empty" << endl;
    MPI_Finalize();
    return 1;
  }
  if (FLAGS_eigenvalues_output == "") {
    cerr << "--eigenvalues_output must not be empty" << endl;
    MPI_Finalize();
    return 1;
  }
  if (FLAGS_eigenvectors_output == "") {
    cerr << "--eigenvectors_output must not be empty" << endl;
    MPI_Finalize();
    return 1;
  }
  Evd evd;
  evd.Read(FLAGS_input);
  evd.Compute(FLAGS_eigenvalue, FLAGS_eigenspace, FLAGS_arpack_iterations, FLAGS_arpack_tolerance);
  evd.Write(FLAGS_eigenvalues_output, FLAGS_eigenvectors_output);
  MPI_Finalize();
  return 0;
}

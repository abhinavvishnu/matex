ACLOCAL_AMFLAGS = -I m4

bin_PROGRAMS =
noinst_PROGRAMS =
check_PROGRAMS =
noinst_LIBRARIES =

EXTRA_DIST =
BUILT_SOURCES =
CLEANFILES =
DISTCLEANFILES =
MAINTAINERCLEANFILES =
MOSTLYCLEANFILES =

AM_FFLAGS =
AM_CFLAGS =
AM_CXXFLAGS =
AM_CPPFLAGS =
AM_LDFLAGS = -lm
LDADD =

AM_CPPFLAGS += $(MPI_CPPFLAGS)

AM_LDFLAGS += $(MPI_LDFLAGS)

LDADD += $(MPI_LIBS) -lm

##############################################################################
# contrib into dist
##############################################################################
EXTRA_DIST += contrib/ga-5-3
EXTRA_DIST += contrib/mpich-3.1
EXTRA_DIST += build.sh
EXTRA_DIST += LICENSE
dist-hook:
	rm -rf `find $(distdir)/contrib/ga-5-3 -type d -name .svn`
	rm -rf `find $(distdir)/contrib/mpich-3.1 -type d -name .svn`

##############################################################################
# contrib/ARPACK
##############################################################################
include contrib/ARPACK/Makefile.inc

##############################################################################
# src/supervised/svm
##############################################################################
bin_PROGRAMS += matex_svm
matex_svm_SOURCES = 
matex_svm_SOURCES += src/supervised/svm/comm.cpp
matex_svm_SOURCES += src/supervised/svm/comm.hpp
matex_svm_SOURCES += src/supervised/svm/dataset.cpp
matex_svm_SOURCES += src/supervised/svm/dataset.hpp
matex_svm_SOURCES += src/supervised/svm/distance.cpp
matex_svm_SOURCES += src/supervised/svm/distance.hpp
matex_svm_SOURCES += src/supervised/svm/driver.cpp
matex_svm_SOURCES += src/supervised/svm/smo.cpp
matex_svm_SOURCES += src/supervised/svm/smo.hpp
matex_svm_SOURCES += src/supervised/svm/testing.cpp

matex_svm_CPPFLAGS =
matex_svm_CPPFLAGS += $(AM_CPPFLAGS)
matex_svm_LDFLAGS =
matex_svm_LDFLAGS += $(AM_LDFLAGS)
matex_svm_LDADD =
matex_svm_LDADD += $(LDADD)
##############################################################################
# src/supervised/smo
##############################################################################
#bin_PROGRAMS += matex_smo
#matex_smo_SOURCES = 
#matex_smo_SOURCES += src/supervised/smo/smo.cpp
#matex_smo_CPPFLAGS =
#matex_smo_CPPFLAGS += $(AM_CPPFLAGS)
#matex_smo_CPPFLAGS += $(GA_CPPFLAGS)
#matex_smo_LDFLAGS =
#matex_smo_LDFLAGS += $(AM_LDFLAGS)
#matex_smo_LDFLAGS += $(GA_LDFLAGS)
#matex_smo_LDADD =
#matex_smo_LDADD += $(GA_LIBS)
#matex_smo_LDADD += $(GA_FLIBS)
#matex_smo_LDADD += $(LDADD) 

##############################################################################
# src/supervised/knn
##############################################################################
bin_PROGRAMS += matex_knn
matex_knn_SOURCES = 
matex_knn_SOURCES += src/supervised/knn/comm.cpp
matex_knn_SOURCES += src/supervised/knn/comm.hpp
matex_knn_SOURCES += src/supervised/knn/dataset.cpp
matex_knn_SOURCES += src/supervised/knn/dataset.hpp
matex_knn_SOURCES += src/supervised/knn/distance.cpp
matex_knn_SOURCES += src/supervised/knn/distance.hpp
matex_knn_SOURCES += src/supervised/knn/driver.cpp
matex_knn_SOURCES += src/supervised/knn/knn.cpp
matex_knn_SOURCES += src/supervised/knn/knn.hpp
matex_knn_CPPFLAGS =
matex_knn_CPPFLAGS += $(AM_CPPFLAGS)
matex_knn_LDFLAGS =
matex_knn_LDFLAGS += $(AM_LDFLAGS)
matex_knn_LDADD =
matex_knn_LDADD += $(LDADD)


##############################################################################
# src/assoc_rules/fpgrowth
##############################################################################
bin_PROGRAMS += matex_fpgrowth
matex_fpgrowth_SOURCES = 
matex_fpgrowth_SOURCES += src/assoc_rules/comm.cpp
matex_fpgrowth_SOURCES += src/assoc_rules/comm.hpp
matex_fpgrowth_SOURCES += src/assoc_rules/dataset.cpp
matex_fpgrowth_SOURCES += src/assoc_rules/dataset.hpp
matex_fpgrowth_SOURCES += src/assoc_rules/distance.cpp
matex_fpgrowth_SOURCES += src/assoc_rules/distance.hpp
matex_fpgrowth_SOURCES += src/assoc_rules/driver.cpp
matex_fpgrowth_SOURCES += src/assoc_rules/fpgrowth.cpp
matex_fpgrowth_SOURCES += src/assoc_rules/fpgrowth.hpp
matex_fpgrowth_SOURCES += src/assoc_rules/fpheader.h
matex_fpgrowth_SOURCES += src/assoc_rules/fpstack.h
matex_fpgrowth_SOURCES += src/assoc_rules/merge_prune.cpp
matex_fpgrowth_CPPFLAGS =
matex_fpgrowth_CPPFLAGS += $(AM_CPPFLAGS)
matex_fpgrowth_LDFLAGS =
matex_fpgrowth_LDFLAGS += $(AM_LDFLAGS)
matex_fpgrowth_LDADD =
matex_fpgrowth_LDADD += $(LDADD)

##############################################################################
# src/unsupervised/kmeans
##############################################################################
bin_PROGRAMS += matex_kmeans
matex_kmeans_SOURCES =
matex_kmeans_SOURCES += src/unsupervised/kmeans/comm.cpp
matex_kmeans_SOURCES += src/unsupervised/kmeans/comm.hpp
matex_kmeans_SOURCES += src/unsupervised/kmeans/dataset.cpp
matex_kmeans_SOURCES += src/unsupervised/kmeans/dataset.hpp
matex_kmeans_SOURCES += src/unsupervised/kmeans/distance.cpp
matex_kmeans_SOURCES += src/unsupervised/kmeans/distance.hpp
matex_kmeans_SOURCES += src/unsupervised/kmeans/driver.cpp
matex_kmeans_SOURCES += src/unsupervised/kmeans/kmeans.cpp
matex_kmeans_SOURCES += src/unsupervised/kmeans/kmeans.hpp
matex_kmeans_CPPFLAGS =
matex_kmeans_CPPFLAGS += $(GA_CPPFLAGS)
matex_kmeans_CPPFLAGS += $(AM_CPPFLAGS)
matex_kmeans_LDFLAGS =
matex_kmeans_LDFLAGS += $(GA_LDFLAGS)
matex_kmeans_LDFLAGS += $(AM_LDFLAGS)
matex_kmeans_LDADD =
matex_kmeans_LDADD += $(GA_LIBS)
matex_kmeans_LDADD += $(LDADD)

##############################################################################
# src/unsupervised/spectralclustering
##############################################################################
bin_PROGRAMS += matex_compute_distance
matex_compute_distance_SOURCES =
matex_compute_distance_SOURCES += src/unsupervised/spectralclustering/common.cc
matex_compute_distance_SOURCES += src/unsupervised/spectralclustering/common.h
matex_compute_distance_SOURCES += src/unsupervised/spectralclustering/compute_distance.cc
matex_compute_distance_SOURCES += src/unsupervised/spectralclustering/compute_distance.h

bin_PROGRAMS += matex_distance_to_similarity
matex_distance_to_similarity_SOURCES =
matex_distance_to_similarity_SOURCES += src/unsupervised/spectralclustering/common.cc
matex_distance_to_similarity_SOURCES += src/unsupervised/spectralclustering/common.h
matex_distance_to_similarity_SOURCES += src/unsupervised/spectralclustering/distance_to_similarity.cc
matex_distance_to_similarity_SOURCES += src/unsupervised/spectralclustering/distance_to_similarity.h

bin_PROGRAMS += matex_evd
matex_evd_CPPFLAGS =
matex_evd_CPPFLAGS += -I$(top_srcdir)/contrib
matex_evd_CPPFLAGS += -I$(top_build_prefix)contrib
matex_evd_LDFLAGS =
matex_evd_LDFLAGS += $(AM_LDFLAGS)
matex_evd_LDFLAGS += -L$(top_build_prefix)contrib/ARPACK
matex_evd_LDADD =
matex_evd_LDADD += contrib/ARPACK/libparpack.a
matex_evd_LDADD += contrib/ARPACK/libarpack.a
if HAVE_LAPACK
matex_evd_LDADD += $(LAPACK_LIBS)
matex_evd_LDFLAGS += $(LAPACK_LDFLAGS)
else
matex_evd_LDADD += contrib/ARPACK/libarpack_lapack.a
endif
if HAVE_BLAS
matex_evd_LDADD += $(BLAS_LIBS)
matex_evd_LDFLAGS += $(BLAS_LDFLAGS)
else
matex_evd_LDADD += contrib/ARPACK/libarpack_blas.a
endif
matex_evd_LDADD += $(LIBS)
matex_evd_LDADD += $(FLIBS)
matex_evd_SOURCES =
matex_evd_SOURCES += src/unsupervised/spectralclustering/common.cc
matex_evd_SOURCES += src/unsupervised/spectralclustering/common.h
matex_evd_SOURCES += src/unsupervised/spectralclustering/evd.cc
matex_evd_SOURCES += src/unsupervised/spectralclustering/evd.h

bin_PROGRAMS += matex_spectral_kmeans
matex_spectral_kmeans_SOURCES =
matex_spectral_kmeans_LDFLAGS = $(AM_LDFLAGS)
matex_spectral_kmeans_SOURCES += src/unsupervised/spectralclustering/common.cc
matex_spectral_kmeans_SOURCES += src/unsupervised/spectralclustering/common.h
matex_spectral_kmeans_SOURCES += src/unsupervised/spectralclustering/kmeans.cc
matex_spectral_kmeans_SOURCES += src/unsupervised/spectralclustering/kmeans.h

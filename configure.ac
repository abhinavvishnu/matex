# Process this file with autoconf to produce a configure script.

###############################################################################
# Init autoconf
###############################################################################
AC_PREREQ([2.69])
AC_INIT([Machine Learning Tools for Exascale (MaTEx)], [0.2],
        [Abhinav.Vishnu@pnnl.gov], [matex],
        [http://hpc.pnl.gov/matex])
AC_CONFIG_AUX_DIR([build-aux])
AC_CONFIG_HEADER([config.h])
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_SRCDIR([src/unsupervised/spectralclustering/kmeans.cc])

###############################################################################
# Init automake
###############################################################################
AM_INIT_AUTOMAKE([1.11 foreign silent-rules subdir-objects tar-ustar])

###############################################################################
# Misc. information and package setup.
###############################################################################
MATEX_TOP_BUILDDIR="`pwd`"
cd "$srcdir"
MATEX_TOP_SRCDIR="`pwd`"
cd "$MATEX_TOP_BUILDDIR"
AS_IF([test "$MATEX_TOP_BUILDDIR" != "$MATEX_TOP_SRCDIR"],
      [AC_MSG_NOTICE([Detected VPATH build])])

MATEX_WITH_MPI

###############################################################################
# C++ compiler
###############################################################################
AC_MSG_NOTICE
AC_MSG_NOTICE([C++ compiler])
AC_MSG_NOTICE
AC_LANG_PUSH([C++])

MATEX_PROG_MPICXX
MATEX_MPI_UNWRAP

# Checks for header files.
AC_CHECK_HEADERS([fenv.h])
AC_CHECK_HEADERS([stdlib.h])
AC_CHECK_HEADERS([string.h])
AC_CHECK_HEADERS([sys/time.h])
AC_CHECK_HEADERS([unistd.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_C_CONST
AC_C_INLINE
AC_C_RESTRICT
AC_C_VOLATILE
AC_TYPE_INT64_T
AC_TYPE_SIZE_T

# Checks for library functions.
AC_CHECK_FUNCS([atexit])
AC_CHECK_FUNCS([memset])
AC_CHECK_FUNCS([strtol])

AC_SEARCH_LIBS([floor], [m])
AC_SEARCH_LIBS([pow], [m])
AC_SEARCH_LIBS([sqrt], [m])

#GA_REQUIRE

AC_LANG_POP([C++])

###############################################################################
# C compiler
###############################################################################
AC_MSG_NOTICE
AC_MSG_NOTICE([C compiler])
AC_MSG_NOTICE
AC_LANG_PUSH([C])

MATEX_PROG_MPICC
MATEX_MPI_UNWRAP

AC_LANG_POP([C])

###############################################################################
# F77 compiler
###############################################################################
AC_MSG_NOTICE
AC_MSG_NOTICE([Fortran 77 compiler])
AC_MSG_NOTICE
AC_LANG_PUSH([Fortran 77])

MATEX_PROG_MPIF77
MATEX_MPI_UNWRAP

AC_F77_LIBRARY_LDFLAGS
AC_F77_WRAPPERS

AC_LANG_POP([Fortran 77])

###############################################################################
# Checks for BLAS/LAPACK require both C and Fortran features
# to
# have been detected. Further, they assume the current AC_LANG is C.
###############################################################################
AC_MSG_NOTICE
AC_MSG_NOTICE([Checks for BLAS,LAPACK])
AC_MSG_NOTICE

MATEX_BLAS
MATEX_LAPACK

###############################################################################
# Libtool
###############################################################################
AC_PROG_RANLIB

###############################################################################
# The End
###############################################################################
AC_CONFIG_FILES([Makefile])

AC_OUTPUT

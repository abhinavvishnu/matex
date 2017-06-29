!
!  Copyright (C) 2013, Northwestern University and Argonne National Laboratory
!  See COPYRIGHT notice in top-level directory.
!
!  $Id: tests.inc 2137 2015-10-07 23:53:04Z wkliao $ 
!

      integer, parameter :: INT1_KIND = selected_int_kind(2)
      integer, parameter :: INT2_KIND = selected_int_kind(4)
      integer, parameter :: INT8_KIND = selected_int_kind(18)

!!!!
! Do not tabify this unless you like hitting the 72 char limit !!!
!!!
#ifndef UD_TESTS_INC
#define UD_TESTS_INC


!/* The following prevents non-FORTRAN code from appearing in the output. */
#if defined(__osf__)
#   undef _POSIX_SOURCE
#   define _POSIX_SOURCE
#endif

#define NO_NETCDF_2 1

#include "../libf/nfconfig.inc"


!/* Parameters of test data */

#ifdef  NF_INT1_T
#   define NF_B 1
#else
#   define NF_B 0
#endif
#ifdef  NF_INT2_T
#   define NF_S 1
#else
#   define NF_S 0
#endif
! Total number of FORTRAN types:
#define NUM_FORTRAN_TYPES (3 + NF_S + NF_B)
#undef NF_B
#undef NF_S

#define NTYPES 11
#define NDIMS 5
#define NRECS 2
#define NGATTS NTYPES
#define RECDIM 1
#define MAX_RANK 3
#define MAX_NELS 64
#define MAX_DIM_LEN 4
#define MAX_NATTS 3

#define NVARS 166
!
!  #define NVARS 136   when NTYPES==6
!  #define NVARS 142   when NTYPES==7
!  #define NVARS 148   when NTYPES==8
!  #define NVARS 154   when NTYPES==9
!  #define NVARS 160   when NTYPES==10
!  #define NVARS 166   when NTYPES==11
!  c:char, b:byte, s:short, i:int, f:float, d:double, y:ubyte, t:ushort,
!  u:uint, x:int64, z:uint64
!


!/*
! * Limits of external types (based on those in ncx.h):
! */
#define X_CHAR_MIN 0
#define X_CHAR_MAX 127
#define X_INT1_MIN (-128)
#define X_INT1_MAX 127
#define X_INT2_MIN (-32768)
#define X_INT2_MAX 32767
#define X_INT_MIN  (-2147483647-1)
#define X_INT_MAX  2147483647
#if 0
#define X_REAL_MAX 3.4028234663852886e+38
#else
#define X_REAL_MAX 3.4028234663852886e+37
#endif
#define X_REAL_MIN (-X_FLOAT_MAX)
#if 0
#define X_DOUBLE_MAX 1.7976931348623157E+308
#else
#define X_DOUBLE_MAX 1.7976931348623157D+200
#endif
#define X_DOUBLE_MIN (-X_DOUBLE_MAX)
#define X_INT8_MIN (-9223372036854775807_INT8_KIND)
#define X_INT8_MAX 9223372036854775807_INT8_KIND
#define X_UINT8_MIN (-18446744073709551615.0)
#define X_UINT8_MAX 18446744073709551615.0

#define X_BYTE_MIN X_INT1_MIN
#define X_BYTE_MAX X_INT1_MAX
#define X_SHORT_MIN X_INT2_MIN
#define X_SHORT_MAX X_INT2_MAX
#define X_FLOAT_MIN X_REAL_MIN
#define X_FLOAT_MAX X_REAL_MAX

#define X_UCHAR_MAX     255
#define X_UCHAR_MIN     0
#define X_UBYTE_MAX     X_UCHAR_MAX
#define X_UBYTE_MIN     X_UCHAR_MIN
#define X_USHORT_MAX    65535
#define X_USHORT_MIN    0
#define X_UINT_MAX      4294967295_INT8_KIND
#define X_UINT_MIN      0


!/*
! * Examples of invalid argument values:
! */
#define BAD_ID -1
#define BAD_DIMID -1
#define BAD_VARID -2
#define BAD_ATTNUM -1
#define BAD_TYPE 0
#define BAD_FILLMODE -1
#define BAD_NAME 'a/b'


!/*
! * Internal data types:
! */
#define NFT_UNSPECIFIED 0
#define NFT_TEXT 16
#define NFT_CHAR NFT_TEXT
#define NFT_INT1 17
#define NFT_INT2 18
#define NFT_INT 20
#define NFT_REAL 36
#define NFT_DOUBLE 40
#define NFT_INT8 43


!/*
! * Define a macro for trimming trailing blanks from character variables.
! */


!
! FORTRAN GETARG() subroutine:
!
#ifdef __hpux
#   define      getarg  getarg_
#endif


#endif /* UD_TESTS_INC */

#include "mpif.h"


!    /* Global variables - filenames */

      CHARACTER*128      testfile  !/* netCDF read-only test data */
      CHARACTER*128      scratch   !/* netCDF test file for writing */

!    /* Global variables - command-line arguments */

      LOGICAL   CREATE_FILE
      LOGICAL   READONLY
      LOGICAL   VERBOSE
      INTEGER   NFAILS
      INTEGER   MAX_NMPT        !/* max num messages per test */

!    /* Global variables - test data */

      CHARACTER*2               DIM_NAME(NDIMS)
      integer(kind=MPI_OFFSET_KIND) DIM_LEN(NDIMS)
      CHARACTER*(2+MAX_RANK)    VAR_NAME(NVARS)
      INTEGER                   VAR_TYPE(NVARS)
      INTEGER                   VAR_RANK(NVARS)
      INTEGER                   VAR_DIMID(MAX_RANK,NVARS)
      integer(kind=MPI_OFFSET_KIND) VAR_SHAPE(MAX_RANK,NVARS)
      INTEGER                   VAR_NELS(NVARS)
      INTEGER                   VAR_NATTS(NVARS)
      CHARACTER*2               ATTNAME(MAX_NATTS,NVARS)
      CHARACTER*2               GATT_NAME(NGATTS)
      INTEGER                   ATTTYPE(NGATTS,NVARS)
      INTEGER                   GATT_TYPE(NGATTS)
      INTEGER                   ATTLEN(MAX_NATTS,NVARS)
      integer(kind=MPI_OFFSET_KIND) GATT_LEN(NGATTS)

!    /* Miscellaneous global variables: */
      CHARACTER*128              PROGNAME   !/* name of the program */
      INTEGER                   COMM       !/* MPI communicator */

      INTEGER                   NFAILSTOTAL

!    /* Common blocks for global variables: */

!/* MPI_OFFSET */
      COMMON    /OFFSETCOM/     DIM_LEN, VAR_SHAPE, GATT_LEN 

      COMMON    /MPICOM/        COMM
      COMMON    /LOGCOM/        CREATE_FILE, &
                                READONLY, & !/* don't change files */
                                VERBOSE     !/* print details of tests */

      COMMON    /TXTCOM/        TESTFILE, &
                                SCRATCH, &
                                DIM_NAME, &
                                VAR_NAME, &
                                ATTNAME, &
                                GATT_NAME, &
                                PROGNAME

      COMMON    /INTCOM/        NFAILS, &  ! number of failures in specific test
                                VAR_TYPE, &
                                VAR_RANK, &
                                VAR_DIMID, &
                                VAR_NELS, &
                                VAR_NATTS, &
                                ATTTYPE, &
                                GATT_TYPE, &
                                ATTLEN, &
                                MAX_NMPT, &
                                NFAILSTOTAL


      integer numTypes, numVars, numGatts, cdf_format, extra_flags, info
      common / PROBLEM_SIZE / numTypes, numVars, numGatts, cdf_format, &
                              extra_flags, info


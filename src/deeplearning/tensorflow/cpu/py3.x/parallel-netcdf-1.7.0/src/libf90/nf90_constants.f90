!
!  Copyright (C) 2013, Northwestern University and Argonne National Laboratory
!  See COPYRIGHT notice in top-level directory.
!
! $Id: nf90_constants.f90 2319 2016-02-04 08:04:01Z wkliao $
!
! This file is taken from netcdf_constants.f90 with changes for PnetCDF use
!
!
  !
  ! external netcdf data types:
  !
  integer, parameter, public :: &
    nf90_byte   = 1,            &
    nf90_int1   = nf90_byte,    &
    nf90_char   = 2,            &
    nf90_short  = 3,            &
    nf90_int2   = nf90_short,   &
    nf90_int    = 4,            &
    nf90_int4   = nf90_int,     &
    nf90_float  = 5,            &
    nf90_real   = nf90_float,   &
    nf90_real4  = nf90_float,   &
    nf90_double = 6,            &
    nf90_real8  = nf90_double,  &
    nf90_ubyte  = 7,            &
    nf90_ushort = 8,            &
    nf90_uint   = 9,            &
    nf90_int64  = 10,           &
    nf90_uint64 = 11

  !
  ! default fill values:
  !
  ! character (len = 1),           parameter, public :: &
  !   nf90_fill_char  = achar(0)
  integer (kind =  OneByteInt),  parameter, public :: &
    nf90_fill_char  = 0,                              &
    nf90_fill_byte  = -127,                           &
    nf90_fill_int1  = nf90_fill_byte
  integer (kind =  TwoByteInt),  parameter, public :: &
    nf90_fill_short = -32767,                         &
    nf90_fill_int2  = nf90_fill_short,                &
    nf90_fill_ubyte  = 255
  integer (kind = FourByteInt),  parameter, public :: &
    nf90_fill_int    = -2147483647,                   &
    nf90_fill_ushort = 65535
  real   (kind =  FourByteReal), parameter, public :: &
    nf90_fill_float = 9.9692099683868690e+36,         &
    nf90_fill_real  = nf90_fill_float,                &
    nf90_fill_real4 = nf90_fill_float
  real   (kind = EightByteReal), parameter, public :: &
    nf90_fill_double = 9.9692099683868690e+36,        &
    nf90_fill_real8  = nf90_fill_double,              &
    nf90_fill_uint64 = 1.8446744073709551614e+19
  integer (kind = EightByteInt), parameter, public :: &
    nf90_fill_uint   = 4294967295_EightByteInt,       &
    nf90_fill_int64  = -9223372036854775806_EightByteInt

  !
  ! mode flags for opening and creating a netcdf dataset:
  !
  integer, parameter, public :: &
    nf90_nowrite      = 0,      &
    nf90_write        = 1,      &
    nf90_clobber      = 0,      &
    nf90_noclobber    = 4,      &
    nf90_fill         = 0,      &
    nf90_nofill       = 256,    &
    nf90_64bit_offset = 512,    &
    nf90_64bit_data   = 32,     &
    nf90_lock         = 1024,   &
    nf90_share        = 2048

  integer, parameter, public ::  &
    nf90_sizehint_default = 0,   &
    nf90_align_chunk      = -1

  !
  ! size argument for defining an unlimited dimension:
  !
  integer, parameter, public :: &
    nf90_unlimited = 0

  integer(KIND=MPI_OFFSET_KIND), parameter, public :: &
    nf90mpi_unlimited = 0

  ! NULL request for non-blocking I/O APIs
  integer, parameter, public :: nf90_req_null = -1

  ! indicate to flush all pending non-blocking requests
  integer, parameter, public :: nf90_req_all = -1
  integer, parameter, public :: nf90_get_req_all = -2
  integer, parameter, public :: nf90_put_req_all = -3

  !
  ! global attribute id:
  !
  integer, parameter, public :: nf90_global = 0

  !
  ! implementation limits:
  !
  integer, parameter, public :: &
    nf90_max_dims     = 1024,    &
    nf90_max_attrs    = 8192,   &
    nf90_max_vars     = 8192,   &
    nf90_max_name     = 256,    &
    nf90_max_var_dims = 1024

  !
  ! error handling modes:
  !
  integer, parameter, public :: &
    nf90_fatal   = 1,           &
    nf90_verbose = 2

  !
  ! format version numbers:
  !
  integer, parameter, public :: &
    nf90_format_classic = 1,    &
    nf90_format_cdf2 = 2,      &
    nf90_format_netcdf4 = 3,    &
    nf90_format_netcdf4_classic = 4, &
    nf90_format_cdf5 = 5, &
    nf90_format_64bit = nf90_format_cdf2, &
    nf90_format_64bit_offset = nf90_format_cdf2, &
    nf90_format_64bit_data = nf90_format_cdf5

  !
  ! error codes:
  !
  integer, parameter, public :: &
    NF90_NOERR          = NF_NOERR         , & ! No Error
    NF90_EBADID         = NF_EBADID        , & ! Not a netcdf id
    NF90_ENFILE         = NF_ENFILE        , & ! Too many netcdfs open
    NF90_EEXIST         = NF_EEXIST        , & ! netcdf file exists and NF_NOCLOBBER
    NF90_EINVAL         = NF_EINVAL        , & ! Invalid Argument
    NF90_EPERM          = NF_EPERM         , & ! Write to read only
    NF90_ENOTINDEFINE   = NF_ENOTINDEFINE  , & ! Operation not allowed in data mode
    NF90_EINDEFINE      = NF_EINDEFINE     , & ! Operation not allowed in define mode
    NF90_EINVALCOORDS   = NF_EINVALCOORDS  , & ! Index exceeds dimension bound
    NF90_EMAXDIMS       = NF_EMAXDIMS      , & ! NF_MAX_DIMS exceeded
    NF90_ENAMEINUSE     = NF_ENAMEINUSE    , & ! String match to name in use
    NF90_ENOTATT        = NF_ENOTATT       , & ! Attribute not found
    NF90_EMAXATTS       = NF_EMAXATTS      , & ! NF_MAX_ATTRS exceeded
    NF90_EBADTYPE       = NF_EBADTYPE      , & ! Not a netcdf data type
    NF90_EBADDIM        = NF_EBADDIM       , & ! Invalid dimension id or name
    NF90_EUNLIMPOS      = NF_EUNLIMPOS     , & ! NFMPI_UNLIMITED in the wrong index
    NF90_EMAXVARS       = NF_EMAXVARS      , & ! NF_MAX_VARS exceeded
    NF90_ENOTVAR        = NF_ENOTVAR       , & ! Variable not found
    NF90_EGLOBAL        = NF_EGLOBAL       , & ! Action prohibited on NF_GLOBAL varid
    NF90_ENOTNC         = NF_ENOTNC        , & ! Not a netcdf file
    NF90_ESTS           = NF_ESTS          , & ! In Fortran, string too short
    NF90_EMAXNAME       = NF_EMAXNAME      , & ! NF_MAX_NAME exceeded
    NF90_EUNLIMIT       = NF_EUNLIMIT      , & ! NFMPI_UNLIMITED size already in use
    NF90_ENORECVARS     = NF_ENORECVARS    , & ! nc_rec op when there are no record vars
    NF90_ECHAR          = NF_ECHAR         , & ! Attempt to convert between text & numbers
    NF90_EEDGE          = NF_EEDGE         , & ! Edge+start exceeds dimension bound
    NF90_ESTRIDE        = NF_ESTRIDE       , & ! Illegal stride
    NF90_EBADNAME       = NF_EBADNAME      , & ! Attribute or variable name contains illegal characters
    NF90_ERANGE         = NF_ERANGE        , & ! Math result not representable
    NF90_ENOMEM         = NF_ENOMEM        , & ! Memory allocation (malloc) failure
    NF90_EVARSIZE       = NF_EVARSIZE      , & ! One or more variable sizes violate format constraints
    NF90_EDIMSIZE       = NF_EDIMSIZE      , & ! Invalid dimension size
    NF90_ETRUNC         = NF_ETRUNC        , & ! File likely truncated or possibly corrupted
    NF90_EAXISTYPE      = NF_EAXISTYPE         ! Unknown axis type

  ! Following errors are added for DAP
  integer, parameter, public :: &
    NF90_EDAP           = NF_EDAP          , & ! Generic DAP error
    NF90_ECURL          = NF_ECURL         , & ! Generic libcurl error
    NF90_EIO            = NF_EIO           , & ! Generic IO error
    NF90_ENODATA        = NF_ENODATA       , & ! Attempt to access variable with no data
    NF90_EDAPSVC        = NF_EDAPSVC       , & ! DAP server error
    NF90_EDAS           = NF_EDAS          , & ! Malformed or inaccessible DAS
    NF90_EDDS           = NF_EDDS          , & ! Malformed or inaccessible DDS
    NF90_EDATADDS       = NF_EDATADDS      , & ! Malformed or inaccessible DATADDS
    NF90_EDAPURL        = NF_EDAPURL       , & ! Malformed DAP URL
    NF90_EDAPCONSTRAINT = NF_EDAPCONSTRAINT, & ! Malformed DAP Constraint
    NF90_ETRANSLATION   = NF_ETRANSLATION  , & ! Untranslatable construct
    NF90_EACCESS        = NF_EACCESS       , & ! Access Failure
    NF90_EAUTH          = NF_EAUTH             ! Authorization Failure

  ! Misc. additional errors
  integer, parameter, public :: &
    NF90_ENOTFOUND      = NF_ENOTFOUND     , & ! No such file
    NF90_ECANTREMOVE    = NF_ECANTREMOVE       ! Can't remove file

  ! netCDF-4 error codes (copied from netCDF release)
  integer, parameter, public :: &
    NF90_EHDFERR        = NF_EHDFERR       , & ! Error at HDF5 layer.
    NF90_ECANTREAD      = NF_ECANTREAD     , & ! Can't read.
    NF90_ECANTWRITE     = NF_ECANTWRITE    , & ! Can't write.
    NF90_ECANTCREATE    = NF_ECANTCREATE   , & ! Can't create.
    NF90_EFILEMETA      = NF_EFILEMETA     , & ! Problem with file metadata.
    NF90_EDIMMETA       = NF_EDIMMETA      , & ! Problem with dimension metadata.
    NF90_EATTMETA       = NF_EATTMETA      , & ! Problem with attribute metadata.
    NF90_EVARMETA       = NF_EVARMETA      , & ! Problem with variable metadata.
    NF90_ENOCOMPOUND    = NF_ENOCOMPOUND   , & ! Not a compound type.
    NF90_EATTEXISTS     = NF_EATTEXISTS    , & ! Attribute already exists.
    NF90_ENOTNC4        = NF_ENOTNC4       , & ! Attempting netcdf-4 operation on netcdf-3 file.
    NF90_ESTRICTNC3     = NF_ESTRICTNC3    , & ! Attempting netcdf-4 operation on strict nc3 netcdf-4 file.
    NF90_ENOTNC3        = NF_ENOTNC3       , & ! Attempting netcdf-3 operation on netcdf-4 file.
    NF90_ENOPAR         = NF_ENOPAR        , & ! Parallel operation on file opened for non-parallel access.
    NF90_EPARINIT       = NF_EPARINIT      , & ! Error initializing for parallel access.
    NF90_EBADGRPID      = NF_EBADGRPID     , & ! Bad group ID.
    NF90_EBADTYPID      = NF_EBADTYPID     , & ! Bad type ID.
    NF90_ETYPDEFINED    = NF_ETYPDEFINED   , & ! Type has already been defined and may not be edited.
    NF90_EBADFIELD      = NF_EBADFIELD     , & ! Bad field ID.
    NF90_EBADCLASS      = NF_EBADCLASS     , & ! Bad class.
    NF90_EMAPTYPE       = NF_EMAPTYPE      , & ! Mapped access for atomic types only.
    NF90_ELATEFILL      = NF_ELATEFILL     , & ! Attempt to define fill value when data already exists.
    NF90_ELATEDEF       = NF_ELATEDEF      , & ! Attempt to define var properties, like deflate, after enddef.
    NF90_EDIMSCALE      = NF_EDIMSCALE     , & ! Probem with HDF5 dimscales.
    NF90_ENOGRP         = NF_ENOGRP        , & ! No group found.
    NF90_ESTORAGE       = NF_ESTORAGE      , & ! Can't specify both contiguous and chunking.
    NF90_EBADCHUNK      = NF_EBADCHUNK     , & ! Bad chunksize.
    NF90_ENOTBUILT      = NF_ENOTBUILT     , & ! Attempt to use feature that was not turned on when netCDF was built
    NF90_EDISKLESS      = NF_EDISKLESS     , & ! Error in using diskless  access.
    NF90_ECANTEXTEND    = NF_ECANTEXTEND   , & ! Attempt to extend dataset during ind. I/O operation.
    NF90_EMPI           = NF_EMPI              ! MPI operation failed.

  ! This is the position of NC_NETCDF4 in cmode, counting from the
  ! right, starting (uncharacteristically for fortran) at 0. It's needed
  ! for the BTEST function calls.
  integer, parameter, private :: NETCDF4_BIT = 12


  ! PnetCDF error codes start here
  integer, parameter, public :: &
      NF90_ESMALL                   = NF_ESMALL                   , & ! size of off_t too small for format
      NF90_ENOTINDEP                = NF_ENOTINDEP                , & ! Operation not allowed in collective data mode
      NF90_EINDEP                   = NF_EINDEP                   , & ! Operation not allowed in independent data mode
      NF90_EFILE                    = NF_EFILE                    , & ! Unknown error in file operation
      NF90_EREAD                    = NF_EREAD                    , & ! Unknown error in reading file
      NF90_EWRITE                   = NF_EWRITE                   , & ! Unknown error in writting to file
      NF90_EOFILE                   = NF_EOFILE                   , & ! file open/creation failed
      NF90_EMULTITYPES              = NF_EMULTITYPES              , & ! Multiple types used in memory data
      NF90_EIOMISMATCH              = NF_EIOMISMATCH              , & ! Input/Output data amount mismatch
      NF90_ENEGATIVECNT             = NF_ENEGATIVECNT             , & ! Negative count is specified
      NF90_EUNSPTETYPE              = NF_EUNSPTETYPE              , & ! Unsupported etype in memory MPI datatype
      NF90_EINVAL_REQUEST           = NF_EINVAL_REQUEST           , & ! invalid nonblocking request ID
      NF90_EAINT_TOO_SMALL          = NF_EAINT_TOO_SMALL          , & ! MPI_Aint not large enough to hold requested value
      NF90_ENOTSUPPORT              = NF_ENOTSUPPORT              , & ! feature is not yet supported
      NF90_ENULLBUF                 = NF_ENULLBUF                 , & ! trying to attach a NULL buffer
      NF90_EPREVATTACHBUF           = NF_EPREVATTACHBUF           , & ! previous attached buffer is found
      NF90_ENULLABUF                = NF_ENULLABUF                , & ! no attached buffer is found
      NF90_EPENDINGBPUT             = NF_EPENDINGBPUT             , & ! pending bput is found, cannot detach buffer
      NF90_EINSUFFBUF               = NF_EINSUFFBUF               , & ! attached buffer is too small
      NF90_ENOENT                   = NF_ENOENT                   , & ! File does not exist when calling nfmpi_open()
      NF90_EINTOVERFLOW             = NF_EINTOVERFLOW             , & ! Overflow when type cast to 4-byte integer
      NF90_ENOTENABLED              = NF_ENOTENABLED              , & ! Overflow when type cast to 4-byte integer
      NF90_EBAD_FILE                = NF_EBAD_FILE                , & ! Invalid file name (e.g., path name too long)
      NF90_ENO_SPACE                = NF_ENO_SPACE                , & ! Not enough space
      NF90_EQUOTA                   = NF_EQUOTA                   , & ! Quota exceeded
      NF90_ENULLSTART               = NF_ENULLSTART               , & ! argument start is a NULL pointer
      NF90_ENULLCOUNT               = NF_ENULLCOUNT               , & ! argument count is a NULL pointer
      NF90_EINVAL_CMODE             = NF_EINVAL_CMODE             , & ! Invalid file create mode, cannot have both
                                                                      ! NC_64BIT_OFFSET & NC_64BIT_DATA
      NF90_ETYPESIZE                = NF_ETYPESIZE                , & ! MPI derived data type size error (bigger than the
                                                                      ! variable size)
      NF90_ETYPE_MISMATCH           = NF_ETYPE_MISMATCH           , & ! element type of the MPI derived data type mismatches
                                                                      ! the variable type
      NF90_ETYPESIZE_MISMATCH       = NF_ETYPESIZE_MISMATCH       , & ! file type size mismatches buffer type size
      NF90_ESTRICTCDF2              = NF_ESTRICTCDF2              , & ! Attempting CDF-5 operation on CDF-2 file
      NF90_ENOTRECVAR               = NF_ENOTRECVAR               , & ! Attempting operation only for record variables
      NF90_ENOTFILL                 = NF_ENOTFILL                 , & ! Attempting to fill a variable when its fill mode is off
      NF90_EMULTIDEFINE             = NF_EMULTIDEFINE             , & ! NC definitions on multiprocesses conflict
      NF90_EMULTIDEFINE_OMODE       = NF_EMULTIDEFINE_OMODE       , & ! file create/open modes are inconsistent among processes
      NF90_EMULTIDEFINE_DIM_NUM     = NF_EMULTIDEFINE_DIM_NUM     , & ! inconsistent number of dimensions
      NF90_EMULTIDEFINE_DIM_SIZE    = NF_EMULTIDEFINE_DIM_SIZE    , & ! inconsistent size of dimension
      NF90_EMULTIDEFINE_DIM_NAME    = NF_EMULTIDEFINE_DIM_NAME    , & ! inconsistent dimension names
      NF90_EMULTIDEFINE_VAR_NUM     = NF_EMULTIDEFINE_VAR_NUM     , & ! inconsistent number of variables
      NF90_EMULTIDEFINE_VAR_NAME    = NF_EMULTIDEFINE_VAR_NAME    , & ! inconsistent variable name
      NF90_EMULTIDEFINE_VAR_NDIMS   = NF_EMULTIDEFINE_VAR_NDIMS   , & ! inconsistent variable's number of dimensions
      NF90_EMULTIDEFINE_VAR_DIMIDS  = NF_EMULTIDEFINE_VAR_DIMIDS  , & ! inconsistent variable's dimid
      NF90_EMULTIDEFINE_VAR_TYPE    = NF_EMULTIDEFINE_VAR_TYPE    , & ! inconsistent variable's data type
      NF90_EMULTIDEFINE_VAR_LEN     = NF_EMULTIDEFINE_VAR_LEN     , & ! inconsistent variable's size
      NF90_EMULTIDEFINE_NUMRECS     = NF_EMULTIDEFINE_NUMRECS     , & ! inconsistent number of records
      NF90_EMULTIDEFINE_VAR_BEGIN   = NF_EMULTIDEFINE_VAR_BEGIN   , & ! inconsistent variable file begin offset (internal use)
      NF90_EMULTIDEFINE_ATTR_NUM    = NF_EMULTIDEFINE_ATTR_NUM    , & ! inconsistent number of attributes
      NF90_EMULTIDEFINE_ATTR_SIZE   = NF_EMULTIDEFINE_ATTR_SIZE   , & ! inconsistent memory space used by attribute (internal use)
      NF90_EMULTIDEFINE_ATTR_NAME   = NF_EMULTIDEFINE_ATTR_NAME   , & ! inconsistent attribute name
      NF90_EMULTIDEFINE_ATTR_TYPE   = NF_EMULTIDEFINE_ATTR_TYPE   , & ! inconsistent attribute type
      NF90_EMULTIDEFINE_ATTR_LEN    = NF_EMULTIDEFINE_ATTR_LEN    , & ! inconsistent attribute length
      NF90_EMULTIDEFINE_ATTR_VAL    = NF_EMULTIDEFINE_ATTR_VAL    , & ! inconsistent attribute value
      NF90_EMULTIDEFINE_FNC_ARGS    = NF_EMULTIDEFINE_FNC_ARGS    , & ! inconsistent function arguments used in collective API
      NF90_EMULTIDEFINE_FILL_MODE   = NF_EMULTIDEFINE_FILL_MODE   , & !  inconsistent dataset fill mode
      NF90_EMULTIDEFINE_VAR_FILL_MODE = NF_EMULTIDEFINE_VAR_FILL_MODE, & ! inconsistent variable fill mode
      NF90_EMULTIDEFINE_VAR_FILL_VALUE = NF_EMULTIDEFINE_VAR_FILL_VALUE, & ! inconsistent variable fill value
      NF90_ECMODE                   = NF_EMULTIDEFINE_OMODE       , &
      NF90_EDIMS_NELEMS_MULTIDEFINE = NF_EMULTIDEFINE_DIM_NUM     , &
      NF90_EDIMS_SIZE_MULTIDEFINE   = NF_EMULTIDEFINE_DIM_SIZE    , &
      NF90_EDIMS_NAME_MULTIDEFINE   = NF_EMULTIDEFINE_DIM_NAME    , &
      NF90_EVARS_NELEMS_MULTIDEFINE = NF_EMULTIDEFINE_VAR_NUM     , &
      NF90_EVARS_NAME_MULTIDEFINE   = NF_EMULTIDEFINE_VAR_NAME    , &
      NF90_EVARS_NDIMS_MULTIDEFINE  = NF_EMULTIDEFINE_VAR_NDIMS   , &
      NF90_EVARS_DIMIDS_MULTIDEFINE = NF_EMULTIDEFINE_VAR_DIMIDS  , &
      NF90_EVARS_TYPE_MULTIDEFINE   = NF_EMULTIDEFINE_VAR_TYPE    , &
      NF90_EVARS_LEN_MULTIDEFINE    = NF_EMULTIDEFINE_VAR_LEN     , &
      NF90_ENUMRECS_MULTIDEFINE     = NF_EMULTIDEFINE_NUMRECS     , &
      NF90_EVARS_BEGIN_MULTIDEFINE  = NF_EMULTIDEFINE_VAR_BEGIN

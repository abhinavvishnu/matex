BEGIN {
D["PACKAGE_NAME"]=" \"parallel-netcdf\""
D["PACKAGE_TARNAME"]=" \"parallel-netcdf\""
D["PACKAGE_VERSION"]=" \"1.7.0\""
D["PACKAGE_STRING"]=" \"parallel-netcdf 1.7.0\""
D["PACKAGE_BUGREPORT"]=" \"parallel-netcdf@mcs.anl.gov\""
D["PACKAGE_URL"]=" \"\""
D["PNETCDF_VERSION_MAJOR"]=" 1"
D["PNETCDF_VERSION_MINOR"]=" 7"
D["PNETCDF_VERSION_SUB"]=" 0"
D["PNETCDF_VERSION_PRE"]=" "
D["PNETCDF_VERSION"]=" \"1.7.0\""
D["PNETCDF_RELEASE_DATE"]=" \"03 Mar 2016\""
D["CONFIGURE_ARGS_CLEAN"]=" \"--prefix=/home/manz551/opt CFLAGS=-g -O2 -fPIC CPPFLAGS=-g -O2 -fPIC CXXFLAGS=-g -O2 -fPIC FFLAGS=-O2 -fPIC FCFLAGS=-O"\
"2 -fPIC --disable-cxx --disable-fortran\""
D["STDC_HEADERS"]=" 1"
D["HAVE_SYS_TYPES_H"]=" 1"
D["HAVE_SYS_STAT_H"]=" 1"
D["HAVE_STDLIB_H"]=" 1"
D["HAVE_STRING_H"]=" 1"
D["HAVE_MEMORY_H"]=" 1"
D["HAVE_STRINGS_H"]=" 1"
D["HAVE_INTTYPES_H"]=" 1"
D["HAVE_STDINT_H"]=" 1"
D["HAVE_UNISTD_H"]=" 1"
D["HAVE__BOOL"]=" 1"
D["HAVE_STDBOOL_H"]=" 1"
D["HAVE_STRERROR"]=" 1"
D["HAVE_ACCESS"]=" 1"
D["HAVE_UNLINK"]=" 1"
D["SIZEOF_MPI_OFFSET"]=" 8"
D["SIZEOF_MPI_AINT"]=" 8"
D["ENABLE_CDF5"]=" /**/"
D["ENABLE_REQ_AGGREGATION"]=" /**/"
D["HAVE_MPI_INFO_DUP"]=" 1"
D["HAVE_MPI_INFO_FREE"]=" 1"
D["HAVE_MPI_GET_ADDRESS"]=" 1"
D["HAVE_MPI_TYPE_CREATE_SUBARRAY"]=" 1"
D["HAVE_MPI_TYPE_CREATE_HVECTOR"]=" 1"
D["HAVE_MPI_TYPE_CREATE_HINDEXED"]=" 1"
D["HAVE_MPI_TYPE_CREATE_STRUCT"]=" 1"
D["HAVE_MPI_TYPE_CREATE_RESIZED"]=" 1"
D["HAVE_MPI_TYPE_GET_EXTENT"]=" 1"
D["HAVE_MPI_COMBINER_DUP"]=" 1"
D["HAVE_MPI_COMBINER_HVECTOR_INTEGER"]=" 1"
D["HAVE_MPI_COMBINER_HINDEXED_INTEGER"]=" 1"
D["HAVE_MPI_COMBINER_SUBARRAY"]=" 1"
D["HAVE_MPI_COMBINER_DARRAY"]=" 1"
D["HAVE_MPI_COMBINER_RESIZED"]=" 1"
D["HAVE_MPI_COMBINER_STRUCT_INTEGER"]=" 1"
D["HAVE_MPI_COMBINER_INDEXED_BLOCK"]=" 1"
D["HAVE_MPI_COMBINER_F90_REAL"]=" 1"
D["HAVE_MPI_COMBINER_F90_INTEGER"]=" 1"
D["HAVE_MPI_COMBINER_F90_COMPLEX"]=" 1"
D["HAVE_MPI_ERR_FILE_EXISTS"]=" 1"
D["HAVE_MPI_ERR_NO_SUCH_FILE"]=" 1"
D["HAVE_MPI_ERR_AMODE"]=" 1"
D["HAVE_MPI_ERR_NOT_SAME"]=" 1"
D["HAVE_MPI_ERR_BAD_FILE"]=" 1"
D["HAVE_MPI_ERR_READ_ONLY"]=" 1"
D["HAVE_MPI_ERR_ACCESS"]=" 1"
D["HAVE_MPI_ERR_NO_SPACE"]=" 1"
D["HAVE_MPI_ERR_QUOTA"]=" 1"
D["HAVE_MPI_CHAR"]=" 1"
D["HAVE_MPI_BYTE"]=" 1"
D["HAVE_MPI_SIGNED_CHAR"]=" 1"
D["HAVE_MPI_UNSIGNED_CHAR"]=" 1"
D["HAVE_MPI_SHORT"]=" 1"
D["HAVE_MPI_UNSIGNED_SHORT"]=" 1"
D["HAVE_MPI_INT"]=" 1"
D["HAVE_MPI_UNSIGNED"]=" 1"
D["HAVE_MPI_LONG"]=" 1"
D["HAVE_MPI_FLOAT"]=" 1"
D["HAVE_MPI_DOUBLE"]=" 1"
D["HAVE_MPI_LONG_LONG_INT"]=" 1"
D["HAVE_MPI_UNSIGNED_LONG_LONG"]=" 1"
D["HAVE_MPI_UB"]=" 1"
D["HAVE_MPI_LB"]=" 1"
D["HAVE_MPI_OFFSET_DATATYPE"]=" 1"
D["HAVE_SSIZE_T"]=" 1"
D["HAVE_PTRDIFF_T"]=" 1"
D["HAVE_USHORT"]=" 1"
D["HAVE_UINT"]=" 1"
D["SIZEOF_SIZE_T"]=" 8"
D["SIZEOF_OFF_T"]=" 8"
D["SIZEOF_SIGNED_CHAR"]=" 1"
D["SIZEOF_UNSIGNED_CHAR"]=" 1"
D["SIZEOF_SHORT"]=" 2"
D["SIZEOF_UNSIGNED_SHORT"]=" 2"
D["SIZEOF_INT"]=" 4"
D["SIZEOF_UNSIGNED_INT"]=" 4"
D["SIZEOF_LONG"]=" 8"
D["SIZEOF_FLOAT"]=" 4"
D["SIZEOF_DOUBLE"]=" 8"
D["SIZEOF_LONG_LONG"]=" 8"
D["SIZEOF_UNSIGNED_LONG_LONG"]=" 8"
D["SIZEOF_USHORT"]=" 2"
D["SIZEOF_UINT"]=" 4"
  for (key in D) D_is_set[key] = 1
  FS = ""
}
/^[\t ]*#[\t ]*(define|undef)[\t ]+[_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ][_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789]*([\t (]|$)/ {
  line = $ 0
  split(line, arg, " ")
  if (arg[1] == "#") {
    defundef = arg[2]
    mac1 = arg[3]
  } else {
    defundef = substr(arg[1], 2)
    mac1 = arg[2]
  }
  split(mac1, mac2, "(") #)
  macro = mac2[1]
  prefix = substr(line, 1, index(line, defundef) - 1)
  if (D_is_set[macro]) {
    # Preserve the white space surrounding the "#".
    print prefix "define", macro P[macro] D[macro]
    next
  } else {
    # Replace #undef with comments.  This is necessary, for example,
    # in the case of _POSIX_SOURCE, which is predefined and required
    # on some systems where configure will not decide to define it.
    if (defundef == "undef") {
      print "/*", prefix defundef, macro, "*/"
      next
    }
  }
}
{ print }

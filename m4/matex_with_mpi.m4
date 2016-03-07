# MATEX_WITH_MPI
# --------------
# --with-mpi[=ARG]
# If ARG is blank, the MPI compilers will be searched for. Otherwise, ARG is
# parsed for any compiler and/or linker options.
AC_DEFUN([MATEX_WITH_MPI], [
# MPI_* vars might exist in environment, but they are really internal.
# Reset them.
MPI_LIBS=
MPI_LDFLAGS=
MPI_CPPFLAGS=
# First of all, which messaging library do we want?
AC_ARG_WITH([mpi],
    [AS_HELP_STRING([--with-mpi[[=ARG]]],
        [leave ARG blank to search for MPI compiler wrappers in PATH])],
    [],
    [with_mpi=yes])
with_mpi_wrappers=no
need_parse=no
AS_CASE([$with_mpi],
[yes], [with_mpi_wrappers=yes],
[no],  [AC_MSG_ERROR([MPI is required])],
[*],   [need_parse=yes])
AS_IF([test x$need_parse = xyes],
    [MATEX_ARG_PARSE([with_mpi], [MPI_LIBS], [MPI_LDFLAGS], [MPI_CPPFLAGS])])
AC_SUBST([MPI_LIBS])
AC_SUBST([MPI_LDFLAGS])
AC_SUBST([MPI_CPPFLAGS])
])dnl

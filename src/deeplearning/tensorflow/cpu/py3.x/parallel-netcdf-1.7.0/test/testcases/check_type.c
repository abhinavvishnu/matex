/*
 *  Copyright (C) 2015, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *  $Id$
 */

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * This program tests conflicted in-memory data types against external data
 * types and see if the correct error codes were returned.
 *
 * The compile and run commands are given below.
 *
 *    % mpicc -g -o check_type check_type.c -lpnetcdf
 *
 *    % mpiexec -l -n 1 check_type testfile.nc
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pnetcdf.h>

#include <testutils.h>

static char* err_code_name(int err);

static char* etype_name(nc_type etype) {
    switch (etype) {
        case (0):  return "NC_NAT";
        case (1):  return "NC_BYTE";
        case (2):  return "NC_CHAR";
        case (3):  return "NC_SHORT";
        case (4):  return "NC_INT";
        case (5):  return "NC_FLOAT";
        case (6):  return "NC_DOUBLE";
        case (7):  return "NC_UBYTE";
        case (8):  return "NC_USHORT";
        case (9):  return "NC_UINT";
        case (10): return "NC_INT64";
        case (11): return "NC_UINT64";
        default:
              return "Invalid nc_type";
    }
}

#define ERR0 { \
    if (err != NC_NOERR) { \
        printf("Error at line %d err=%s\n",__LINE__,err_code_name(err)); \
        nerrs++; \
    } \
}

#define ERR(expect_err,itype,etype) { \
    if (err != expect_err) { \
        printf("Error at line %3d: itype=%9s etype=%-9s err=%s\n", \
               __LINE__,itype,etype_name(etype),err_code_name(err)); \
        nerrs++; \
    }  \
} 

int main(int argc, char* argv[])
{
    char filename[256], *varname[12], buf[1024], attname[256];
    int i, err, nerrs=0, rank, ncid, dimid, varid[12];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc > 2) {
        if (!rank) printf("Usage: %s [filename]\n",argv[0]);
        MPI_Finalize();
        return 0;
    }
    strcpy(filename, "testfile.nc");
    if (argc == 2) strcpy(filename, argv[1]);
    MPI_Bcast(filename, 256, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        char cmd_str[256];
        sprintf(cmd_str, "*** TESTING C   %s for checking for type conflict ", argv[0]);
        printf("%-66s ------ ", cmd_str); fflush(stdout);
    }

    err = ncmpi_create(MPI_COMM_WORLD, filename, NC_CLOBBER|NC_64BIT_DATA, MPI_INFO_NULL, &ncid); ERR0
    err = ncmpi_def_dim(ncid, "x", 2, &dimid); ERR0

    varname[0]  = "var_nat";
    varname[1]  = "var_byte";
    varname[2]  = "var_char";
    varname[3]  = "var_short";
    varname[4]  = "var_int";
    varname[5]  = "var_float";
    varname[6]  = "var_double";
    varname[7]  = "var_ubyte";
    varname[8]  = "var_ushort";
    varname[9]  = "var_uint";
    varname[10] = "var_int64";
    varname[11] = "var_uint64";
    for (i=NC_BYTE; i<=NC_UINT64; i++) {
        err = ncmpi_def_var(ncid, varname[i], i, 1, &dimid, &varid[i]); ERR0
    }

    for (i=0; i<1024; i++) buf[i]=i;
    err = ncmpi_put_att_text(ncid, NC_GLOBAL, "att_text", 3, (char*)buf); ERR0
    ERR(NC_NOERR, "text", NC_CHAR)

    for (i=0; i<1024; i++) buf[i]=0;
    for (i=NC_BYTE; i<=NC_UINT64; i++) {
        int expect_err = NC_NOERR;

        if (i == NC_CHAR) expect_err = NC_ECHAR;

        sprintf(attname,"att_uchar_for_var_%s",varname[i]);
        err = ncmpi_put_att_uchar    (ncid, NC_GLOBAL, attname, i, 3, (unsigned char*)      buf);
        ERR(expect_err, "uchar", i)
        sprintf(attname,"att_schar_for_var_%s",varname[i]);
        err = ncmpi_put_att_schar    (ncid, NC_GLOBAL, attname, i, 3, (signed char*)        buf);
        ERR(expect_err, "schar", i)
        sprintf(attname,"att_short_for_var_%s",varname[i]);
        err = ncmpi_put_att_short    (ncid, NC_GLOBAL, attname, i, 3, (short*)              buf);
        ERR(expect_err, "short", i)
        sprintf(attname,"att_int_for_var_%s",varname[i]);
        err = ncmpi_put_att_int      (ncid, NC_GLOBAL, attname, i, 3, (int*)                buf);
        ERR(expect_err, "int", i)
        sprintf(attname,"att_float_for_var_%s",varname[i]);
        err = ncmpi_put_att_float    (ncid, NC_GLOBAL, attname, i, 3, (float*)              buf);
        ERR(expect_err, "float", i)
        sprintf(attname,"att_double_for_var_%s",varname[i]);
        err = ncmpi_put_att_double   (ncid, NC_GLOBAL, attname, i, 3, (double*)             buf);
        ERR(expect_err, "double", i)
        sprintf(attname,"att_ushort_for_var_%s",varname[i]);
        err = ncmpi_put_att_ushort   (ncid, NC_GLOBAL, attname, i, 3, (unsigned short*)     buf);
        ERR(expect_err, "ushort", i)
        sprintf(attname,"att_uint_for_var_%s",varname[i]);
        err = ncmpi_put_att_uint     (ncid, NC_GLOBAL, attname, i, 3, (unsigned int*)       buf);
        ERR(expect_err, "uint", i)
        sprintf(attname,"att_longlong_for_var_%s",varname[i]);
        err = ncmpi_put_att_longlong (ncid, NC_GLOBAL, attname, i, 3, (long long*)          buf);
        ERR(expect_err, "longlong", i)
        sprintf(attname,"att_ulonglong_for_var_%s",varname[i]);
        err = ncmpi_put_att_ulonglong(ncid, NC_GLOBAL, attname, i, 3, (unsigned long long*) buf);
        ERR(expect_err, "ulonglong", i)
    }

    for (i=NC_BYTE; i<=NC_UINT64; i++) {
        int expect_err = NC_NOERR;

        if (i == NC_CHAR) continue;

        sprintf(attname,"att_uchar_for_var_%s",varname[i]);
        err = ncmpi_get_att_uchar    (ncid, NC_GLOBAL, attname, (unsigned char*)      buf);
        ERR(expect_err, "uchar", i)
        sprintf(attname,"att_schar_for_var_%s",varname[i]);
        err = ncmpi_get_att_schar    (ncid, NC_GLOBAL, attname, (signed char*)        buf);
        ERR(expect_err, "schar", i)
        sprintf(attname,"att_short_for_var_%s",varname[i]);
        err = ncmpi_get_att_short    (ncid, NC_GLOBAL, attname, (short*)              buf);
        ERR(expect_err, "short", i)
        sprintf(attname,"att_int_for_var_%s",varname[i]);
        err = ncmpi_get_att_int      (ncid, NC_GLOBAL, attname, (int*)                buf);
        ERR(expect_err, "int", i)
        sprintf(attname,"att_float_for_var_%s",varname[i]);
        err = ncmpi_get_att_float    (ncid, NC_GLOBAL, attname, (float*)              buf);
        ERR(expect_err, "float", i)
        sprintf(attname,"att_double_for_var_%s",varname[i]);
        err = ncmpi_get_att_double   (ncid, NC_GLOBAL, attname, (double*)             buf);
        ERR(expect_err, "double", i)
        sprintf(attname,"att_ushort_for_var_%s",varname[i]);
        err = ncmpi_get_att_ushort   (ncid, NC_GLOBAL, attname, (unsigned short*)     buf);
        ERR(expect_err, "ushort", i)
        sprintf(attname,"att_uint_for_var_%s",varname[i]);
        err = ncmpi_get_att_uint     (ncid, NC_GLOBAL, attname, (unsigned int*)       buf);
        ERR(expect_err, "uint", i)
        sprintf(attname,"att_longlong_for_var_%s",varname[i]);
        err = ncmpi_get_att_longlong (ncid, NC_GLOBAL, attname, (long long*)          buf);
        ERR(expect_err, "longlong", i)
        sprintf(attname,"att_ulonglong_for_var_%s",varname[i]);
        err = ncmpi_get_att_ulonglong(ncid, NC_GLOBAL, attname, (unsigned long long*) buf);
        ERR(expect_err, "ulonglong", i)
    }

    err = ncmpi_close(ncid); ERR0

    /* check if PnetCDF freed all internal malloc */
    MPI_Offset malloc_size, sum_size;
    err = ncmpi_inq_malloc_size(&malloc_size);
    if (err == NC_NOERR) {
        MPI_Reduce(&malloc_size, &sum_size, 1, MPI_OFFSET, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0 && sum_size > 0)
            printf("heap memory allocated by PnetCDF internally has %lld bytes yet to be freed\n",
                   sum_size);
    }

    MPI_Allreduce(MPI_IN_PLACE, &nerrs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        if (nerrs) printf(FAIL_STR,nerrs);
        else       printf(PASS_STR);
    }

    MPI_Finalize();

    return 0;
}

static char* err_code_name(int err)
{
    static char unknown_str[32];
    switch (err) {
        case (NC_NOERR):			return "NC_NOERR";
        case (NC_EBADID):			return "NC_EBADID";
        case (NC_ENFILE):			return "NC_ENFILE";
        case (NC_EEXIST):			return "NC_EEXIST";
        case (NC_EINVAL):			return "NC_EINVAL";
        case (NC_EPERM):			return "NC_EPERM";
        case (NC_ENOTINDEFINE):			return "NC_ENOTINDEFINE";
        case (NC_EINDEFINE):			return "NC_EINDEFINE";
        case (NC_EINVALCOORDS):			return "NC_EINVALCOORDS";
        case (NC_EMAXDIMS):			return "NC_EMAXDIMS";
        case (NC_ENAMEINUSE):			return "NC_ENAMEINUSE";
        case (NC_ENOTATT):			return "NC_ENOTATT";
        case (NC_EMAXATTS):			return "NC_EMAXATTS";
        case (NC_EBADTYPE):			return "NC_EBADTYPE";
        case (NC_EBADDIM):			return "NC_EBADDIM";
        case (NC_EUNLIMPOS):			return "NC_EUNLIMPOS";
        case (NC_EMAXVARS):			return "NC_EMAXVARS";
        case (NC_ENOTVAR):			return "NC_ENOTVAR";
        case (NC_EGLOBAL):			return "NC_EGLOBAL";
        case (NC_ENOTNC):			return "NC_ENOTNC";
        case (NC_ESTS):				return "NC_ESTS";
        case (NC_EMAXNAME):			return "NC_EMAXNAME";
        case (NC_EUNLIMIT):			return "NC_EUNLIMIT";
        case (NC_ENORECVARS):			return "NC_ENORECVARS";
        case (NC_ECHAR):			return "NC_ECHAR";
        case (NC_EEDGE):			return "NC_EEDGE";
        case (NC_ESTRIDE):			return "NC_ESTRIDE";
        case (NC_EBADNAME):			return "NC_EBADNAME";
        case (NC_ERANGE):			return "NC_ERANGE";
        case (NC_ENOMEM):			return "NC_ENOMEM";
        case (NC_EVARSIZE):			return "NC_EVARSIZE";
        case (NC_EDIMSIZE):			return "NC_EDIMSIZE";
        case (NC_ETRUNC):			return "NC_ETRUNC";
        default:
              sprintf(unknown_str,"Unknown code %d",err);
    }
    return unknown_str;
}


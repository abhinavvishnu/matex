/*********************************************************************
 *
 *  Copyright (C) 2015, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *********************************************************************/
/* $Id: last_large_var.c 2291 2016-01-03 05:14:45Z wkliao $ */

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * This program tests the special case when there is no record variable, the
 * last fixed-size variable can be larger than 2GiB if its starting file offset
 * is less than 2GiB. See NetCDF Classic Format Limitations (The NetCDF Users
 * Guide).  Quoted here:
 * "If you don't use the unlimited dimension, only one variable can exceed 2
 * GiB in size, but it can be as large as the underlying file system permits.
 * It must be the last variable in the dataset, and the offset to the beginning
 * of this variable must be less than about 2 GiB."
 * http://www.unidata.ucar.edu/software/netcdf/old_docs/docs_3_6_3/netcdf-c/nc_005fcreate.html
 *
 *    To compile:
 *        mpicc -O2 last_large_var.c -o last_large_var -lpnetcdf
 *
 * Example commands for MPI run and outputs from running ncmpidump on the
 * NC file produced by this example program:
 *
 *    % mpiexec -n 4 ./last_large_var /pvfs2/wkliao/testfile.nc
 *
 *    % ncmpidump /pvfs2/wkliao/testfile.nc
 *    netcdf testfile {
 *    // file format: CDF-1
 *    netcdf testfile {
 *    dimensions:
 *    	    Y = 4 ;
 *    	    X = 5 ;
 *    	    YY = 66661 ;
 *    	    XX = 66661 ;
 *    variables:
 *    	    int var(Y, X) ;
 *    	    float var_last(YY, XX) ;
 *    }
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pnetcdf.h>

#include <testutils.h>

#define ERR {if(err!=NC_NOERR){printf("Error at line=%d: %s\n", __LINE__, ncmpi_strerror(err)); nerrs++;}}

static
int check_last_var(char *filename)
{
    int err, nerrs=0, ncid, cmode, varid, dimid[4];

    /* create a new file ---------------------------------------------------*/
    cmode = NC_CLOBBER;
    err = ncmpi_create(MPI_COMM_WORLD, filename, cmode, MPI_INFO_NULL, &ncid);
    ERR

    err = ncmpi_def_dim(ncid, "Y", NC_UNLIMITED, &dimid[0]); ERR
    err = ncmpi_def_dim(ncid, "X", 5, &dimid[1]); ERR
    err = ncmpi_def_dim(ncid, "YY", 66661, &dimid[2]); ERR
    err = ncmpi_def_dim(ncid, "XX", 66661, &dimid[3]); ERR

    /* define only fixed-size variables */
    err = ncmpi_def_var(ncid, "var", NC_INT, 1, dimid+1, &varid); ERR
    err = ncmpi_def_var(ncid, "var_last", NC_FLOAT, 2, dimid+2, &varid); ERR

    err = ncmpi_enddef(ncid); ERR
    err = ncmpi_close(ncid); ERR

    return nerrs;
}

static
int check_rec_var(char *filename)
{
    int err, nerrs=0, ncid, cmode, varid, dimid[4];

    /* create a new file ---------------------------------------------------*/
    cmode = NC_CLOBBER;
    err = ncmpi_create(MPI_COMM_WORLD, filename, cmode, MPI_INFO_NULL, &ncid);
    ERR

    err = ncmpi_def_dim(ncid, "Y", NC_UNLIMITED, &dimid[0]); ERR
    err = ncmpi_def_dim(ncid, "X", 5, &dimid[1]); ERR
    err = ncmpi_def_dim(ncid, "YY", 66661, &dimid[2]); ERR
    err = ncmpi_def_dim(ncid, "XX", 66661, &dimid[3]); ERR

    /* define a record variable */
    err = ncmpi_def_var(ncid, "var", NC_INT, 1, dimid, &varid); ERR
    err = ncmpi_def_var(ncid, "var_last", NC_FLOAT, 2, dimid+2, &varid); ERR

    err = ncmpi_enddef(ncid);
    if (err != NC_EVARSIZE) {
        printf("\nError at line=%d: expecting error code NC_EVARSIZE but got %s\n",__LINE__,nc_err_code_name(err));
        nerrs++;
    }
    err = ncmpi_close(ncid);
    if (err != NC_EVARSIZE) {
        printf("\nError at line=%d: expecting error code NC_EVARSIZE but got %s\n",__LINE__,nc_err_code_name(err));
        nerrs++;
    }

    return nerrs;
}

static
int check_not_last_var(char *filename)
{
    int err, nerrs=0, ncid, cmode, varid, dimid[4];

    /* create a new file ---------------------------------------------------*/
    cmode = NC_CLOBBER;
    err = ncmpi_create(MPI_COMM_WORLD, filename, cmode, MPI_INFO_NULL, &ncid);
    ERR

    err = ncmpi_def_dim(ncid, "Y", NC_UNLIMITED, &dimid[0]); ERR
    err = ncmpi_def_dim(ncid, "X", 5, &dimid[1]); ERR
    err = ncmpi_def_dim(ncid, "YY", 66661, &dimid[2]); ERR
    err = ncmpi_def_dim(ncid, "XX", 66661, &dimid[3]); ERR

    /* the large variable is not the last */
    err = ncmpi_def_var(ncid, "var_last", NC_FLOAT, 2, dimid+2, &varid); ERR
    err = ncmpi_def_var(ncid, "var", NC_INT, 1, dimid+1, &varid); ERR

    err = ncmpi_enddef(ncid);
    if (err != NC_EVARSIZE) {
        printf("\nError at line=%d: expecting error code NC_EVARSIZE but got %s\n",__LINE__,nc_err_code_name(err));
        nerrs++;
    }

    err = ncmpi_close(ncid);
    if (err != NC_EVARSIZE) {
        printf("\nError at line=%d: expecting error code NC_EVARSIZE but got %s\n",__LINE__,nc_err_code_name(err));
        nerrs++;
    }
    return nerrs;
}

static
int check_add_var(char *filename)
{
    int err, nerrs=0, ncid, cmode, varid, dimid[4];

    /* create a new file ---------------------------------------------------*/
    cmode = NC_CLOBBER;
    err = ncmpi_create(MPI_COMM_WORLD, filename, cmode, MPI_INFO_NULL, &ncid);
    ERR

    err = ncmpi_def_dim(ncid, "Y", NC_UNLIMITED, &dimid[0]); ERR
    err = ncmpi_def_dim(ncid, "X", 5, &dimid[1]); ERR
    err = ncmpi_def_dim(ncid, "YY", 66661, &dimid[2]); ERR
    err = ncmpi_def_dim(ncid, "XX", 66661, &dimid[3]); ERR

    err = ncmpi_def_var(ncid, "var", NC_INT, 1, dimid+1, &varid); ERR
    err = ncmpi_def_var(ncid, "var_last", NC_FLOAT, 2, dimid+2, &varid); ERR

    err = ncmpi_enddef(ncid); ERR

    /* add a new fixed-size variable */
    err = ncmpi_redef(ncid); ERR
    err = ncmpi_def_var(ncid, "var_new", NC_INT, 2, dimid, &varid); ERR

    err = ncmpi_enddef(ncid);
    if (err != NC_EVARSIZE) {
        printf("\nError at line=%d: expecting error code NC_EVARSIZE but got %s\n",__LINE__,nc_err_code_name(err));
        nerrs++;
    }

    err = ncmpi_close(ncid);
    if (err != NC_EVARSIZE) {
        printf("\nError at line=%d: expecting error code NC_EVARSIZE but got %s\n",__LINE__,nc_err_code_name(err));
        nerrs++;
    }
    return nerrs;
}

static
int check_var_offset(char *filename)
{
    int err, nerrs=0, ncid, cmode, varid, dimid[4];

    /* create a new file ---------------------------------------------------*/
    cmode = NC_CLOBBER;
    err = ncmpi_create(MPI_COMM_WORLD, filename, cmode, MPI_INFO_NULL, &ncid);
    ERR

    err = ncmpi_def_dim(ncid, "Y", NC_UNLIMITED, &dimid[0]); ERR
    err = ncmpi_def_dim(ncid, "X", 5, &dimid[1]); ERR
    err = ncmpi_def_dim(ncid, "YY", 66661, &dimid[2]); ERR
    err = ncmpi_def_dim(ncid, "XX", 66661, &dimid[3]); ERR

    err = ncmpi_def_var(ncid, "var", NC_INT, 1, dimid+1, &varid); ERR
    err = ncmpi_def_var(ncid, "var_last", NC_FLOAT, 2, dimid+2, &varid); ERR

    /* make the file header size larger than 2 GiB */
    err = ncmpi__enddef(ncid, 2147483648LL, 1, 1, 1);
    if (err != NC_EVARSIZE) {
        printf("\nError at line=%d: expecting error code NC_EVARSIZE but got %s\n",__LINE__,nc_err_code_name(err));
        nerrs++;
    }

    err = ncmpi_close(ncid); ERR
    return nerrs;
}

int main(int argc, char** argv)
{
    char filename[256];
    int  rank, nprocs, err, nerrs=0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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
        sprintf(cmd_str, "*** TESTING C   %s for last large var in CDF-1/2", argv[0]);
        printf("%-66s ------ ", cmd_str); fflush(stdout);
    }

    nerrs += check_last_var(filename);
    nerrs += check_rec_var(filename);
    nerrs += check_not_last_var(filename);
    nerrs += check_add_var(filename);
    nerrs += check_var_offset(filename);

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


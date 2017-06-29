/*
 *  Copyright (C) 2014, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *  $Id: inq_recsize.c 2133 2015-09-26 19:16:01Z wkliao $
 */

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * This program tests if one can get the size of record block correctly. The
 * record block size is the sum of individual record of all record variables.
 * It first defines some number of record and fixed-size variables and then
 * calls the API ncmpi_inq_recsize() and varify if the numbers are correct.
 *
 * The compile and run commands are given below. This program is to be run on
 * one MPI process.
 *
 *    % mpicc -g -o inq_recsize inq_recsize.c -lpnetcdf
 *
 *    % mpiexec -l -n 1 inq_recsize testfile.nc
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <pnetcdf.h>

#include <testutils.h>

#define ERR {if(err!=NC_NOERR)printf("Error at line=%d: %s\n", __LINE__, ncmpi_strerror(err));}

int main(int argc, char** argv) {
    char *filename="testfile.nc";
    int nerrs, rank, nprocs, err;
    int ncid, cmode, varid[7], dimid[3];
    MPI_Offset expected_recsize, recsize;
    MPI_Info info=MPI_INFO_NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc > 2) {
        if (!rank) printf("Usage: %s [filename]\n",argv[0]);
        goto fn_exit;
    }
    if (argc == 2) filename = argv[1];

    if (rank == 0) {
        char cmd_str[256];
        sprintf(cmd_str, "*** TESTING C   %s for inquiring record size ", argv[0]);
        printf("%-66s ------ ", cmd_str); fflush(stdout);
    }

    /* printf("PnetCDF version string: \"%s\"\n", ncmpi_inq_libvers()); */

    /* create a new file for writing ----------------------------------------*/
    cmode = NC_CLOBBER;
    err = ncmpi_create(MPI_COMM_WORLD, filename, cmode, info, &ncid); ERR

    /* define dimension and variable */
    err = ncmpi_def_dim(ncid, "REC_DIM", NC_UNLIMITED, &dimid[0]); ERR
    err = ncmpi_def_dim(ncid, "Y",       2,            &dimid[1]); ERR
    err = ncmpi_def_dim(ncid, "X",       10,           &dimid[2]); ERR

    nerrs = 0;
    expected_recsize = 0;

    /* define some record variables */
    err = ncmpi_def_var(ncid, "REC_VAR_1", NC_INT, 1, dimid, &varid[0]); ERR
    expected_recsize += sizeof(int);
    err = ncmpi_def_var(ncid, "REC_VAR_2", NC_INT, 3, dimid, &varid[1]); ERR
    expected_recsize += 2 * 10 * sizeof(int);
    err = ncmpi_def_var(ncid, "REC_VAR_3", NC_INT, 2, dimid, &varid[2]); ERR
    expected_recsize += 2 * sizeof(int);
    err = ncmpi_def_var(ncid, "REC_VAR_4", NC_INT, 1, dimid, &varid[3]); ERR
    expected_recsize += sizeof(int);

    /* define some fixed-size variables */
    err = ncmpi_def_var(ncid, "FIX_VAR_1", NC_INT, 2, dimid+1, &varid[4]); ERR
    err = ncmpi_def_var(ncid, "FIX_VAR_2", NC_INT, 1, dimid+1, &varid[5]); ERR
    err = ncmpi_def_var(ncid, "FIX_VAR_3", NC_INT, 1, dimid+2, &varid[6]); ERR

    err = ncmpi_enddef(ncid); ERR

    err = ncmpi_inq_recsize(ncid, &recsize); ERR
    if (expected_recsize != recsize) {
        printf("Error at line %d: expecting record size %lld but got %lld\n", __LINE__,expected_recsize, recsize);
        nerrs++;
    }

    err = ncmpi_close(ncid); ERR

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

fn_exit:
    MPI_Finalize();
    return 0;
}


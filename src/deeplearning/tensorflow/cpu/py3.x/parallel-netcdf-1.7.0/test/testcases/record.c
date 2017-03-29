/*
 *  Copyright (C) 2014, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *  $Id: record.c 2219 2015-12-11 22:30:03Z wkliao $
 */

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * This program tests if the number of records is updated correctly. It first
 * writes to the 2nd record of a variable, followed by writing to the 1st
 * record. After the 2nd write, a call to ncmpi_inq_dimlen() or ncmpi_inq_dim()
 * should report seeing 2 records. Then, it does a similar test under
 * independent data mode. The same test will repeat for 3 cases:
 * 1) there is only one 1D record variable
 * 2) there is only one 3D record variable
 * 3) there are one 1D record variable and one 3D record variable
 *
 * The compile and run commands are given below. This program is to be run on
 * one MPI process.
 *
 *    % mpicc -g -o record record.c -lpnetcdf
 *
 *    % mpiexec -l -n 1 record record.nc
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <pnetcdf.h>

#include <testutils.h>

#define ERR {if(err!=NC_NOERR) {nerrs++; printf("Error at line=%d: %s\n", __LINE__, ncmpi_strerror(err));}}

static
int test_only_record_var_1D(char *filename)
{
    int ncid, cmode, varid, dimid, buf[20], err, nerrs=0;
    MPI_Offset start, count, length;
    MPI_Info info=MPI_INFO_NULL;

    /* create a new file for writing ----------------------------------------*/
    cmode = NC_CLOBBER;
    err = ncmpi_create(MPI_COMM_SELF, filename, cmode, info, &ncid); ERR

    /* define dimension and variable */
    err = ncmpi_def_dim(ncid, "REC_DIM", NC_UNLIMITED, &dimid); ERR
    err = ncmpi_def_var(ncid, "REC_VAR_1D", NC_INT, 1, &dimid, &varid); ERR
    err = ncmpi_enddef(ncid); ERR

    /* write the 2nd record first */
    buf[0] = 91;
    start = 1; count = 1;
    err = ncmpi_put_vara_int_all(ncid, varid, &start, &count, buf); ERR

    /* write the 1st record now */
    buf[0] = 90;
    start = 0; count = 1;
    err = ncmpi_put_vara_int_all(ncid, varid, &start, &count, buf); ERR

    err = ncmpi_inq_dimlen(ncid, dimid, &length); ERR
    if (length != 2) {
        printf("Error: expecting 2 records, but got %lld record(s)\n",length);
        nerrs++;
    }

    if (nerrs == 0) { /* test independent data mode */
        err = ncmpi_begin_indep_data(ncid); ERR
        /* write the 4th record */
        buf[0] = 93;
        start = 3; count = 1;
        err = ncmpi_put_vara_int(ncid, varid, &start, &count, buf); ERR

        /* write the 3rd record */
        buf[0] = 92; buf[1] = 93;
        start = 2; count = 2;
        err = ncmpi_put_vara_int(ncid, varid, &start, &count, buf); ERR

        err = ncmpi_inq_dimlen(ncid, dimid, &length); ERR
        if (length != 4) {
            printf("Error: expecting 4 records, but got %lld record(s)\n",
                   length);
            nerrs++;
        }
        err = ncmpi_end_indep_data(ncid); ERR
    }
    err = ncmpi_close(ncid); ERR
    return nerrs;
}

static
int test_only_record_var_3D(char *filename)
{
    int i, ncid, cmode, varid, dimid[3], buf[20], err, nerrs=0;
    MPI_Offset start[3], count[3], length;
    MPI_Info info=MPI_INFO_NULL;

    /* create a new file for writing ----------------------------------------*/
    cmode = NC_CLOBBER;
    err = ncmpi_create(MPI_COMM_SELF, filename, cmode, info, &ncid); ERR

    /* define dimension and variable */
    err = ncmpi_def_dim(ncid, "REC_DIM", NC_UNLIMITED, &dimid[0]); ERR
    err = ncmpi_def_dim(ncid, "FIX_DIM_Y", 2,          &dimid[1]); ERR
    err = ncmpi_def_dim(ncid, "FIX_DIM_X", 10,         &dimid[2]); ERR
    err = ncmpi_def_var(ncid, "REC_VAR_3D", NC_INT, 3, dimid, &varid); ERR
    err = ncmpi_enddef(ncid); ERR

    start[1] = 0; start[2] = 0; count[0] = 1; count[1] = 2; count[2] = 5;

    /* write the 2nd record first */
    for (i=0; i<20; i++) buf[i] = 91;
    start[0] = 1;
    err = ncmpi_put_vara_int_all(ncid, varid, start, count, buf); ERR

    /* write the 1st record now */
    for (i=0; i<20; i++) buf[i] = 90;
    start[0] = 0;
    err = ncmpi_put_vara_int_all(ncid, varid, start, count, buf); ERR

    err = ncmpi_inq_dimlen(ncid, dimid[0], &length); ERR
    if (length != 2) {
        printf("Error: expecting 2 records, but got %lld record(s)\n",length);
        nerrs++;
    }

    if (nerrs == 0) { /* test independent data mode */
        err = ncmpi_begin_indep_data(ncid); ERR
        /* write the 4th record */
        for (i=0; i<20; i++) buf[i] = 93;
        start[0] = 3;
        err = ncmpi_put_vara_int(ncid, varid, start, count, buf); ERR

        /* write the 3rd record */
        for (i=0; i<20; i++) buf[i] = 92;
        start[0] = 2;
        err = ncmpi_put_vara_int(ncid, varid, start, count, buf); ERR

        err = ncmpi_inq_dimlen(ncid, dimid[0], &length); ERR
        if (length != 4) {
            printf("Error: expecting 4 records, but got %lld record(s)\n",
                   length);
            nerrs++;
        }
        err = ncmpi_end_indep_data(ncid); ERR
    }
    err = ncmpi_close(ncid); ERR
    return nerrs;
}

static
int test_two_record_var(char *filename)
{
    int i, ncid, cmode, varid[2], dimid[3], buf[20], err, nerrs=0;
    MPI_Offset start[3], count[3], length;
    MPI_Info info=MPI_INFO_NULL;

    /* create a new file for writing ----------------------------------------*/
    cmode = NC_CLOBBER;
    err = ncmpi_create(MPI_COMM_SELF, filename, cmode, info, &ncid); ERR

    /* define dimension and variable */
    err = ncmpi_def_dim(ncid, "REC_DIM", NC_UNLIMITED, &dimid[0]); ERR
    err = ncmpi_def_dim(ncid, "FIX_DIM_Y", 2,          &dimid[1]); ERR
    err = ncmpi_def_dim(ncid, "FIX_DIM_X", 10,         &dimid[2]); ERR
    err = ncmpi_def_var(ncid, "REC_VAR_1D", NC_INT, 1, dimid, &varid[0]); ERR
    err = ncmpi_def_var(ncid, "REC_VAR_3D", NC_INT, 3, dimid, &varid[1]); ERR
    err = ncmpi_enddef(ncid); ERR

    /* REC_VAR_1D: write the 2nd record first */
    buf[0] = 91;
    start[0] = 1; count[0] = 1;
    err = ncmpi_put_vara_int_all(ncid, varid[0], start, count, buf); ERR

    /* write the 1st record now */
    buf[0] = 90;
    start[0] = 0; count[0] = 1;
    err = ncmpi_put_vara_int_all(ncid, varid[0], start, count, buf); ERR

    err = ncmpi_inq_dimlen(ncid, dimid[0], &length); ERR
    if (length != 2) {
        printf("Error: expecting 2 records, but got %lld record(s)\n",length);
        nerrs++;
    }

    if (nerrs == 0) { /* test independent data mode */
        err = ncmpi_begin_indep_data(ncid); ERR
        /* write the 4th record */
        buf[0] = 93;
        start[0] = 3; count[0] = 1;
        err = ncmpi_put_vara_int(ncid, varid[0], start, count, buf); ERR

        /* write the 3rd and 4th records */
        buf[0] = 92; buf[1] = 93;
        start[0] = 2; count[0] = 2;
        err = ncmpi_put_vara_int(ncid, varid[0], start, count, buf); ERR

        err = ncmpi_inq_dimlen(ncid, dimid[0], &length); ERR
        if (length != 4) {
            printf("Error: expecting 4 records, but got %lld record(s)\n",
                   length);
            nerrs++;
        }
        err = ncmpi_end_indep_data(ncid); ERR
    }

    /* REC_VAR_3D: write the 2nd record first */
    start[1] = 0; start[2] = 0; count[0] = 1; count[1] = 2; count[2] = 5;

    for (i=0; i<20; i++) buf[i] = 91;
    start[0] = 1;
    err = ncmpi_put_vara_int_all(ncid, varid[1], start, count, buf); ERR

    /* write the 1st record now */
    for (i=0; i<20; i++) buf[i] = 90;
    start[0] = 0;
    err = ncmpi_put_vara_int_all(ncid, varid[1], start, count, buf); ERR

    err = ncmpi_inq_dimlen(ncid, dimid[0], &length); ERR
    if (length != 4) {
        printf("Error: expecting 4 records, but got %lld record(s)\n",length);
        nerrs++;
    }

    if (nerrs == 0) { /* test independent data mode */
        err = ncmpi_begin_indep_data(ncid); ERR
        /* write the 4th record */
        for (i=0; i<20; i++) buf[i] = 93;
        start[0] = 3;
        err = ncmpi_put_vara_int(ncid, varid[1], start, count, buf); ERR

        /* write the 3rd record */
        for (i=0; i<20; i++) buf[i] = 92;
        start[0] = 2;
        err = ncmpi_put_vara_int(ncid, varid[1], start, count, buf); ERR

        err = ncmpi_inq_dimlen(ncid, dimid[0], &length); ERR
        if (length != 4) {
            printf("Error: expecting 4 records, but got %lld record(s)\n",
                   length);
            nerrs++;
        }
        err = ncmpi_end_indep_data(ncid); ERR
    }
    err = ncmpi_close(ncid); ERR
    return nerrs;
}

int main(int argc, char** argv) {
    char *filename="testfile.nc";
    int nerrs=0, rank, nprocs, err;

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
        sprintf(cmd_str, "*** TESTING C   %s for write records in reversed order", argv[0]);
        printf("%-66s ------ ", cmd_str); fflush(stdout);
    }

    if (rank >= 1) goto fn_exit; /* this test is for running 1 process */

    /* test only one 1D record variable */
    nerrs += test_only_record_var_1D(filename);

    /* test only one 3D record variable */
    nerrs += test_only_record_var_3D(filename);

    /* test two record variables */
    nerrs += test_two_record_var(filename);

    /* check if PnetCDF freed all internal malloc */
    MPI_Offset malloc_size;
    err = ncmpi_inq_malloc_size(&malloc_size);
    if (err == NC_NOERR && malloc_size > 0) /* this test is for running 1 process */
        printf("heap memory allocated by PnetCDF internally has %lld bytes yet to be freed\n",
               malloc_size);

fn_exit:
    MPI_Allreduce(MPI_IN_PLACE, &nerrs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        if (nerrs) printf(FAIL_STR,nerrs);
        else       printf(PASS_STR);
    }

    MPI_Finalize();
    return 0;
}


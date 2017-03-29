/*********************************************************************
 *
 *  Copyright (C) 2014, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *********************************************************************/
/* $Id: wait_after_indep.c 2133 2015-09-26 19:16:01Z wkliao $ */

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * This example tests if ncmpi_end_indep_data() works properly when nonblocking
 * APIs are called in the independent data mode, but the wait call is made later
 * in collective data mode.
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <pnetcdf.h>

#include <testutils.h>

#define NY 4
#define NX 10
#define NDIMS 2

#define ERR \
    if (err != NC_NOERR) { \
        printf("Error at line=%d: %s\n", __LINE__, ncmpi_strerror(err)); \
        nerrs++; \
    }

int main(int argc, char** argv)
{
    int i, j, rank, nprocs, err, nerrs=0;
    int ncid, varid, dimid[2], req, st;
    MPI_Offset start[2], count[2], stride[2];
    unsigned char buffer[NY][NX];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (rank == 0) {
        char cmd_str[256];
        sprintf(cmd_str, "*** TESTING C   %s for ncmpi_end_indep_data ", argv[0]);
        printf("%-66s ------ ",cmd_str);
    }

    err = ncmpi_create(MPI_COMM_WORLD, "testfile.nc", NC_CLOBBER|NC_64BIT_DATA,
                       MPI_INFO_NULL, &ncid);
    ERR

    err = ncmpi_def_dim(ncid, "Y", NC_UNLIMITED, &dimid[0]); ERR
    err = ncmpi_def_dim(ncid, "X", NX*nprocs,    &dimid[1]); ERR
    err = ncmpi_def_var(ncid, "var", NC_UBYTE, NDIMS, dimid, &varid); ERR
    err = ncmpi_enddef(ncid); ERR

    for (i=0; i<NY; i++) for (j=0; j<NX; j++) buffer[i][j] = rank;

     start[0] = 0;     start[1] = NX*rank;
     count[0] = NY/2;  count[1] = NX/2;
    stride[0] = 2;    stride[1] = 2;
    err = ncmpi_buffer_attach(ncid, NY*NX); ERR

    err = ncmpi_begin_indep_data(ncid); ERR
    err = ncmpi_bput_vars_uchar(ncid, varid, start, count, stride,
                                &buffer[0][0], &req);
    ERR
    err = ncmpi_end_indep_data(ncid); ERR

    /* calling wait API after exiting independent data mode on purpose */
    err = ncmpi_wait_all(ncid, 1, &req, &st); ERR

    err = ncmpi_buffer_detach(ncid); ERR
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

    MPI_Finalize();
    return 0;
}


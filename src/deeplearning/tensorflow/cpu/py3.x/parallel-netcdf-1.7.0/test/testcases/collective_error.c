/*
 *  Copyright (C) 2003, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *  $Id: collective_error.c 2133 2015-09-26 19:16:01Z wkliao $
 */

/* This test program checks if a collective API can be nicely aborted without
 * causing the program to hang. It runs on 2 processes. One process deliberately
 * produces an error (using an illegal start argument), while the other does not.
 */

#include <mpi.h>
#include <pnetcdf.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <testutils.h>

#define ERR { if (err!=NC_NOERR){printf("PE %d: error at line %d (%s)\n",rank,__LINE__,ncmpi_strerror(err)); nerrs++;}}
#define CHECK_ERROR(fn) { \
   if (rank == 0 && err != NC_NOERR) \
       printf("PE %d: %s error is %s\n",rank,fn,ncmpi_strerror(err)); \
   if (rank == 1 && err != NC_EINVALCOORDS) \
       printf("PE %d: %s error code should be NC_EINVALCOORDS but got %s",rank,fn,nc_err_code_name(err)); \
}

int main(int argc, char *argv[])
{
    char *filename="testfile.nc";
    int rank, nproc, ncid, err, nerrs=0, varid, dimids[1];
    int req, status, verbose;
    MPI_Offset start[1], count[1];
    double buf[2];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    if (argc > 2) {
        if (!rank) printf("Usage: %s [filename]\n",argv[0]);
        MPI_Finalize();
        return 0;
    }
    if (argc == 2) filename = argv[1];

    if (rank == 0) {
        char cmd_str[256];
        sprintf(cmd_str, "*** TESTING C   %s for collective abort ", argv[0]);
        printf("%-66s ------ ", cmd_str); fflush(stdout);
    }

    verbose = 0;
    if (nproc != 2 && rank == 0 && verbose)
        printf("Warning: %s is designed to run on 2 processes\n",argv[0]);

    /* Create a 2 element vector of doubles */
    err = ncmpi_create(MPI_COMM_WORLD, filename, NC_CLOBBER, MPI_INFO_NULL, &ncid);
    ERR

    err = ncmpi_def_dim(ncid, "dim", 2, &dimids[0]);
    ERR

    err = ncmpi_def_var(ncid, "var", NC_DOUBLE, 1, dimids, &varid);
    ERR

    err = ncmpi_enddef(ncid);
    ERR

    if (rank == 0) {
        start[0] = 0;
        count[0] = 2;
    } else if (rank == 1) {
        start[0] = 2; /* illegal for a start > defined shape */
        count[0] = 0;
    }
    else
        count[0] = 0;

    err = ncmpi_put_vara_all(ncid, varid, start, count,
			     buf, count[0], MPI_DOUBLE);
    CHECK_ERROR("ncmpi_put_vara_all")

    err = ncmpi_put_vara_double_all(ncid, varid, start, count, buf);
    CHECK_ERROR("ncmpi_put_vara_double_all")

    err = ncmpi_iput_vara_double(ncid, varid, start, count, buf, &req);
    CHECK_ERROR("ncmpi_iput_vara_double")

    err = ncmpi_wait_all(ncid, 1, &req, &status);
    ERR

    err = ncmpi_get_vara_all(ncid, varid, start, count,
			     buf, count[0], MPI_DOUBLE);
    CHECK_ERROR("ncmpi_get_vara_all")

    err = ncmpi_get_vara_double_all(ncid, varid, start, count, buf);
    CHECK_ERROR("ncmpi_get_vara_double_all")

    err = ncmpi_iget_vara_double(ncid, varid, start, count, buf, &req);
    CHECK_ERROR("ncmpi_iget_vara_double")

    err = ncmpi_wait_all(ncid, 1, &req, &status);
    ERR

    err = ncmpi_close(ncid);
    ERR

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
    return (nerrs == 0) ? 0 : 1;
}

/*
 *  Copyright (C) 2003, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *  $Id: ncmpi_vars_null_stride.c 2133 2015-09-26 19:16:01Z wkliao $
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <pnetcdf.h>

#include <testutils.h>

#define HANDLE_ERROR(err) { \
    nerrs++; \
    fprintf(stderr, "Error at line %d: %s\n", __LINE__, ncmpi_strerror(err)); \
    goto fn_exit; \
}

#define NDIMS 1
int main(int argc, char **argv)
{
    int err, nerrs=0, ncfile, dimid, varid, ndims=NDIMS;
    int i, nprocs, rank;
    MPI_Offset start[NDIMS] = {0};
    MPI_Offset count[NDIMS] = {0};
    int buf[512];
    char *filename="testfile.nc";

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc > 2) {
        if (!rank) printf("Usage: %s [filename]\n",argv[0]);
        MPI_Finalize();
        return 0;
    }
    if (argc == 2) filename = argv[1];

    if (rank == 0) {
        char cmd_str[256];
        sprintf(cmd_str, "*** TESTING C   %s for NULL stride ", argv[0]);
        printf("%-66s ------ ", cmd_str); fflush(stdout);
    }

    err = ncmpi_create(MPI_COMM_WORLD, filename, 0, MPI_INFO_NULL, &ncfile);
    if (err != NC_NOERR) HANDLE_ERROR(err)

    err = ncmpi_def_dim(ncfile, "d1", nprocs, &dimid);
    if (err != NC_NOERR) HANDLE_ERROR(err)

    err = ncmpi_def_var(ncfile, "v1", NC_INT, ndims, &dimid, &varid);
    if (err != NC_NOERR) HANDLE_ERROR(err)

    err = ncmpi_enddef(ncfile);
    if (err != NC_NOERR) HANDLE_ERROR(err)

    start[0] = rank;
    count[0] = 1;
    for (i=0; i<512; i++) buf[i] = rank;
    err = ncmpi_put_vars_int_all(ncfile, varid, start, count, NULL, buf);
    if (err != NC_NOERR) HANDLE_ERROR(err)

    err = ncmpi_close(ncfile);
    if (err != NC_NOERR) HANDLE_ERROR(err)

    /* check if PnetCDF freed all internal malloc */
    MPI_Offset malloc_size, sum_size;
    err = ncmpi_inq_malloc_size(&malloc_size);
    if (err == NC_NOERR) {
        MPI_Reduce(&malloc_size, &sum_size, 1, MPI_OFFSET, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0 && sum_size > 0)
            printf("heap memory allocated by PnetCDF internally has %lld bytes yet to be freed\n",
                   sum_size);
    }

fn_exit:
    MPI_Allreduce(MPI_IN_PLACE, &nerrs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        if (nerrs) printf(FAIL_STR,nerrs);
        else       printf(PASS_STR);
    }

    MPI_Finalize();
    return 0;
}

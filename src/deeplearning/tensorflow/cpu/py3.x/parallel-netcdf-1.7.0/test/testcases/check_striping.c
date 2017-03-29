/*
 *  Copyright (C) 2015, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *  $Id: check_striping.c 2133 2015-09-26 19:16:01Z wkliao $
 */

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * This program tests whether the file striping size and count retrieved from
 * MPI-IO hints are consistent among all MPI processes.
 *
 * The compile and run commands are given below.
 *
 *    % mpicc -g -o get_striping get_striping.c -lpnetcdf
 *
 *    % mpiexec -l -n 4 get_striping testfile.nc
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>
#include <pnetcdf.h>

#include <testutils.h>

#define ERR {if(err!=NC_NOERR){nerrs++;printf("Error at line=%d: %s\n", __LINE__, ncmpi_strerror(err));}}

int main(int argc, char** argv) {
    char *filename="testfile.nc";
    int rank, nprocs, err, nerrs=0, ncid, cmode;
    int striping_size, striping_count, root_striping_size, root_striping_count;

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
        sprintf(cmd_str, "*** TESTING C   %s for strining info ", argv[0]);
        printf("%-66s ------ ", cmd_str); fflush(stdout);
    }

    cmode = NC_CLOBBER;
    err = ncmpi_create(MPI_COMM_WORLD, filename, cmode, MPI_INFO_NULL, &ncid); ERR
    err = ncmpi_enddef(ncid); ERR

    err = ncmpi_inq_striping(ncid, &striping_size, &striping_count); ERR

    root_striping_size  = striping_size;
    root_striping_count = striping_count;
    MPI_Bcast(&root_striping_size,  1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&root_striping_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (root_striping_size != striping_size) {
        printf("Error at PE %2d: inconsistent striping_size (root=%d local=%d)\n",
               rank, root_striping_size, striping_size);
        nerrs++;
    }
    if (root_striping_count != striping_count) {
        printf("Error at PE %2d: inconsistent striping_count (root=%d local=%d)\n",
               rank, root_striping_count, striping_count);
        nerrs++;
    }
/*
    if (nerrs == 0 && rank == 0)
        printf("Success: striping_size=%d striping_count=%d\n",striping_size,striping_count);
*/

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


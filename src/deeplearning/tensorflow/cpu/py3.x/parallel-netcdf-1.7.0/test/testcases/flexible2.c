/*********************************************************************
 *
 *  Copyright (C) 2014, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *********************************************************************/
/* $Id: flexible2.c 2133 2015-09-26 19:16:01Z wkliao $ */

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * This program tests PnetCDF flexible APIs, ncmpi_put_vara_all(),
 * ncmpi_iput_vara() to write two 2D array variables (one is of 4-byte
 * integer byte and the other float type) in parallel. It then uses flexible
 * get/iget APIs to read data back and check the contents. The program first
 * defines 2 netCDF variables of sizes
 *    var_zy: NZ*nprocs x NY
 *    var_yx: NY x NX*nprocs
 *
 * The data partitioning patterns on the 2 variables are row-wise and
 * column-wise, respectively. Each process writes a subarray of size
 * NZ x NY and NY x NX to var_zy and var_yx, respectively.
 * Both local buffers have a ghost cell of length 3 surrounded along each
 * dimension.
 *
 * The compile and run commands are given below.
 *
 *    % mpicc -O2 -o flexible2 flexible2.c -lpnetcdf
 *
 *    % mpiexec -l -n 4 ./flexible2 /pvfs2/wkliao/testfile.nc
 *
 *    % ncmpidump /pvfs2/wkliao/testfile.nc
 *    netcdf testfile {
 *    // file format: CDF-5 (big variables)
 *    dimensions:
 *            Z = 20 ;
 *            Y = 5 ;
 *            X = 20 ;
 *    variables:
 *            int var_zy(Z, Y) ;
 *            float var_yx(Y, X) ;
 *    data:
 *
 *     var_zy =
 *      0, 0, 0, 0, 0,
 *      0, 0, 0, 0, 0,
 *      0, 0, 0, 0, 0,
 *      0, 0, 0, 0, 0,
 *      0, 0, 0, 0, 0,
 *      1, 1, 1, 1, 1,
 *      1, 1, 1, 1, 1,
 *      1, 1, 1, 1, 1,
 *      1, 1, 1, 1, 1,
 *      1, 1, 1, 1, 1,
 *      2, 2, 2, 2, 2,
 *      2, 2, 2, 2, 2,
 *      2, 2, 2, 2, 2,
 *      2, 2, 2, 2, 2,
 *      2, 2, 2, 2, 2,
 *      3, 3, 3, 3, 3,
 *      3, 3, 3, 3, 3,
 *      3, 3, 3, 3, 3,
 *      3, 3, 3, 3, 3,
 *      3, 3, 3, 3, 3 ;
 *
 *     var_yx =
 *      0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
 *      0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
 *      0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
 *      0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
 *      0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3 ;
 *    }
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <string.h> /* strcpy() */
#include <assert.h>
#include <mpi.h>
#include <pnetcdf.h>

#include <testutils.h>

#define NZ 5
#define NY 5
#define NX 5

#define ERR {if(err!=NC_NOERR){printf("Error at line=%d: %s\n", __LINE__, ncmpi_strerror(err)); nerrs++;}}

int main(int argc, char** argv)
{
    char filename[256];
    int i, j, rank, nprocs, err, nerrs=0, req, status, ghost_len=3;
    int ncid, cmode, varid0, varid1, dimid[3], *buf_zy;
    int array_of_sizes[2], array_of_subsizes[2], array_of_starts[2];
    double *buf_yx;
    MPI_Offset start[2], count[2];
    MPI_Datatype  subarray;

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
        sprintf(cmd_str, "*** TESTING C   %s for flexible APIs ", argv[0]);
        printf("%-66s ------ ", cmd_str); fflush(stdout);
    }

    /* create a new file for writing ----------------------------------------*/
    cmode = NC_CLOBBER | NC_64BIT_DATA;
    err = ncmpi_create(MPI_COMM_WORLD, filename, cmode, MPI_INFO_NULL, &ncid);
    ERR

    /* define 3 dimensions */
    err = ncmpi_def_dim(ncid, "Z", NZ*nprocs, &dimid[0]); ERR
    err = ncmpi_def_dim(ncid, "Y", NY,        &dimid[1]); ERR
    err = ncmpi_def_dim(ncid, "X", NX*nprocs, &dimid[2]); ERR

    /* define a variable of size (NZ * nprocs) * NY */
    err = ncmpi_def_var(ncid, "var_zy", NC_INT,   2, &dimid[0], &varid0); ERR
    /* define a variable of size NY * (NX * nprocs) */
    err = ncmpi_def_var(ncid, "var_yx", NC_FLOAT, 2, &dimid[1], &varid1); ERR
    err = ncmpi_enddef(ncid); ERR

    /* var_zy is partitioned along Z dimension */
    array_of_sizes[0]    = NZ + 2*ghost_len;
    array_of_sizes[1]    = NY + 2*ghost_len;
    array_of_subsizes[0] = NZ;
    array_of_subsizes[1] = NY;
    array_of_starts[0]   = ghost_len;
    array_of_starts[1]   = ghost_len;
    MPI_Type_create_subarray(2, array_of_sizes, array_of_subsizes,
                             array_of_starts, MPI_ORDER_C, MPI_INT, &subarray);
    MPI_Type_commit(&subarray);

    int buffer_len = (NZ+2*ghost_len) * (NY+2*ghost_len);
    buf_zy = (int*) malloc(buffer_len * sizeof(int));
    for (i=0; i<buffer_len; i++) buf_zy[i] = rank;

    start[0] = NZ * rank; start[1] = 0;
    count[0] = NZ;        count[1] = NY;
    /* calling a blocking flexible API */
    err = ncmpi_put_vara_all(ncid, varid0, start, count, buf_zy, 1, subarray);
    ERR

    /* check the contents of put buffer */
    for (i=0; i<buffer_len; i++) {
        if (buf_zy[i] != rank) {
            printf("Error put buffer[%d] is altered\n",i);
            nerrs++;
        }
    }

    for (i=0; i<buffer_len; i++) buf_zy[i] = -1;
    /* calling a blocking flexible API */
    err = ncmpi_get_vara_all(ncid, varid0, start, count, buf_zy, 1, subarray);
    ERR

    /* check the contents of get buffer */
    for (i=0; i<array_of_sizes[0]; i++) {
        for (j=0; j<array_of_sizes[1]; j++) {
            int index = i*array_of_sizes[1] + j;
            if (i < ghost_len || ghost_len+array_of_subsizes[0] <= i ||
                j < ghost_len || ghost_len+array_of_subsizes[1] <= j) {
                if (buf_zy[index] != -1) {
                    printf("Unexpected get buffer[%d][%d]=%d\n",
                           i,j,buf_zy[index]);
                    nerrs++;
                }
            }
            else {
                if (buf_zy[index] != rank) {
                    printf("Unexpected get buffer[%d][%d]=%d\n",
                           i,j,buf_zy[index]);
                    nerrs++;
                }
            }
        }
    }
    free(buf_zy);
    MPI_Type_free(&subarray);

    /* var_yx is partitioned along X dimension */
    array_of_sizes[0]    = NY + 2*ghost_len;
    array_of_sizes[1]    = NX + 2*ghost_len;
    array_of_subsizes[0] = NY;
    array_of_subsizes[1] = NX;
    array_of_starts[0]   = ghost_len;
    array_of_starts[1]   = ghost_len;
    MPI_Type_create_subarray(2, array_of_sizes, array_of_subsizes,
                             array_of_starts, MPI_ORDER_C, MPI_DOUBLE,
                             &subarray);
    MPI_Type_commit(&subarray);

    buffer_len = (NY+2*ghost_len) * (NX+2*ghost_len);
    buf_yx = (double*) malloc(buffer_len * sizeof(double));
    for (i=0; i<buffer_len; i++) buf_yx[i] = rank;

    start[0] = 0;  start[1] = NX * rank;
    count[0] = NY; count[1] = NX;

    /* calling a non-blocking flexible API */
    err = ncmpi_iput_vara(ncid, varid1, start, count, buf_yx, 1, subarray,&req);
    ERR
    err = ncmpi_wait_all(ncid, 1, &req, &status); ERR
    err = status; ERR

    /* check the contents of put buffer */
    for (i=0; i<buffer_len; i++) {
        if (buf_yx[i] != rank) {
            printf("Error iput buffer[%d]=%f is altered\n",i,buf_yx[i]);
            nerrs++;
        }
    }

    for (i=0; i<buffer_len; i++) buf_yx[i] = -1;

    /* calling a non-blocking flexible API */
    err = ncmpi_iget_vara(ncid, varid1, start, count, buf_yx, 1, subarray,&req);
    ERR
    err = ncmpi_wait_all(ncid, 1, &req, &status); ERR
    err = status; ERR

    /* check the contents of iget buffer */
    for (i=0; i<array_of_sizes[0]; i++) {
        for (j=0; j<array_of_sizes[1]; j++) {
            int index = i*array_of_sizes[1] + j;
            if (i < ghost_len || ghost_len+array_of_subsizes[0] <= i ||
                j < ghost_len || ghost_len+array_of_subsizes[1] <= j) {
                if (buf_yx[index] != -1) {
                    printf("Unexpected get buffer[%d][%d]=%f\n",
                           i,j,buf_yx[index]);
                    nerrs++;
                }
            }
            else {
                if (buf_yx[index] != rank) {
                    printf("Unexpected get buffer[%d][%d]=%f\n",
                           i,j,buf_yx[index]);
                    nerrs++;
                }
            }
        }
    }
    free(buf_yx);
    MPI_Type_free(&subarray);

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


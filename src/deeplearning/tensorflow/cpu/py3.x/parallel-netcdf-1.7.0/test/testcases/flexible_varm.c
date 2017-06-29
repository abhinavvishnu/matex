/*********************************************************************
 *
 *  Copyright (C) 2014, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *********************************************************************/
/* $Id: flexible_varm.c 2133 2015-09-26 19:16:01Z wkliao $ */

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * This program tests PnetCDF flexible varm APIs, i.e. ncmpi_put_varm_all(),
 * ncmpi_get_varm_all(), and their nonblocking versions, to write a 2D array
 * double variable of size NY x NX*nproc in parallel.
 *
 * The data partitioning patterns on the variable is column-wise.
 * The local buffer has ghost cells surrounded along both dimensions.
 *
 * The compile and run commands are given below.
 *
 *    % mpicc -O2 -o flexible_varm flexible_varm.c -lpnetcdf
 *
 *    % mpiexec -l -n 4 ./flexible_varm /pvfs2/wkliao/testfile.nc
 *
 *    % ncmpidump /pvfs2/wkliao/testfile.nc
 *    netcdf testfile {
 *    // file format: CDF-5 (big variables)
 *    dimensions:
 *            Y = 6 ;
 *            X = 16 ;
 *    variables:
 *            double var(Y, X) ;
 *    data:
 *            
 *    var =
 *      0,  6, 12, 18, 0,  6, 12, 18, 0,  6, 12, 18, 0,  6, 12, 18,
 *      1,  7, 13, 19, 1,  7, 13, 19, 1,  7, 13, 19, 1,  7, 13, 19,
 *      2,  8, 14, 20, 2,  8, 14, 20, 2,  8, 14, 20, 2,  8, 14, 20,
 *      3,  9, 15, 21, 3,  9, 15, 21, 3,  9, 15, 21, 3,  9, 15, 21,
 *      4, 10, 16, 22, 4, 10, 16, 22, 4, 10, 16, 22, 4, 10, 16, 22,
 *      5, 11, 17, 23, 5, 11, 17, 23, 5, 11, 17, 23, 5, 11, 17, 23 ;
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

#define NY 6
#define NX 4
#define GHOST 2

#define ERR {if(err!=NC_NOERR){printf("Error at line=%d: %s\n", __LINE__, ncmpi_strerror(err)); nerrs++;}}

#define INIT_PUT_BUF \
    for (i=0; i<array_of_sizes[0]; i++) { \
        for (j=0; j<array_of_sizes[1]; j++) { \
            if (i < GHOST || GHOST+array_of_subsizes[0] <= i || \
                j < GHOST || GHOST+array_of_subsizes[1] <= j) \
                buf[i][j] = -1; \
            else \
                buf[i][j] = (i-GHOST)*array_of_subsizes[1]+(j-GHOST); \
        } \
    }

#define CHECK_PUT_BUF \
    for (i=0; i<array_of_sizes[0]; i++) { \
        for (j=0; j<array_of_sizes[1]; j++) { \
            if (i < GHOST || GHOST+array_of_subsizes[0] <= i || \
                j < GHOST || GHOST+array_of_subsizes[1] <= j) { \
                if (buf[i][j] != -1) { \
                    printf("Error: put buffer altered buffer[%d][%d]=%d\n", \
                           i,j,buf[i][j]); \
                    nerrs++; \
                } \
            } \
            else { \
                if (buf[i][j] != (i-GHOST)*array_of_subsizes[1]+(j-GHOST)) { \
                    printf("Error: put buffer altered buffer[%d][%d]=%d\n", \
                           i,j,buf[i][j]); \
                    nerrs++; \
                } \
            } \
        } \
    }

#define INIT_GET_BUF \
    for (i=0; i<array_of_sizes[0]; i++) \
        for (j=0; j<array_of_sizes[1]; j++) \
            buf[i][j] = -2;

#define CHECK_GET_BUF \
    for (i=0; i<array_of_sizes[0]; i++) { \
        for (j=0; j<array_of_sizes[1]; j++) { \
            if (i < GHOST || GHOST+array_of_subsizes[0] <= i || \
                j < GHOST || GHOST+array_of_subsizes[1] <= j) { \
                if (buf[i][j] != -2) { \
                    printf("Unexpected get buffer[%d][%d]=%d\n", \
                           i,j,buf[i][j]); \
                    nerrs++; \
                } \
            } \
            else { \
                if (buf[i][j] != (i-GHOST)*array_of_subsizes[1]+(j-GHOST)) { \
                    printf("Unexpected get buffer[%d][%d]=%d\n", \
                           i,j,buf[i][j]); \
                    nerrs++; \
                } \
            } \
        } \
    }

int main(int argc, char** argv)
{
    char filename[256];
    int i, j, rank, nprocs, err, nerrs=0, req, status;
    int ncid, cmode, varid, dimid[2];
    int array_of_sizes[2], array_of_subsizes[2], array_of_starts[2];
    int buf[NX+2*GHOST][NY+2*GHOST];
    MPI_Offset start[2], count[2], stride[2], imap[2];
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
        sprintf(cmd_str, "*** TESTING C   %s for flexible varm APIs ", argv[0]);
        printf("%-66s ------ ", cmd_str); fflush(stdout);
    }

    /* create a new file for writing ----------------------------------------*/
    cmode = NC_CLOBBER | NC_64BIT_DATA;
    err = ncmpi_create(MPI_COMM_WORLD, filename, cmode, MPI_INFO_NULL, &ncid);
    ERR

    /* define 2 dimensions */
    err = ncmpi_def_dim(ncid, "Y", NY,        &dimid[0]); ERR
    err = ncmpi_def_dim(ncid, "X", NX*nprocs, &dimid[1]); ERR

    /* define a variable of size NY * (NX * nprocs) */
    err = ncmpi_def_var(ncid, "var", NC_DOUBLE, 2, dimid, &varid); ERR
    err = ncmpi_enddef(ncid); ERR

     start[0] = 0;  start[1] = NX * rank;
     count[0] = NY; count[1] = NX;
    stride[0] = 1; stride[1] = 1;
      imap[0] = 1;   imap[1] = NY; /* would be {NX, 1} if not transposing */

    /* var is partitioned along X dimension in a matrix transported way */
    array_of_sizes[0]    = NX + 2*GHOST;
    array_of_sizes[1]    = NY + 2*GHOST;
    array_of_subsizes[0] = NX;
    array_of_subsizes[1] = NY;
    array_of_starts[0]   = GHOST;
    array_of_starts[1]   = GHOST;
    MPI_Type_create_subarray(2, array_of_sizes, array_of_subsizes,
                             array_of_starts, MPI_ORDER_C, MPI_INT, &subarray);
    MPI_Type_commit(&subarray);

    /* calling a blocking put_varm flexible API -----------------------------*/
    /* initiate put buffer contents */
    INIT_PUT_BUF
    err = ncmpi_put_varm_all(ncid, varid, start, count, stride, imap, buf,
                             1, subarray);
    ERR

    /* check the contents of put buffer */
    CHECK_PUT_BUF

    /* calling a nonblocking put_varm flexible API --------------------------*/
    /* initiate put buffer contents */
    INIT_PUT_BUF
    err = ncmpi_iput_varm(ncid, varid, start, count, stride, imap, buf,
                          1, subarray, &req);
    ERR
    err = ncmpi_wait_all(ncid, 1, &req, &status); ERR
    err = status; ERR

    /* check the contents of put buffer */
    CHECK_PUT_BUF

    /* read back using a blocking get_varm flexible API ---------------------*/
    /* initiate get buffer contents */
    INIT_GET_BUF

    /* calling a blocking flexible API */
    err = ncmpi_get_varm_all(ncid, varid, start, count, stride, imap, buf,
                             1, subarray);
    ERR

    /* check the contents of get buffer */
    CHECK_GET_BUF

    /* read back using a non-blocking flexible API --------------------------*/
    /* initiate get buffer contents */
    INIT_GET_BUF

    /* calling a blocking flexible API */
    err = ncmpi_iget_varm(ncid, varid, start, count, stride, imap, buf,
                          1, subarray, &req);
    ERR
    err = ncmpi_wait_all(ncid, 1, &req, &status); ERR
    err = status; ERR

    /* check the contents of get buffer */
    CHECK_GET_BUF

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


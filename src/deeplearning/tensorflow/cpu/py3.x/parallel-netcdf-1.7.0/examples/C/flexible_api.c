/*********************************************************************
 *
 *  Copyright (C) 2013, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *********************************************************************/
/* $Id: flexible_api.c 2150 2015-10-10 05:52:57Z wkliao $ */

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * This example shows how to use PnetCDF flexible APIs, ncmpi_put_vara_all()
 * and ncmpi_iput_vara() to write two 2D array variables (one is of 4-byte
 * integer byte and the other float type) in parallel. It first defines 2 netCDF
 * variables of sizes
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
 *    % mpicc -O2 -o flexible_api flexible_api.c -lpnetcdf
 *
 *    % mpiexec -l -n 4 ./flexible_api /pvfs2/wkliao/testfile.nc
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
#include <unistd.h> /* getopt() */
#include <assert.h>
#include <mpi.h>
#include <pnetcdf.h>

#define NZ 5
#define NY 5
#define NX 5

#define ERR {if(err!=NC_NOERR)printf("Error at line=%d: %s\n", __LINE__, ncmpi_strerror(err));}

static void
usage(char *argv0)
{
    char *help =
    "Usage: %s [-h] | [-q] [file_name]\n"
    "       [-h] Print help\n"
    "       [-q] Quiet mode (reports when fail)\n"
    "       [filename] output netCDF file name\n";
    fprintf(stderr, help, argv0);
}

int main(int argc, char** argv)
{
    extern int optind;
    char *filename="testfile.nc";
    int i, j, rank, nprocs, verbose=1, err, req, status, ghost_len=3;
    int ncid, cmode, varid0, varid1, dimid[3], *buf_zy;
    int array_of_sizes[2], array_of_subsizes[2], array_of_starts[2];
    double *buf_yx;
    MPI_Offset start[2], count[2];
    MPI_Datatype  subarray;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /* get command-line arguments */
    while ((i = getopt(argc, argv, "hq")) != EOF)
        switch(i) {
            case 'q': verbose = 0;
                      break;
            case 'h':
            default:  if (rank==0) usage(argv[0]);
                      MPI_Finalize();
                      return 0;
        }
    argc -= optind;
    argv += optind;
    if (argc == 1) filename = argv[0]; /* optional argument */

    if (verbose && rank == 0) printf("%s: example of using flexible APIs\n",__FILE__);

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
        if (buf_zy[i] != rank)
            printf("Error put buffer[%d] is altered\n",i);
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
                if (buf_zy[index] != -1)
                    printf("Unexpected get buffer[%d][%d]=%d\n",
                           i,j,buf_zy[index]);
            }
            else {
                if (buf_zy[index] != rank)
                    printf("Unexpected get buffer[%d][%d]=%d\n",
                           i,j,buf_zy[index]);
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
        if (buf_yx[i] != rank)
            printf("Error iput buffer[%d]=%f is altered\n",i,buf_yx[i]);
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
                if (buf_yx[index] != -1)
                    printf("Unexpected get buffer[%d][%d]=%f\n",
                           i,j,buf_yx[index]);
            }
            else {
                if (buf_yx[index] != rank)
                    printf("Unexpected get buffer[%d][%d]=%f\n",
                           i,j,buf_yx[index]);
            }
        }
    }
    free(buf_yx);
    MPI_Type_free(&subarray);

    err = ncmpi_close(ncid); ERR

    /* check if there is any PnetCDF internal malloc residue */
    MPI_Offset malloc_size, sum_size;
    err = ncmpi_inq_malloc_size(&malloc_size);
    if (err == NC_NOERR) {
        MPI_Reduce(&malloc_size, &sum_size, 1, MPI_OFFSET, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0 && sum_size > 0)
            printf("heap memory allocated by PnetCDF internally has %lld bytes yet to be freed\n",
                   sum_size);
    }

    MPI_Finalize();
    return 0;
}


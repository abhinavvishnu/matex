/*********************************************************************
 *
 *  Copyright (C) 2015, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *********************************************************************/
/* $Id: column_wise.c 2216 2015-12-08 05:47:19Z wkliao $ */

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * This example tests a number of nonblocking API calls, each writes a single
 * column of a 2D integer array. Each process writes NX columns and any two
 * consecutive columns are of nprocs columns distance apart from each other. In
 * this case, the fileview of each process interleaves with all other processes.
 * If simply concatenating fileviews of all the nonblocking calls will result
 * in a fileview that violates the MPI-IO requirement on the fileview of which
 * flattened file offsets must be monotonically non-decreasing. PnetCDF handles
 * this case by breaking down each nonblocking call into a lsit of offset-length
 * pairs, merging the pairs across multiple nonblocking calls, and sorting
 * them into an increasing order. The sorted pairs are used to construct a
 * fileview that meets the monotonically non-decreasing offset requirement,
 * and thus the nonblocking requests can be serviced by a single MPI-IO call. 
 * 
 * The compile and run commands are given below, together with an ncmpidump of
 * the output file.
 *
 *    % mpicc -O2 -o column_wise column_wise.c -lpnetcdf
 *    % mpiexec -l -n 4 ./column_wise /pvfs2/wkliao/testfile.nc
 *    0:  0: myOff=  0 myNX=  4
 *    1:  1: myOff=  4 myNX=  4
 *    2:  2: myOff=  8 myNX=  4
 *    3:  3: myOff= 12 myNX=  4
 *    0:  0: start=  0   0 count= 10   1
 *    1:  1: start=  0   1 count= 10   1
 *    2:  2: start=  0   2 count= 10   1
 *    3:  3: start=  0   3 count= 10   1
 *
 *    % ncmpidump /pvfs2/wkliao/testfile.nc
 *    netcdf testfile {
 *    // file format: CDF-5 (big variables)
 *    dimensions:
 *            Y = 10 ;
 *            X = 16 ;
 *    variables:
 *            int var(Y, X) ;
 *    data:
 *
 *     var =
 *      0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
 *      0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
 *      0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
 *      0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
 *      0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
 *      0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
 *      0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
 *      0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
 *      0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
 *      0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3 ;
 *    }
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <string.h> /* strcpy() */
#include <unistd.h> /* getopt() */
#include <mpi.h>
#include <pnetcdf.h>

#include <testutils.h>

#define NY 10
#define NX 4

#define ERR {if(err!=NC_NOERR){nerrs++; printf("Error at line=%d: %s\n", __LINE__, ncmpi_strerror(err));}}

int main(int argc, char** argv)
{
    extern int optind;
    char filename[256];
    int i, j, nerrs=0, rank, nprocs, err, myNX, G_NX, myOff, num_reqs;
    int ncid, cmode, varid, dimid[2], *reqs, *sts, **buf;
    MPI_Offset start[2], count[2];

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
        sprintf(cmd_str, "*** TESTING C   %s for iput/iget interleaved access ", argv[0]);
        printf("%-66s ------ ", cmd_str);
    }

    cmode = NC_CLOBBER | NC_64BIT_DATA;
    err = ncmpi_create(MPI_COMM_WORLD, filename, cmode, MPI_INFO_NULL, &ncid);
    ERR

    /* the global array is NY * (NX * nprocs) */
    G_NX  = NX * nprocs;
    myOff = NX * rank;
    myNX  = NX;

    err = ncmpi_def_dim(ncid, "Y", NY, &dimid[0]);
    ERR
    err = ncmpi_def_dim(ncid, "X", G_NX, &dimid[1]);
    ERR
    err = ncmpi_def_var(ncid, "var", NC_INT, 2, dimid, &varid);
    ERR
    err = ncmpi_enddef(ncid);
    ERR

    /* First, fill the entire array with zeros, using a blocking I/O.
       Every process writes a subarray of size NY * myNX */
    buf    = (int**) malloc(myNX * sizeof(int*));
    buf[0] = (int*)  calloc(NY * myNX, sizeof(int));
    start[0] = 0;   start[1] = myOff;
    count[0] = NY;  count[1] = myNX;
    err = ncmpi_put_vara_int_all(ncid, varid, start, count, buf[0]);
    free(buf[0]);

    /* initialize the buffer with rank ID. Also make the case interesting,
       by allocatsing buffersd separately */
    for (i=0; i<myNX; i++) {
        buf[i] = (int*) malloc(NY * sizeof(int));
        for (j=0; j<NY; j++) buf[i][j] = rank;
    }

    reqs = (int*) malloc(myNX * sizeof(int));
    sts  = (int*) malloc(myNX * sizeof(int));

    /* each proc writes myNX single columns of the 2D array */
    start[0]  = 0;   start[1] = rank;
    count[0]  = NY;  count[1] = 1;

    num_reqs = 0;
    for (i=0; i<myNX; i++) {
        err = ncmpi_iput_vara_int(ncid, varid, start, count, buf[i],
                                  &reqs[num_reqs++]);
        ERR
        start[1] += nprocs;
    }
    err = ncmpi_wait_all(ncid, num_reqs, reqs, sts);
    ERR

    /* check status of all requests */
    for (i=0; i<num_reqs; i++) {
        if (reqs[i] != NC_REQ_NULL) { /* add in PnetCDF v1.7.0 */
            printf("Error: request ID %d fails to be set to NC_REQ_NULL\n",i);
            nerrs++;
        }
        if (sts[i] != NC_NOERR) {
            printf("Error: nonblocking write fails on request %d (%s)\n",
                   i, ncmpi_strerror(sts[i]));
            nerrs++;
        }
    }

    /* read back using the same access pattern */
    for (i=0; i<myNX; i++)
        for (j=0; j<NY; j++) buf[i][j] = -1;

    /* each proc reads myNX single columns of the 2D array */
    start[0]  = 0;   start[1] = rank;
    count[0]  = NY;  count[1] = 1;

    num_reqs = 0;
    for (i=0; i<myNX; i++) {
        err = ncmpi_iget_vara_int(ncid, varid, start, count, buf[i],
                                  &reqs[num_reqs++]);
        ERR
        start[1] += nprocs;
    }
    err = ncmpi_wait_all(ncid, num_reqs, reqs, sts);
    ERR

    /* check status of all requests */
    for (i=0; i<num_reqs; i++)
        if (sts[i] != NC_NOERR) {
            printf("Error: nonblocking write fails on request %d (%s)\n",
                   i, ncmpi_strerror(sts[i]));
            nerrs++;
        }

    for (i=0; i<myNX; i++) {
        for (j=0; j<NY; j++)
            if (buf[i][j] != rank) {
                printf("Error: expect buf[%d][%d]=%d but got %d\n",i,j,rank,buf[i][j]);
                nerrs++;
            }
    }

    err = ncmpi_close(ncid);
    ERR

    free(sts);
    free(reqs);
    for (i=0; i<myNX; i++) free(buf[i]);
    free(buf);

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


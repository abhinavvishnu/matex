/*********************************************************************
 *
 *  Copyright (C) 2015, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *********************************************************************/
/* $Id$ */

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * This example is mostly the same as column_wise.c but shows how to use
 * NC_REQ_ALL to commit all pending nonblocking I/O without checking the status
 * of individual requests. In this case, the first encountered error will be
 * returned in ncmpi_wait_all().
 * 
 * The compile and run commands are given below, together with an ncmpidump of
 * the output file.
 *
 *    % mpicc -O2 -o req_all req_all.c -lpnetcdf
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
 *    // file format: CDF-1
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

#define NY 10
#define NX 4

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
    char *filename = "testfile.nc";
    int i, j, verbose=1, rank, nprocs, err, myNX, G_NX, myOff;
    int ncid, cmode, varid, dimid[2], **buf;
    MPI_Offset start[2], count[2];
    MPI_Info info;

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

    /* set an MPI-IO hint to disable file offset alignment for fix-sized
     * variables */
    MPI_Info_create(&info);
    MPI_Info_set(info, "nc_var_align_size", "1");

    cmode = NC_CLOBBER;
    err = ncmpi_create(MPI_COMM_WORLD, filename, cmode, info, &ncid);
    ERR

    MPI_Info_free(&info);

    /* the global array is NY * (NX * nprocs) */
    G_NX  = NX * nprocs;
    myOff = NX * rank;
    myNX  = NX;
    if (verbose) printf("%2d: myOff=%3d myNX=%3d\n",rank,myOff,myNX);

    err = ncmpi_def_dim(ncid, "Y", NY, &dimid[0]);
    ERR
    err = ncmpi_def_dim(ncid, "X", G_NX, &dimid[1]);
    ERR
    err = ncmpi_def_var(ncid, "var", NC_INT, 2, dimid, &varid);
    ERR
    err = ncmpi_enddef(ncid);
    ERR

    /* initialize the buffer with rank ID */
    buf = (int**) malloc(myNX * sizeof(int*));
    for (i=0; i<myNX; i++) {
        buf[i] = (int*) malloc(NY * sizeof(int));
        for (j=0; j<NY; j++) buf[i][j] = rank;
    }

    /* each proc writes myNX single columns of the 2D array */
    start[0]  = 0;   start[1] = rank;
    count[0]  = NY;  count[1] = 1;
    if (verbose)
        printf("%2d: start=%3lld %3lld count=%3lld %3lld\n",
               rank, start[0],start[1], count[0],count[1]);

    for (i=0; i<myNX; i++) {
        err = ncmpi_iput_vara_int(ncid, varid, start, count, buf[i], NULL);
        ERR
        start[1] += nprocs;
    }
    err = ncmpi_wait_all(ncid, NC_REQ_ALL, NULL, NULL);
    ERR

    /* read back using the same access pattern */
    for (i=0; i<myNX; i++)
        for (j=0; j<NY; j++) buf[i][j] = -1;

    /* each proc reads myNX single columns of the 2D array */
    start[0]  = 0;   start[1] = rank;
    count[0]  = NY;  count[1] = 1;

    for (i=0; i<myNX; i++) {
        err = ncmpi_iget_vara_int(ncid, varid, start, count, buf[i], NULL);
        ERR
        start[1] += nprocs;
    }
    err = ncmpi_wait_all(ncid, NC_REQ_ALL, NULL, NULL);
    ERR

    /* check contents of read data of all requests */
    for (i=0; i<myNX; i++) {
        for (j=0; j<NY; j++)
            if (buf[i][j] != rank)
                printf("Error: expect buf[%d][%d]=%d but got %d\n",i,j,rank,buf[i][j]);
    }

    err = ncmpi_close(ncid);
    ERR

    for (i=0; i<myNX; i++) free(buf[i]);
    free(buf);

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


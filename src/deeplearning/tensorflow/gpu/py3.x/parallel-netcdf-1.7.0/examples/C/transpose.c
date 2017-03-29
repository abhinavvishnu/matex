/*********************************************************************
 *
 *  Copyright (C) 2014, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *********************************************************************/
/* $Id: transpose.c 2325 2016-02-28 07:49:13Z wkliao $ */

/*
 *    This example shows how to use varm API to write six 3D integer array
 *    variables into a file. Each variable in the file is a dimensional
 *    transposed array from the one stored in memory. In memory, a 3D array is
 *    partitioned among all processes in a block-block-block fashion and in
 *    ZYX (i.e. C) order. The dimension structures of the transposed six
 *    arrays are
 *       int ZYX_var(Z, Y, X) ;     ZYX -> ZYX
 *       int ZXY_var(Z, X, Y) ;     ZYX -> ZXY
 *       int YZX_var(Y, Z, X) ;     ZYX -> YZX
 *       int YXZ_var(Y, X, Z) ;     ZYX -> YXZ
 *       int XZY_var(X, Z, Y) ;     ZYX -> XZY
 *       int XYZ_var(X, Y, Z) ;     ZYX -> XYZ
 *
 *    To compile:
 *        mpicc -O2 transpose.c -o transpose -lpnetcdf
 *    To run:
 *        mpiexec -n num_processes ./transpose [filename] [len]
 *    where len decides the size of local array, which is len x len+1 x len+2.
 *    So, each variable is of size len*(len+1)*(len+2) * nprocs * sizeof(int)
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h> /* strlen() */
#include <unistd.h> /* getopt() */
#include <mpi.h>
#include <pnetcdf.h>

#define NDIMS 3

#define HANDLE_ERROR {                                \
    if (err != NC_NOERR)                              \
        printf("Error at line %d (%s)\n", __LINE__,   \
               ncmpi_strerror(err));                  \
}

static void
usage(char *argv0)
{
    char *help =
    "Usage: %s [-h] | [-q] [file_name] [len]\n"
    "       [-h] Print this help\n"
    "       [-q] quiet mode\n"
    "       [filename] output netCDF file name\n"
    "       [len] size of each dimension of the local array\n";
    fprintf(stderr, help, argv0);
}


/*----< main() >------------------------------------------------------------*/
int main(int argc, char **argv)
{
    extern int optind;
    char *filename="testfile.nc", str[512];
    int i, j, k, rank, nprocs, len, ncid, bufsize, verbose=1, err;
    int *buf, psizes[NDIMS], dimids[NDIMS], dimidsT[NDIMS];
    int XYZ_id, XZY_id, YZX_id, YXZ_id, ZYX_id, ZXY_id;
    MPI_Offset gsizes[NDIMS], starts[NDIMS], counts[NDIMS], imap[NDIMS];
    MPI_Offset startsT[NDIMS], countsT[NDIMS];
    MPI_Info info;

    MPI_Init(&argc,&argv);
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
    if (argc >= 1) filename = argv[0];  /* optional argument */
    len = 2;
    if (argc >= 2) len = atoi(argv[1]); /* optional argument */

    for (i=0; i<NDIMS; i++)
        psizes[i] = 0;

    /* calculate number of processes along each dimension */
    MPI_Dims_create(nprocs, NDIMS, psizes);
    if (verbose && rank == 0) {
        sprintf(str, "psizes= ");
        for (i=0; i<NDIMS; i++) sprintf(str+strlen(str), "%d ",psizes[i]);
        printf("%s\n",str);
    }

    /* for each MPI rank, find its local rank IDs along each dimension in
     * starts[] */
    int lower_dims=1;
    for (i=NDIMS-1; i>=0; i--) {
        starts[i] = rank / lower_dims % psizes[i];
        lower_dims *= psizes[i];
    }
    if (verbose) {
        sprintf(str, "proc %d: dim rank= ", rank);
        for (i=0; i<NDIMS; i++) sprintf(str+strlen(str), "%lld ",starts[i]);
        printf("%s\n",str);
    }

    bufsize = 1;
    for (i=0; i<NDIMS; i++) {
        gsizes[i]  = (len + i) * psizes[i]; /* global array size */
        starts[i] *= (len + i);             /* start indices */
        counts[i]  = (len + i);             /* array elements */
        bufsize   *= (len + i);
    }

    /* allocate buffer and initialize with contiguous numbers */
    buf = (int *) malloc(bufsize * sizeof(int));
    for (k=0; k<counts[0]; k++)
    for (j=0; j<counts[1]; j++)
    for (i=0; i<counts[2]; i++)
        buf[k*counts[1]*counts[2] +
                      j*counts[2] + i] = (starts[0]+k)*gsizes[1]*gsizes[2]
                                       + (starts[1]+j)*gsizes[2]
                                       + (starts[2]+i);

    /* set an MPI-IO hint to disable file offset alignment for fix-sized
     * variables */
    MPI_Info_create(&info);
    MPI_Info_set(info, "nc_var_align_size", "1");

    /* create the file */
    err = ncmpi_create(MPI_COMM_WORLD, filename, NC_CLOBBER|NC_64BIT_DATA,
                       info, &ncid);
    if (err != NC_NOERR) {
        printf("Error: ncmpi_create() file %s (%s)\n",filename,ncmpi_strerror(err));
        MPI_Abort(MPI_COMM_WORLD, -1);
        exit(1);
    }

    MPI_Info_free(&info);

    /* define dimensions */
    for (i=0; i<NDIMS; i++) {
        sprintf(str, "%c", 'Z'-i);
        err = ncmpi_def_dim(ncid, str, gsizes[i], &dimids[i]);
        HANDLE_ERROR
    }

    /* define variable with no transposed file layout: ZYX */
    err = ncmpi_def_var(ncid, "ZYX_var", NC_INT, NDIMS, dimids, &ZYX_id);
    HANDLE_ERROR

    /* define variable with transposed file layout: ZYX -> ZXY */
    dimidsT[0] = dimids[0]; dimidsT[1] = dimids[2]; dimidsT[2] = dimids[1];
    err = ncmpi_def_var(ncid, "ZXY_var", NC_INT, NDIMS, dimidsT, &ZXY_id);
    HANDLE_ERROR

    /* define variable with transposed file layout: ZYX -> YZX */
    dimidsT[0] = dimids[1]; dimidsT[1] = dimids[0]; dimidsT[2] = dimids[2];
    err = ncmpi_def_var(ncid, "YZX_var", NC_INT, NDIMS, dimidsT, &YZX_id);
    HANDLE_ERROR

    /* define variable with transposed file layout: ZYX -> YXZ */
    dimidsT[0] = dimids[1]; dimidsT[1] = dimids[2]; dimidsT[2] = dimids[0];
    err = ncmpi_def_var(ncid, "YXZ_var", NC_INT, NDIMS, dimidsT, &YXZ_id);
    HANDLE_ERROR

    /* define variable with transposed file layout: ZYX -> XZY */
    dimidsT[0] = dimids[2]; dimidsT[1] = dimids[0]; dimidsT[2] = dimids[1];
    err = ncmpi_def_var(ncid, "XZY_var", NC_INT, NDIMS, dimidsT, &XZY_id);
    HANDLE_ERROR

    /* define variable with transposed file layout: ZYX -> XYZ */
    dimidsT[0] = dimids[2]; dimidsT[1] = dimids[1]; dimidsT[2] = dimids[0];
    err = ncmpi_def_var(ncid, "XYZ_var", NC_INT, NDIMS, dimidsT, &XYZ_id);
    HANDLE_ERROR

    /* exit the define mode */
    err = ncmpi_enddef(ncid);
    HANDLE_ERROR

    /* write the whole variable in file: ZYX */
    err = ncmpi_put_vara_int_all(ncid, ZYX_id, starts, counts, buf);
    HANDLE_ERROR

    /* ZYX -> ZXY: */
    imap[1] = 1; imap[2] = counts[2]; imap[0] = counts[1]*counts[2];
    startsT[0] = starts[0]; startsT[1] = starts[2]; startsT[2] = starts[1];
    countsT[0] = counts[0]; countsT[1] = counts[2]; countsT[2] = counts[1];
    /* write the transposed variable */
    err = ncmpi_put_varm_int_all(ncid, ZXY_id, startsT, countsT, NULL, imap, buf);
    HANDLE_ERROR

    /* ZYX -> YZX: */
    imap[2] = 1; imap[0] = counts[2]; imap[1] = counts[1]*counts[2];
    startsT[0] = starts[1]; startsT[1] = starts[0]; startsT[2] = starts[2];
    countsT[0] = counts[1]; countsT[1] = counts[0]; countsT[2] = counts[2];
    /* write the transposed variable */
    err = ncmpi_put_varm_int_all(ncid, YZX_id, startsT, countsT, NULL, imap, buf);
    HANDLE_ERROR

    /* ZYX -> YXZ: */
    imap[1] = 1; imap[0] = counts[2]; imap[2] = counts[1]*counts[2];
    startsT[0] = starts[1]; startsT[1] = starts[2]; startsT[2] = starts[0];
    countsT[0] = counts[1]; countsT[1] = counts[2]; countsT[2] = counts[0];
    /* write the transposed variable */
    err = ncmpi_put_varm_int_all(ncid, YXZ_id, startsT, countsT, NULL, imap, buf);
    HANDLE_ERROR

    /* ZYX -> XZY: */
    imap[0] = 1; imap[2] = counts[2]; imap[1] = counts[1]*counts[2];
    startsT[0] = starts[2]; startsT[1] = starts[0]; startsT[2] = starts[1];
    countsT[0] = counts[2]; countsT[1] = counts[0]; countsT[2] = counts[1];
    /* write the transposed variable */
    err = ncmpi_put_varm_int_all(ncid, XZY_id, startsT, countsT, NULL, imap, buf);
    HANDLE_ERROR

    /* ZYX -> XYZ: */
    imap[0] = 1; imap[1] = counts[2]; imap[2] = counts[1]*counts[2];
    startsT[0] = starts[2]; startsT[1] = starts[1]; startsT[2] = starts[0];
    countsT[0] = counts[2]; countsT[1] = counts[1]; countsT[2] = counts[0];
    /* write the transposed variable */
    err = ncmpi_put_varm_int_all(ncid, XYZ_id, startsT, countsT, NULL, imap, buf);
    HANDLE_ERROR

    /* close the file */
    err = ncmpi_close(ncid);
    HANDLE_ERROR

    free(buf);

    MPI_Finalize();
    return 0;
}


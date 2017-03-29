/*********************************************************************
 *
 *  Copyright (C) 2013, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *********************************************************************/
/* $Id: mput.c 2325 2016-02-28 07:49:13Z wkliao $ */

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * This example shows how to use a single call of ncmpi_mput_vara_all() to
 * write a sequence of requests with arbitrary array indices and lengths.
 * Using ncmpi_mput_vara_all() can achieve the same effect of HDF5 writing
 * a sequence of selected file locations through the following 2 APIs.
 *
 *   H5Sselect_elements(fid, H5S_SELECT_SET, NUMP, (const hssize_t **)coord);
 *   H5Dwrite(dataset, H5T_NATIVE_INT, mid, fid, H5P_DEFAULT, val); 
 *
 * Note that in ncmpi_mput_vara_all(), users can write more than one element
 * starting at each selected location.
 *
 * The compile and run commands are given below, together with an ncmpidump of
 * the output file.
 *
 *    % mpicc -O2 -o mput mput.c -lpnetcdf
 *    % mpiexec -l -n 4 ./mput /pvfs2/wkliao/testfile.nc
 *    % ncmpidump /pvfs2/wkliao/testfile.nc
 *    netcdf testfile {
 *    // file format: CDF-5 (big variables)
 *    dimensions:
 *             Y = 4 ;
 *             X = 10 ;
 *    variables:
 *             int var(Y, X) ;
 *    data:
 *
 *     var =
 *       3, 3, 3, 1, 1, 0, 0, 2, 1, 1,
 *       0, 2, 2, 2, 3, 1, 1, 2, 2, 2,
 *       1, 1, 2, 3, 3, 3, 0, 0, 1, 1,
 *       0, 0, 0, 2, 1, 1, 1, 3, 3, 3 ;
 *    }
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <string.h> /* strcpy() */
#include <unistd.h> /* getopt() */
#include <assert.h>
#include <mpi.h>
#include <pnetcdf.h>

#define NY 4
#define NX 10
#define NDIMS 2


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
    int i, rank, nprocs, verbose=1, err;
    int ncid, cmode, varid, dimid[2], num_reqs, *buffer, **bufs, *nvarids;
    MPI_Offset w_len, **starts, **counts, *bufcounts;
    MPI_Datatype *datatypes;
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

    if (nprocs != 4 && rank == 0 && verbose)
        printf("Warning: this program is intended to run on 4 processes\n");

    /* set an MPI-IO hint to disable file offset alignment for fix-sized
     * variables */
    MPI_Info_create(&info);
    MPI_Info_set(info, "nc_var_align_size", "1");

    /* create a new file for writing ----------------------------------------*/
    cmode = NC_CLOBBER | NC_64BIT_DATA;
    err = ncmpi_create(MPI_COMM_WORLD, filename, cmode, info, &ncid);
    ERR

    MPI_Info_free(&info);

    /* create a global array of size NY * NX */
    err = ncmpi_def_dim(ncid, "Y", NY, &dimid[0]);
    ERR
    err = ncmpi_def_dim(ncid, "X", NX, &dimid[1]);
    ERR
    err = ncmpi_def_var(ncid, "var", NC_INT, NDIMS, dimid, &varid);
    ERR
    err = ncmpi_enddef(ncid);
    ERR

    /* pick arbitrary numbers of requests for 4 processes */
    num_reqs = 0;
    if (rank == 0)      num_reqs = 4;
    else if (rank == 1) num_reqs = 6;
    else if (rank == 2) num_reqs = 5;
    else if (rank == 3) num_reqs = 4;

    starts    = (MPI_Offset**) malloc(num_reqs *        sizeof(MPI_Offset*));
    counts    = (MPI_Offset**) malloc(num_reqs *        sizeof(MPI_Offset*));
    starts[0] = (MPI_Offset*)  calloc(num_reqs * NDIMS, sizeof(MPI_Offset));
    counts[0] = (MPI_Offset*)  calloc(num_reqs * NDIMS, sizeof(MPI_Offset));
    for (i=1; i<num_reqs; i++) {
        starts[i] = starts[i-1] + NDIMS;
        counts[i] = counts[i-1] + NDIMS;
    }

    /* assign arbitrary starts and counts */
    const int y=0, x=1;
    if (rank == 0) {
        starts[0][y] = 0; starts[0][x] = 5; counts[0][y] = 1; counts[0][x] = 2;
        starts[1][y] = 1; starts[1][x] = 0; counts[1][y] = 1; counts[1][x] = 1;
        starts[2][y] = 2; starts[2][x] = 6; counts[2][y] = 1; counts[2][x] = 2;
        starts[3][y] = 3; starts[3][x] = 0; counts[3][y] = 1; counts[3][x] = 3;
        /* rank 0 is writing the followings: ("-" means skip)
                  -  -  -  -  -  0  0  -  -  - 
                  0  -  -  -  -  -  -  -  -  - 
                  -  -  -  -  -  -  0  0  -  - 
                  0  0  0  -  -  -  -  -  -  - 
         */
    } else if (rank ==1) {
        starts[0][y] = 0; starts[0][x] = 3; counts[0][y] = 1; counts[0][x] = 2;
        starts[1][y] = 0; starts[1][x] = 8; counts[1][y] = 1; counts[1][x] = 2;
        starts[2][y] = 1; starts[2][x] = 5; counts[2][y] = 1; counts[2][x] = 2;
        starts[3][y] = 2; starts[3][x] = 0; counts[3][y] = 1; counts[3][x] = 2;
        starts[4][y] = 2; starts[4][x] = 8; counts[4][y] = 1; counts[4][x] = 2;
        starts[5][y] = 3; starts[5][x] = 4; counts[5][y] = 1; counts[5][x] = 3;
        /* rank 1 is writing the followings: ("-" means skip)
                  -  -  -  1  1  -  -  -  1  1 
                  -  -  -  -  -  1  1  -  -  - 
                  1  1  -  -  -  -  -  -  1  1 
                  -  -  -  -  1  1  1  -  -  - 
         */
    } else if (rank ==2) {
        starts[0][y] = 0; starts[0][x] = 7; counts[0][y] = 1; counts[0][x] = 1;
        starts[1][y] = 1; starts[1][x] = 1; counts[1][y] = 1; counts[1][x] = 3;
        starts[2][y] = 1; starts[2][x] = 7; counts[2][y] = 1; counts[2][x] = 3;
        starts[3][y] = 2; starts[3][x] = 2; counts[3][y] = 1; counts[3][x] = 1;
        starts[4][y] = 3; starts[4][x] = 3; counts[4][y] = 1; counts[4][x] = 1;
        /* rank 2 is writing the followings: ("-" means skip)
                  -  -  -  -  -  -  -  2  -  - 
                  -  2  2  2  -  -  -  2  2  2 
                  -  -  2  -  -  -  -  -  -  - 
                  -  -  -  2  -  -  -  -  -  - 
         */
    } else if (rank ==3) {
        starts[0][y] = 0; starts[0][x] = 0; counts[0][y] = 1; counts[0][x] = 3;
        starts[1][y] = 1; starts[1][x] = 4; counts[1][y] = 1; counts[1][x] = 1;
        starts[2][y] = 2; starts[2][x] = 3; counts[2][y] = 1; counts[2][x] = 3;
        starts[3][y] = 3; starts[3][x] = 7; counts[3][y] = 1; counts[3][x] = 3;
        /* rank 3 is writing the followings: ("-" means skip)
                  3  3  3  -  -  -  -  -  -  - 
                  -  -  -  -  3  -  -  -  -  - 
                  -  -  -  3  3  3  -  -  -  - 
                  -  -  -  -  -  -  -  3  3  3 
         */
    }

    nvarids   = (int*)          malloc(num_reqs * sizeof(int));
    bufcounts = (MPI_Offset*)   malloc(num_reqs * sizeof(MPI_Offset));
    datatypes = (MPI_Datatype*) malloc(num_reqs * sizeof(MPI_Datatype));
    w_len = 0;
    for (i=0; i<num_reqs; i++) {
        nvarids[i]    = varid;
        bufcounts[i]  = counts[i][x];
        datatypes[i]  = MPI_INT;
        w_len        += bufcounts[i];
    }

    /* allocate I/O buffer and initialize its contents */
    buffer = (int*)  malloc(w_len * sizeof(int));
    for (i=0; i<w_len; i++) buffer[i] = rank;

    /* set the buffer pointers to different offsets to the I/O buffer */
    bufs    = (int**) malloc(num_reqs * sizeof(int*));
    bufs[0] = buffer;
    for (i=1; i<num_reqs; i++) bufs[i] = bufs[i-1] + bufcounts[i-1];

    err = ncmpi_mput_vara_all(ncid, num_reqs, nvarids, starts, counts,
                              (void**)bufs, bufcounts, datatypes);
    ERR

    err = ncmpi_close(ncid);
    ERR

    free(buffer);
    free(bufs);
    free(nvarids);
    free(bufcounts);
    free(datatypes);
    free(starts[0]);
    free(counts[0]);
    free(starts);
    free(counts);

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


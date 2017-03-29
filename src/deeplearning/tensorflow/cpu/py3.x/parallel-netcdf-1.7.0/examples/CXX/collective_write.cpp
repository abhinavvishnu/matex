/*********************************************************************
 *
 *  Copyright (C) 2014, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *********************************************************************/
/* $Id: collective_write.cpp 2245 2015-12-20 18:39:52Z wkliao $ */

/*
 *    This example mimics the coll_perf.c from ROMIO.
 *    It creates a netcdf file in CD-5 format and writes a number of
 *    3D integer non-record variables. The measured write bandwidth is reported
 *    at the end. Usage:
 *    To compile:
 *        mpiccxx -O2 collective_write.cpp -o collective_write -lpnetcdf
 *    To run:
 *        mpiexec -n num_processes ./collective_write [filename] [len]
 *    where len decides the size of each local array, which is len x len x len.
 *    So, each non-record variable is of size len*len*len * nprocs * sizeof(int)
 *    All variables are partitioned among all processes in a 3D
 *    block-block-block fashion. Below is an example standard output from
 *    command:
 *        mpiexec -n 32 ./collective_write /pvfs2/wkliao/testfile.nc 100
 *
 *    MPI hint: cb_nodes        = 2
 *    MPI hint: cb_buffer_size  = 16777216
 *    MPI hint: striping_factor = 32
 *    MPI hint: striping_unit   = 1048576
 *    Local array size 100 x 100 x 100 integers, size = 3.81 MB
 *    Global array size 400 x 400 x 200 integers, write size = 0.30 GB
 *     procs    Global array size  exec(sec)  write(MB/s)
 *     -------  ------------------  ---------  -----------
 *        32     400 x  400 x  200     6.67       45.72
 */

#include <stdio.h>
#include <stdlib.h>

#include <iostream>
using namespace std;

#include <string.h> /* strcpy() */
#include <unistd.h> /* getopt() */
#include <pnetcdf>

using namespace PnetCDF;
using namespace PnetCDF::exceptions;

#define NDIMS    3
#define NUM_VARS 10

static void
usage(char *argv0)
{
    cerr <<
    "Usage: %s [-h] | [-q] [file_name] [len]\n"
    "       [-h] Print help\n"
    "       [-q] Quiet mode (reports when fail)\n"
    "       [filename] output netCDF file name\n"
    "       [len] size of each dimension of the local array\n"
    << argv0;
}

/*----< print_info() >------------------------------------------------------*/
static
void print_info(MPI_Info *info_used)
{
    int     flag;
    char    info_cb_nodes[64], info_cb_buffer_size[64];
    char    info_striping_factor[64], info_striping_unit[64];

    strcpy(info_cb_nodes,        "undefined");
    strcpy(info_cb_buffer_size,  "undefined");
    strcpy(info_striping_factor, "undefined");
    strcpy(info_striping_unit,   "undefined");

    MPI_Info_get(*info_used, (char*)"cb_nodes", 64, info_cb_nodes, &flag);
    MPI_Info_get(*info_used, (char*)"cb_buffer_size", 64, &info_cb_buffer_size[0], &flag);
    MPI_Info_get(*info_used, (char*)"striping_factor", 64, &info_striping_factor[0], &flag);
    MPI_Info_get(*info_used, (char*)"striping_unit", 64, info_striping_unit, &flag);

    printf("MPI hint: cb_nodes        = %s\n", info_cb_nodes);
    printf("MPI hint: cb_buffer_size  = %s\n", info_cb_buffer_size);
    printf("MPI hint: striping_factor = %s\n", info_striping_factor);
    printf("MPI hint: striping_unit   = %s\n", info_striping_unit);
}

/*----< main() >------------------------------------------------------------*/
int main(int argc, char **argv)
{
    extern int optind;
    char filename[128], str[512];
    int i, j, rank, nprocs, len, bufsize, verbose=1;
    int *buf[NUM_VARS], psizes[NDIMS];
    double write_timing, max_write_timing, write_bw;
    vector<MPI_Offset> starts(NDIMS), counts(NDIMS);
    MPI_Offset gsizes[NDIMS];
    MPI_Offset write_size, sum_write_size;
    MPI_Info info_used;

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
    if (argc >= 1) strcpy(filename, argv[0]); /* optional argument */
    else           strcpy(filename, "testfile.nc");
    len = 10; 
    if (argc >= 2) len = atoi(argv[1]); /* optional argument */

    for (i=0; i<NDIMS; i++)
        psizes[i] = 0;

    MPI_Dims_create(nprocs, NDIMS, psizes);
    starts[0] = rank % psizes[0];
    starts[1] = (rank / psizes[1]) % psizes[1];
    starts[2] = (rank / (psizes[0] * psizes[1])) % psizes[2];

    bufsize = 1;
    for (i=0; i<NDIMS; i++) {
        gsizes[i] = len * psizes[i];
        starts[i] *= len;
        counts[i]  = len;
        bufsize   *= len;
    }

    /* allocate buffer and initialize with random numbers */
    srand(rank);
    for (i=0; i<NUM_VARS; i++) {
        buf[i] = (int *) malloc(bufsize * sizeof(int));
        for (j=0; j<bufsize; j++) buf[i][j] = rand();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    write_timing = MPI_Wtime();

    try {
        /* create the file */
        NcmpiFile nc(MPI_COMM_WORLD, filename, NcmpiFile::replace,
                     NcmpiFile::classic5);

        /* define dimensions */
        vector<NcmpiDim> dimids(NDIMS);
        for (i=0; i<NDIMS; i++) {
            sprintf(str, "%c", 'x'+i);
            dimids[i] = nc.addDim(str, gsizes[i]);
        }

        /* define variables */
        vector<NcmpiVar> vars(NUM_VARS);
        for (i=0; i<NUM_VARS; i++) {
            sprintf(str, "var%d", i);
            vars[i] = nc.addVar(str, ncmpiInt, dimids);
        }

        /* get all the hints used */
        nc.Inq_file_info(&info_used);

        /* write one variable at a time */
        for (i=0; i<NUM_VARS; i++)
            vars[i].putVar_all(starts, counts, &buf[i][0]);
    }
    catch(NcmpiException& e) {
       cout << e.what() << " error code=" << e.errorCode() << " Error!\n";
       return 1;
    }

    write_timing = MPI_Wtime() - write_timing;

    write_size = bufsize * NUM_VARS * sizeof(int);
    for (i=0; i<NUM_VARS; i++) free(buf[i]);

    MPI_Reduce(&write_size, &sum_write_size, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&write_timing, &max_write_timing, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0 && verbose) {
        double write_amount = sum_write_size;
        float subarray_size = (float)bufsize*sizeof(int)/1048576.0;
        print_info(&info_used);
        printf("Local array size %d x %d x %d integers, size = %.2f MB\n",len,len,len,subarray_size);
        write_amount /= 1048576.0;
        printf("Global array size %lld x %lld x %lld integers, write size = %.2f GB\n",
               gsizes[0], gsizes[1], gsizes[2], write_amount/1024.0);

        write_bw = write_amount/max_write_timing;
        printf(" procs    Global array size  exec(sec)  write(MB/s)\n");
        printf("-------  ------------------  ---------  -----------\n");
        printf(" %4d    %4lld x %4lld x %4lld %8.2f  %10.2f\n", nprocs,
               gsizes[0], gsizes[1], gsizes[2], max_write_timing, write_bw);
    }
    MPI_Info_free(&info_used);

    /* check if there is any PnetCDF internal malloc residue */
    MPI_Offset malloc_size, sum_size;
    int err = ncmpi_inq_malloc_size(&malloc_size);
    if (err == NC_NOERR) {
        MPI_Reduce(&malloc_size, &sum_size, 1, MPI_OFFSET, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0 && sum_size > 0)
            printf("heap memory allocated by PnetCDF internally has %lld bytes yet to be freed\n",
                   sum_size);
    }

    MPI_Finalize();
    return 0;
}


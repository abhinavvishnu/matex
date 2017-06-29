/*********************************************************************
 *
 *  Copyright (C) 2015, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *********************************************************************/
/* $Id$ */

/* This example is the same as nonblocking_write.c expect all nonblocking
 * write requests (calls to iput and bput) are posted in define mode.
 * It creates a netcdf file in CD-5 format and writes a number of
 * 3D integer non-record variables. The measured write bandwidth is reported
 * at the end. Usage: (for example)
 * To compile:
 *   mpicc -O2 nonblocking_write_in_def.c -o nonblocking_write_in_def -lpnetcdf
 * To run:
 *   mpiexec -n num_processes ./nonblocking_write_in_def [filename] [len]
 * where len decides the size of each local array, which is len x len x len.
 * So, each non-record variable is of size len*len*len * nprocs * sizeof(int)
 * All variables are partitioned among all processes in a 3D
 * block-block-block fashion. Below is an example standard output from
 * command:
 *   mpiexec -n 32 ./nonblocking_write_in_def /pvfs2/wkliao/testfile.nc 100
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
#include <string.h> /* strcpy() */
#include <unistd.h> /* getopt() */
#include <mpi.h>
#include <pnetcdf.h>

#define NDIMS    3
#define NUM_VARS 10

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
    "       [-h] Print help\n"
    "       [-q] Quiet mode (reports when fail)\n"
    "       [filename] output netCDF file name\n"
    "       [len] size of each dimension of the local array\n";
    fprintf(stderr, help, argv0);
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

    MPI_Info_get(*info_used, "cb_nodes", 64, info_cb_nodes, &flag);
    MPI_Info_get(*info_used, "cb_buffer_size", 64, info_cb_buffer_size, &flag);
    MPI_Info_get(*info_used, "striping_factor", 64, info_striping_factor, &flag);
    MPI_Info_get(*info_used, "striping_unit", 64, info_striping_unit, &flag);

    printf("MPI hint: cb_nodes        = %s\n", info_cb_nodes);
    printf("MPI hint: cb_buffer_size  = %s\n", info_cb_buffer_size);
    printf("MPI hint: striping_factor = %s\n", info_striping_factor);
    printf("MPI hint: striping_unit   = %s\n", info_striping_unit);
}

/*----< main() >------------------------------------------------------------*/
int main(int argc, char **argv)
{
    extern int optind;
    int i, j, verbose=1, err;
    int nprocs, len, *buf[NUM_VARS], bufsize, rank;
    int gsizes[NDIMS], psizes[NDIMS];
    double write_timing, max_write_timing, write_bw;
    char *filename="testfile.nc", str[512];
    int ncid, dimids[NDIMS], varids[NUM_VARS];
    MPI_Offset start[NDIMS], count[NDIMS], write_size, sum_write_size;
    MPI_Offset bbufsize;
    MPI_Info info, info_used;
    MPI_Comm comm=MPI_COMM_WORLD;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

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
    len = 10; 
    if (argc >= 2) len = atoi(argv[1]); /* optional argument */

    for (i=0; i<NDIMS; i++) psizes[i] = 0;

    MPI_Dims_create(nprocs, NDIMS, psizes);
    start[0] = rank % psizes[0];
    start[1] = (rank / psizes[1]) % psizes[1];
    start[2] = (rank / (psizes[0] * psizes[1])) % psizes[2];

    bufsize = 1;
    for (i=0; i<NDIMS; i++) {
        gsizes[i] = len * psizes[i];
        start[i] *= len;
        count[i]  = len;
        bufsize  *= len;
    }

    /* allocate buffer and initialize with random numbers */
    srand(rank);
    for (i=0; i<NUM_VARS; i++) {
        buf[i] = (int *) malloc(bufsize * sizeof(int));
        for (j=0; j<bufsize; j++) buf[i][j] = rand();
    }

    MPI_Barrier(comm);
    write_timing = MPI_Wtime();

    /* set an MPI-IO hint to disable file offset alignment for fix-sized
     * variables */
    MPI_Info_create(&info);
    MPI_Info_set(info, "nc_var_align_size", "1");

    /* create the file */
    err = ncmpi_create(comm, filename, NC_CLOBBER|NC_64BIT_DATA, info, &ncid);
    if (err != NC_NOERR) {
        printf("Error: ncmpi_create() file %s (%s)\n",filename,ncmpi_strerror(err));
        MPI_Abort(comm, -1);
        exit(1);
    }

    MPI_Info_free(&info);

    /* define dimensions */
    for (i=0; i<NDIMS; i++) {
        sprintf(str, "%c", 'x'+i);
        err = ncmpi_def_dim(ncid, str, gsizes[i], &dimids[i]);
        HANDLE_ERROR
    }

    /* define variables */
    for (i=0; i<NUM_VARS; i++) {
        sprintf(str, "var%d", i);
        err = ncmpi_def_var(ncid, str, NC_INT, NDIMS, dimids, &varids[i]);
        HANDLE_ERROR
    }

    /* post a nonblocking request for writing one variable at a time using
     * iput and ignore the request ID, as we will flush all pending request
     * in one ncmpi_wait_all() call */
    for (i=0; i<NUM_VARS; i++) {
        err = ncmpi_iput_vara_int(ncid, varids[i], start, count, buf[i], NULL);
        HANDLE_ERROR
    }

    /* write one variable at a time using bput */

    /* bbufsize must be max of data type converted before and after */
    bbufsize = bufsize * NUM_VARS * sizeof(int);
    err = ncmpi_buffer_attach(ncid, bbufsize);
    HANDLE_ERROR

    /* post a nonblocking request for writing one variable at a time using
     * bput and ignore the request ID, as we will flush all pending request
     * in one ncmpi_wait_all() call */
    for (i=0; i<NUM_VARS; i++) {
        err = ncmpi_bput_vara_int(ncid, varids[i], start, count, buf[i], NULL);
        HANDLE_ERROR
        /* can safely change contents or free up the buf[i] here */
    }

    /* exit the define mode */
    err = ncmpi_enddef(ncid);
    HANDLE_ERROR

    /* wait for all the nonblocking I/O to complete */
    err = ncmpi_wait_all(ncid, NC_REQ_ALL, NULL, NULL);
    HANDLE_ERROR

    /* detach the temporary buffer */
    err = ncmpi_buffer_detach(ncid);
    HANDLE_ERROR

    /* get all the hints used */
    err = ncmpi_inq_file_info(ncid, &info_used);
    HANDLE_ERROR

    MPI_Offset put_size;
    ncmpi_inq_put_size(ncid, &put_size);
#ifdef MPI_OFFSET
    MPI_Allreduce(MPI_IN_PLACE, &put_size, 1, MPI_OFFSET, MPI_SUM, comm);
#else
    MPI_Allreduce(MPI_IN_PLACE, &put_size, 1, MPI_LONG_LONG, MPI_SUM, comm);
#endif

    /* close the file */
    err = ncmpi_close(ncid);
    HANDLE_ERROR

    write_timing = MPI_Wtime() - write_timing;

    write_size = bufsize * NUM_VARS * sizeof(int);
    for (i=0; i<NUM_VARS; i++) free(buf[i]);

#ifdef MPI_OFFSET
    MPI_Reduce(&write_size, &sum_write_size, 1, MPI_OFFSET, MPI_SUM, 0, comm);
#else
    MPI_Reduce(&write_size, &sum_write_size, 1, MPI_LONG_LONG, MPI_SUM, 0, comm);
#endif
    MPI_Reduce(&write_timing, &max_write_timing, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (rank == 0 && verbose) {
        printf("\n");
        printf("Total amount writes to variables only   (exclude header) = %lld bytes\n", sum_write_size);
        printf("Total amount writes reported by pnetcdf (include header) = %lld bytes\n", put_size);
        printf("\n");
        float subarray_size = (float)bufsize*sizeof(int)/1048576.0;
        print_info(&info_used);
        printf("Local array size %d x %d x %d integers, size = %.2f MB\n",len,len,len,subarray_size);
        sum_write_size /= 1048576.0;
        printf("Global array size %d x %d x %d integers, write size = %.2f GB\n",
               gsizes[0], gsizes[1], gsizes[2], sum_write_size/1024.0);

        write_bw = sum_write_size/max_write_timing;
        printf(" procs    Global array size  exec(sec)  write(MB/s)\n");
        printf("-------  ------------------  ---------  -----------\n");
        printf(" %4d    %4d x %4d x %4d %8.2f  %10.2f\n", nprocs,
               gsizes[0], gsizes[1], gsizes[2], max_write_timing, write_bw);
    }
    MPI_Info_free(&info_used);

    /* check if there is any PnetCDF internal malloc residue */
    MPI_Offset malloc_size, sum_size;
    err = ncmpi_inq_malloc_size(&malloc_size);
    if (err == NC_NOERR) {
        MPI_Reduce(&malloc_size, &sum_size, 1, MPI_OFFSET, MPI_SUM, 0, comm);
        if (rank == 0 && sum_size > 0)
            printf("heap memory allocated by PnetCDF internally has %lld bytes yet to be freed\n",
                   sum_size);
    }

    MPI_Finalize();
    return 0;
}


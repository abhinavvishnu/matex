/*********************************************************************
 *
 *  Copyright (C) 2012, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *********************************************************************/
/* $Id: pnetcdf-write-nb.c 2095 2015-09-07 21:15:24Z wkliao $ */

/* simple demonstration of pnetcdf 
 * text attribute on dataset
 * write out rank into 1-d array collectively.
 * This example demonstrates the non-blocking write interface */

/* This program creates a file, say named output.nc, with the following
   contents, shown by running ncmpidump command .

    % mpiexec -n 4 pnetcdf-write-nb /orangefs/wkliao/output.nc

    % ncmpidump /orangefs/wkliao/output.nc 
    netcdf output {
    // file format: CDF-2 (large file)
    dimensions:
            d1 = 4 ;
    variables:
            int v1(d1) ;
            int v2(d1) ;

    // global attributes:
                :string = "Hello World\n",
        "" ;
    data:

         v1 = 0, 1, 2, 3 ;

         v2 = 0, 1, 2, 3 ;
    }
*/

#include <stdlib.h>
#include <mpi.h>
#include <pnetcdf.h>
#include <stdio.h>

static void handle_error(int status, int lineno)
{
    fprintf(stderr, "Error at line %d: %s\n", lineno, ncmpi_strerror(status));
    MPI_Abort(MPI_COMM_WORLD, 1);
}

int main(int argc, char **argv) {

    int ret, ncfile, nprocs, rank, dimid, varid1, varid2, ndims=1;
    MPI_Offset start, count=1;
    char buf[13] = "Hello World\n";
    int data1, data2;
    int requests[2], statuses[2];

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc != 2) {
        if (rank == 0) printf("Usage: %s filename\n", argv[0]);
        MPI_Finalize();
        exit(-1);
    }

    ret = ncmpi_create(MPI_COMM_WORLD, argv[1],
                       NC_CLOBBER|NC_64BIT_OFFSET, MPI_INFO_NULL, &ncfile);
    if (ret != NC_NOERR) handle_error(ret, __LINE__);

    ret = ncmpi_def_dim(ncfile, "d1", nprocs, &dimid);
    if (ret != NC_NOERR) handle_error(ret, __LINE__);

    ret = ncmpi_def_var(ncfile, "v1", NC_INT, ndims, &dimid, &varid1);
    if (ret != NC_NOERR) handle_error(ret, __LINE__);

    ret = ncmpi_def_var(ncfile, "v2", NC_INT, ndims, &dimid, &varid2);
    if (ret != NC_NOERR) handle_error(ret, __LINE__);

    ret = ncmpi_put_att_text(ncfile, NC_GLOBAL, "string", 13, buf);
    if (ret != NC_NOERR) handle_error(ret, __LINE__);

    /* all processors defined the dimensions, attributes, and variables,
     * but here in ncmpi_enddef is the one place where metadata I/O
     * happens.  Behind the scenes, rank 0 takes the information and writes
     * the netcdf header.  All processes communicate to ensure they have
     * the same (cached) view of the dataset */

    ret = ncmpi_enddef(ncfile);
    if (ret != NC_NOERR) handle_error(ret, __LINE__);

    start=rank, count=1, data1=rank, data2=rank;

    /* in this simple example every process writes its rank to two 1d variables */

    /* we used a basic MPI_INT type to this flexible mode call, but could
     * have used any derived MPI datatype that describes application data
     * structures */

    /* furthermore, we use the non-blocking interface to essentially
     * schedule the two write operations.  No i/o actually happens here,
     * which is why these routines do not need to be collective.   */
    ret = ncmpi_iput_vara(ncfile, varid1, &start, &count, &data1, count,
            MPI_INT, &requests[0]);
    if (ret != NC_NOERR) handle_error(ret, __LINE__);

    ret = ncmpi_iput_vara(ncfile, varid2, &start, &count, &data2, count, 
            MPI_INT, &requests[1]);
    if (ret != NC_NOERR) handle_error(ret, __LINE__);

    ret = ncmpi_wait_all(ncfile, 2, requests, statuses);
    if (ret != NC_NOERR) handle_error(ret, __LINE__);

    /* check status of each nonblocking call */
    if (statuses[0] != NC_NOERR) handle_error(statuses[0], __LINE__);
    if (statuses[1] != NC_NOERR) handle_error(statuses[1], __LINE__);

    ret = ncmpi_close(ncfile);
    if (ret != NC_NOERR) handle_error(ret, __LINE__);

    MPI_Finalize();

    return 0;
}

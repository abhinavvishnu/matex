/*********************************************************************
 *
 *  Copyright (C) 2012, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *********************************************************************/
/* $Id: pnetcdf-write-nfiles.c 2245 2015-12-20 18:39:52Z wkliao $ */

/* simple demonstration of pnetcdf 
 * text attribute on dataset
 * Each process writes out rank into 1-d array to a separate file.
 * This is not a good way to do parallel I/O */

/*
To run on 4 processes for example,
% mpiexec -l -n 4 pnetcdf-write-nfiles output.nc

There will be 4+1 files created.
    output.nc
    output.nc.0-4.nc
    output.nc.1-4.nc
    output.nc.2-4.nc 
    output.nc.3-4.nc  

The contents of files are shown at the bottom of this files.
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

#define DSET_NAME_LEN 1024

int main(int argc, char **argv) {

    int ret, ncfile, nprocs, rank, dimid, varid1, varid2, ndims=1;
    char buf[13] = "Hello World\n";
    int data;
    char filename[DSET_NAME_LEN];

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc != 2) {
        if (rank == 0) printf("Usage: %s filename\n", argv[0]);
        MPI_Finalize();
        exit(-1);
    }

    /* Many applications find "one file per process" easy, but there are
     * several deficiencies with that approach:
     * - here we need to construct a unique file name for each processor */
    ret = snprintf(filename, DSET_NAME_LEN, "%s.%d-%d.nc", argv[1], rank, nprocs);
    if (ret >= DSET_NAME_LEN) {
        fprintf(stderr, "name too long \n");
        exit(-1);
    }

    /* note that the communicator is still needed but now it is
     * MPI_COMM_SELF: since each processor opens its own file we cannot use
     * MPI_COMM_WORLD */
    ret = ncmpi_create(MPI_COMM_SELF, filename,
                       NC_CLOBBER|NC_64BIT_OFFSET, MPI_INFO_NULL, &ncfile);
    if (ret != NC_NOERR) handle_error(ret, __LINE__);

    /* each processor writes its data to a file, so instead of an "nprocs"
     * sized array, we just have an array big enough to hold one
     * processor's data */
    ret = ncmpi_def_dim(ncfile, "d1", 1, &dimid);
    if (ret != NC_NOERR) handle_error(ret, __LINE__);

    ret = ncmpi_def_var(ncfile, "v1", NC_INT, ndims, &dimid, &varid1);
    if (ret != NC_NOERR) handle_error(ret, __LINE__);

    ret = ncmpi_def_var(ncfile, "v2", NC_INT, ndims, &dimid, &varid2);
    if (ret != NC_NOERR) handle_error(ret, __LINE__);

    ret = ncmpi_put_att_text(ncfile, NC_GLOBAL, "string", 13, buf);
    if (ret != NC_NOERR) handle_error(ret, __LINE__);

    /* ncmpi_enddef writes the header out as in other examples, but because
     * each processor opened the file independently, there can be no "write
     * and broadcast" optimization.  Instead, every processor does header
     * i/o.  */        
    ret = ncmpi_enddef(ncfile); if (ret != NC_NOERR) handle_error(ret, __LINE__);

    /* the one advantage to this approach: data decomposition is easy the
     * application does not need to worry about the shape and location of
     * the data (the 'start' and 'count' parameters in the 'vara' family of
     * functions) and can instead just write the entire (small) variable */

    data=rank;

    /* in this simple example every process writes its rank to two 1d
     * variables */
    /* When each processor writes to its own file, a whole host of
     * optimizations cannot take place.   */
        
    ret = ncmpi_put_var_int_all(ncfile, varid1, &data);
    if (ret != NC_NOERR) handle_error(ret, __LINE__);

    ret = ncmpi_put_var_int_all(ncfile, varid2, &data);
    if (ret != NC_NOERR) handle_error(ret, __LINE__);

    ret = ncmpi_close(ncfile);
    if (ret != NC_NOERR) handle_error(ret, __LINE__);

    MPI_Finalize();

    return 0;
}

/* The contents of files created by this program are:
% ncmpidump  output.nc
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

% ncmpidump output.nc.0-4.nc
netcdf output.nc.0-4 {
// file format: CDF-2 (large file)
dimensions:
        d1 = 1 ;
variables:
        int v1(d1) ;
        int v2(d1) ;

// global attributes:
                :string = "Hello World\n",
    "" ;
data:

 v1 = 0 ;

 v2 = 0 ;
}

% ncmpidump output.nc.1-4.nc
netcdf output.nc.1-4 {
// file format: CDF-2 (large file)
dimensions:
        d1 = 1 ;
variables:
        int v1(d1) ;
        int v2(d1) ;

// global attributes:
                :string = "Hello World\n",
    "" ;
data:

 v1 = 1 ;

 v2 = 1 ;
}

% ncmpidump output.nc.2-4.nc
netcdf output.nc.2-4 {
// file format: CDF-2 (large file)
dimensions:
        d1 = 1 ;
variables:
        int v1(d1) ;
        int v2(d1) ;

// global attributes:
                :string = "Hello World\n",
    "" ;
data:

 v1 = 2 ;

 v2 = 2 ;
}

% ncmpidump output.nc.3-4.nc
netcdf output.nc.3-4 {
// file format: CDF-2 (large file)
dimensions:
        d1 = 1 ;
variables:
        int v1(d1) ;
        int v2(d1) ;

// global attributes:
                :string = "Hello World\n",
    "" ;
data:

 v1 = 3 ;

 v2 = 3 ;
}
*/

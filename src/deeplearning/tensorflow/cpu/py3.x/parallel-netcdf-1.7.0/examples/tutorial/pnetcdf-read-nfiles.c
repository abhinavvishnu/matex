/*********************************************************************
 *
 *  Copyright (C) 2012, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *********************************************************************/
/* $Id: pnetcdf-read-nfiles.c 2245 2015-12-20 18:39:52Z wkliao $ */

/* simple demonstration of pnetcdf 
 * text attribute on dataset
 * rank 0 reads into 1-d array, broadcasts to all.  This is a dumb way
 * to do parallel I/O but folks do this sometimes...
 *
 * This program reads the files generated from its counterpart program 
 * pnetcdf-write-nfiles.c. See comments in pnetcdf-write-nfiles.c for
 * the contents of the netCDF files.
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

    int i, j, rank, nprocs, ret;
    int ncfile, ndims, nvars, ngatts, unlimited;
    int var_ndims, var_natts;;
    MPI_Offset *dim_sizes, var_size;
    MPI_Offset *count;
    char filename[DSET_NAME_LEN];
    char varname[NC_MAX_NAME+1];
    int dimids[NC_MAX_VAR_DIMS];
    nc_type type;
    int *data=NULL;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc != 2) {
        if (rank == 0) printf("Usage: %s filename\n", argv[0]);
        MPI_Finalize();
        exit(-1);
    }

    /* the most significant challenge with the "one file per processor"
     * approach is the challenge in reading back on a different number of
     * processors.  For example, 4k processors output data during a simulation.
     * Later on, 100 processors do analysis or visualization of the data.
     * Stitching together the many smaller files into a form usable by other
     * programs poses a challenge */

    ret = snprintf(filename, DSET_NAME_LEN, "%s.%d-%d.nc", argv[1], rank, nprocs);
    if (ret >= DSET_NAME_LEN) {
        fprintf(stderr, "name too long \n");
        exit(-1);
    }
    ret = ncmpi_open(MPI_COMM_SELF, filename, NC_NOWRITE, MPI_INFO_NULL, &ncfile);
    if (ret != NC_NOERR) handle_error(ret, __LINE__);

    /* reader knows nothing about dataset, but we can interrogate with query
     * routines: ncmpi_inq tells us how many of each kind of "thing"
     * (dimension, variable, attribute) we will find in the file  */

    /* In the "one file per processor case" all processors open a file and
     * interrogate it */

    ret = ncmpi_inq(ncfile, &ndims, &nvars, &ngatts, &unlimited);
    if (ret != NC_NOERR) handle_error(ret, __LINE__);

    /* we do not really need the name of the dimension or the variable for
     * reading in this example.  we could, in a different example, take the
     * name of a variable on the command line and read just that one */

    dim_sizes = (MPI_Offset*) calloc(ndims, sizeof(MPI_Offset));
    /* netcdf dimension identifiers are allocated sequentially starting
     * at zero; same for variable identifiers */
    for (i=0; i<ndims; i++)  {
        ret = ncmpi_inq_dimlen(ncfile, i, &(dim_sizes[i]) );
        if (ret != NC_NOERR) handle_error(ret, __LINE__);
    }

    for (i=0; i<nvars; i++) { 
        /* much less coordination in this case compared to rank 0 doing all
         * the i/o: everyone already has the necessary information */
        ret = ncmpi_inq_var(ncfile, i, varname, &type, &var_ndims, dimids,
                            &var_natts);
        if (ret != NC_NOERR) handle_error(ret, __LINE__);

        count = (MPI_Offset*) calloc(var_ndims, sizeof(MPI_Offset));

        /* as long as the number of readers is identical to the number of
         * writers, we can simply read entire variables back */

        count[0] = dim_sizes[dimids[0]];

        var_size = count[0];
        for (j=1; j<var_ndims; j++) {
            count[j] = dim_sizes[dimids[j]];
            var_size *= count[j];
        }

        switch(type) {
            case NC_INT:
                data = (int*) calloc(var_size, sizeof(int));
                ret = ncmpi_get_var_int_all(ncfile, i, data);
                if (ret != NC_NOERR) handle_error(ret, __LINE__);
                break;
            default:
                /* we can do this for all the known netcdf types but this
                 * example is already getting too long  */
                fprintf(stderr, "unsupported NetCDF type \n");
        }

        free(count);
        if (data != NULL) free(data);
    }

    ret = ncmpi_close(ncfile);
    if (ret != NC_NOERR) handle_error(ret, __LINE__);

    MPI_Finalize();
    return 0;
}

/*
 *vim: ts=8 sts=4 sw=4 noexpandtab */

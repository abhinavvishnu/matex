/*********************************************************************
 *
 *  Copyright (C) 2012, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *********************************************************************/
/* $Id: pnetcdf-read-from-master.c 1723 2014-07-06 05:41:41Z wkliao $ */

/* simple demonstration of pnetcdf 
 * text attribute on dataset
 * rank 0 reads into 1-d array, broadcasts to all.  This is a dumb way
 * to do parallel I/O, but folks do this sometimes... */

/* This program reads a file created by pnetcdf-write-from-master.c, say file
   named output.nc with the following contents, shown by running ncmpidump command .

    % mpiexec -n 4 pnetcdf-read-from-master /orangefs/wkliao/output.nc

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

    int i, j=0, rank, nprocs, ret;
    int ncfile, ndims, nvars, ngatts, unlimited, var_ndims, var_natts;;
    int dimids[NC_MAX_VAR_DIMS];
    char varname[NC_MAX_NAME+1];
    MPI_Offset *dim_sizes=NULL, var_size;
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

    if (rank == 0) {
        ret = ncmpi_open(MPI_COMM_SELF, argv[1],
                         NC_NOWRITE, MPI_INFO_NULL, &ncfile);
        if (ret != NC_NOERR) handle_error(ret, __LINE__);

        /* reader knows nothing about dataset, but we can interrogate with
         * query routines: ncmpi_inq tells us how many of each kind of
         * "thing" (dimension, variable, attribute) we will find in the
         * file  */

        ret = ncmpi_inq(ncfile, &ndims, &nvars, &ngatts, &unlimited);
        if (ret != NC_NOERR) handle_error(ret, __LINE__);

        /* we do not really need the name of the dimension or the variable
         * for reading in this example.  we could, in a different example,
         * take the name of a variable on the command line and read just
         * that one */

        dim_sizes = (MPI_Offset*) calloc(ndims, sizeof(MPI_Offset));
        /* netcdf dimension identifiers are allocated sequentially starting
         * at zero; same for variable identifiers */
        for(i=0; i<ndims; i++)  {
            ret = ncmpi_inq_dimlen(ncfile, i, &(dim_sizes[i]) );
            if (ret != NC_NOERR) handle_error(ret, __LINE__);
        }
    }

    /* need to inform other MPI processors how many variables we will send */
    MPI_Bcast(&nvars, 1, MPI_INT, 0, MPI_COMM_WORLD);

    for(i=0; i<nvars; i++) { 
        /* rank 0 will find out the size of each variable, read it, and
         * broadcast it to the rest of the processors */
        if (rank == 0) {
            ret = ncmpi_inq_var(ncfile, i, varname, &type, &var_ndims, dimids,
                    &var_natts);
            if (ret != NC_NOERR) handle_error(ret, __LINE__);

            for (j=0, var_size=1; j<var_ndims; j++)  {
                var_size *= dim_sizes[dimids[j]];
            }
        }
        /* oddity: there's no predefined MPI_Offset type */
        MPI_Bcast(&var_size, 1, MPI_OFFSET, 0, MPI_COMM_WORLD);

        data = (int*) calloc(var_size, sizeof(int));

        if (rank == 0) {
            switch(type) {
                case NC_INT:
                    /* now we have the variable identifiers and we know how big
                     * they are */

                    /* this approach is not scalable: i/o happens from a single
                     * processor.  This approach can be ok if the amount of
                     * data is quite small, but almost always the underlying
                     * MPI-IO library can do a better job */

                    ret = ncmpi_get_var_int_all(ncfile, j, data);
                    if (ret != NC_NOERR) handle_error(ret, __LINE__);
                    break;
                default:
                    /* we can do this for all the known netcdf types but this
                     * example is already getting too long  */
                    fprintf(stderr, "unsupported NetCDF type \n");
            }
        }

        /*and finally all processors have the data */
        MPI_Bcast(data, var_size, MPI_INT, 0, MPI_COMM_WORLD);

        /* Here, every process can do computation on the local buffer, data,
           or copy the contents to somewhere else */

        free(data);
    }

    if (rank == 0) {
        ret = ncmpi_close(ncfile);
        if (ret != NC_NOERR) handle_error(ret, __LINE__);
    }

    MPI_Finalize();
    return 0;
}

/*
 *vim: ts=8 sts=4 sw=4 noexpandtab */

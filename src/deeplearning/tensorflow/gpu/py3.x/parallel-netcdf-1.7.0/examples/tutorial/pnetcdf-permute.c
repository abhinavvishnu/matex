/*********************************************************************
 *
 *  Copyright (C) 2012, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *********************************************************************/
/* $Id: pnetcdf-permute.c 2245 2015-12-20 18:39:52Z wkliao $ */

/* simple demonstration of pnetcdf 
 * knowing nothing about the file, read in the variables. 
 *
 * This example demonstrates the flexible interface, using the MPI derived
 * datatype to transpose the matrix.
 *
 * Note this program demonstrates transposition for one process only
 */

#include <stdlib.h>
#include <mpi.h>
#include <pnetcdf.h>
#include <stdio.h>

#include <assert.h>

static void handle_error(int status, int lineno)
{
    fprintf(stderr, "Error at line %d: %s\n", lineno, ncmpi_strerror(status));
    MPI_Abort(MPI_COMM_WORLD, 1);
}

int main(int argc, char **argv) {

#define NDIMS 3
    int i, j, k, rank, nprocs, ret;
    int ncfile, ndims=NDIMS;
    MPI_Offset dim_sizes[NDIMS];
    MPI_Offset start[NDIMS], count[NDIMS], nitems;
    int dimids[NDIMS], transposed_dims[NDIMS];
    int varid1, transposed_varid, flexible_varid;
    double *data, *transposed_data;
    MPI_Datatype transposed_type, one_d, two_d;

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

    dim_sizes[0] = 4;
    dim_sizes[1] = 5;
    dim_sizes[2] = 6;

    ret = ncmpi_def_dim(ncfile, "x", dim_sizes[0], &(dimids[0]));
    if (ret != NC_NOERR) handle_error(ret, __LINE__);

    ret = ncmpi_def_dim(ncfile, "y", dim_sizes[1], &(dimids[1]));
    if (ret != NC_NOERR) handle_error(ret, __LINE__);

    ret = ncmpi_def_dim(ncfile, "z", dim_sizes[2], &(dimids[2]));
    if (ret != NC_NOERR) handle_error(ret, __LINE__);

    ret = ncmpi_def_var(ncfile, "v1", NC_INT, ndims, dimids, &varid1);
    if (ret != NC_NOERR) handle_error(ret, __LINE__);

    /* moab wants to permute {i,j,k} to {j,k,i} */
    transposed_dims[0] = dimids[1];
    transposed_dims[1] = dimids[2];
    transposed_dims[2] = dimids[0];
    ret = ncmpi_def_var(ncfile, "transposed-v1", NC_INT, ndims, transposed_dims,
	    &transposed_varid);
    if (ret != NC_NOERR) handle_error(ret, __LINE__);

    /* want this to end up looking like transposed-v1 */
    ret = ncmpi_def_var(ncfile, "flexible-v1", NC_INT, ndims, transposed_dims,
	    &flexible_varid);
    if (ret != NC_NOERR) handle_error(ret, __LINE__);

    ret = ncmpi_enddef(ncfile);
    if (ret != NC_NOERR) handle_error(ret, __LINE__);

    nitems = dim_sizes[0]*dim_sizes[1]*dim_sizes[2];
    data = (double*) calloc(nitems, sizeof(double));
    transposed_data = (double*) calloc(nitems, sizeof(double));

    for (i=0; i<dim_sizes[0]; i++) {
	for (j=0; j<dim_sizes[1]; j++) {
	    for (k=0; k<dim_sizes[2]; k++) {
		/* data in x,y,z order */
		data[i*dim_sizes[1]*dim_sizes[2] + j*dim_sizes[2] + k] =
		    100*i*dim_sizes[1]*dim_sizes[2] + 10*j*dim_sizes[2] + k;
		/* permute the array data[X][Y][Z] to transpose[Y][Z][X] */
		assert((j*dim_sizes[2]*dim_sizes[0] + k*dim_sizes[0] + i) < nitems);
		transposed_data[j*dim_sizes[2]*dim_sizes[0] + k*dim_sizes[0] + i] = 
		    100*i*dim_sizes[1]*dim_sizes[2] + 10*j*dim_sizes[2] + k;
	    }
	}
    }

    /* initial array written in i,j,k order  */

    start[0] = start[1] = start[2] = 0;
    count[0] = dim_sizes[0];
    count[1] = dim_sizes[1];
    count[2] = dim_sizes[2];
    if (rank > 0) nitems = count[0] = count[1] = count[2] = 0;
    ret = ncmpi_put_vara_all(ncfile, varid1, start, count, data, nitems, MPI_DOUBLE);
    if (ret != NC_NOERR) handle_error(ret, __LINE__);

    count[0] = dim_sizes[1];
    count[1] = dim_sizes[2];
    count[2] = dim_sizes[0];
    if (rank > 0) nitems = count[0] = count[1] = count[2] = 0;
    ret = ncmpi_put_vara_all(ncfile, transposed_varid, start, count, 
	    transposed_data, nitems, MPI_DOUBLE);
    if (ret != NC_NOERR) handle_error(ret, __LINE__);

    /* permute ijk (4x5x6) into jki (5x6x4)*/
    /* new innermost dimension is I items, strided across the old JK face*/
    MPI_Type_vector(dim_sizes[0], 1, dim_sizes[1]*dim_sizes[2], MPI_DOUBLE, &one_d);
    /* new middle dimension is K items, strided over the K row, which isn't
     * actually a stride in this case.  We use hvector here because we 
     * operate directly in terms of array items */
    MPI_Type_create_hvector(dim_sizes[2], 1, sizeof(double), one_d, &two_d);
    /* new outermost dimension is J items, strided over the old J row */
    MPI_Type_create_hvector(dim_sizes[1], 1, dim_sizes[2]*sizeof(double), two_d, &transposed_type);

    MPI_Type_commit(&transposed_type);
    MPI_Type_free(&one_d);
    MPI_Type_free(&two_d);

    nitems = 1;
    if (rank > 0) nitems = 0;
    ret = ncmpi_put_vara_all(ncfile, flexible_varid, start, count, 
	    data, nitems, transposed_type);

    MPI_Type_free(&transposed_type);

    ret = ncmpi_close(ncfile);
    if (ret != NC_NOERR) handle_error(ret, __LINE__);

    free(data);
    free(transposed_data);

    MPI_Finalize();
    return 0;
}

/*
 *vim: ts=8 sts=4 sw=4 noexpandtab */

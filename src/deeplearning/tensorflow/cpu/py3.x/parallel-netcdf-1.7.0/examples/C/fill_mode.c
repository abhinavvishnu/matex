/*********************************************************************
 *
 *  Copyright (C) 2015, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *********************************************************************/
/* $Id: fill_mode.c 2245 2015-12-20 18:39:52Z wkliao $ */

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * This example shows how to use 
 * 1. ncmpi_set_fill() to enable fill mode
 * 2. ncmpi_def_var_fill() to define the variable's fill value
 * 3. ncmpi_inq_var_fill() to inquire the variable's fill mode information
 * 4. ncmpi_put_vara_int_all() to write two 2D 4-byte integer array in parallel.
 * It first defines a netCDF record variable of size NC_UNLIMITED * global_ny
 * where
 *    global_nx == (NX * number of MPI processes).
 * It then defines a netCDF fixed-size variable of size global_nx * global_ny
 * where
 *    global_ny == NY and
 *    global_nx == (NX * number of MPI processes).
 * The data partitioning pattern for both variables are a column-wise
 * partitioning across all processes. Each process writes a subarray of size
 * ny * nx.
 *
 *    To compile:
 *        mpicc -O2 fill_mode.c -o fill_mode -lpnetcdf
 *
 * Example commands for MPI run and outputs from running ncmpidump on the
 * NC file produced by this example program:
 *
 *    % mpiexec -n 4 ./fill_mode -q /pvfs2/wkliao/testfile.nc
 *
 *    % ncmpidump /pvfs2/wkliao/testfile.nc
 *    netcdf testfile {
 *    // file format: CDF-1
 *    dimensions:
 *            REC_DIM = UNLIMITED ; // (2 currently)
 *            X = 16 ;
 *            Y = 3 ;
 *    variables:
 *            int rec_var(REC_DIM, X) ;
 *                rec_var:_FillValue = -1 ;
 *            int fix_var(Y, X) ;
 *    data:
 *
 *    rec_var =
 *           0, 0, 0, _, 1, 1, 1, _, 2, 2, 2, _, 3, 3, 3, _,
 *           0, 0, 0, _, 1, 1, 1, _, 2, 2, 2, _, 3, 3, 3, _ ;
 *
 *    fix_var =
 *           0, 0, 0, _, 1, 1, 1, _, 2, 2, 2, _, 3, 3, 3, _,
 *           0, 0, 0, _, 1, 1, 1, _, 2, 2, 2, _, 3, 3, 3, _,
 *           0, 0, 0, _, 1, 1, 1, _, 2, 2, 2, _, 3, 3, 3, _ ;
 *    }
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <string.h> /* strcpy() */
#include <unistd.h> /* getopt() */
#include <assert.h>
#include <mpi.h>
#include <pnetcdf.h>

#define NY 3
#define NX 4


#define ERR {if(err!=NC_NOERR)printf("Error at %s line=%d: %s\n",__FILE__,__LINE__, ncmpi_strerror(err));}

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
    char filename[256];
    int i, j, rank, nprocs, verbose=1, err, rec_varid, fix_varid;
    int ncid, cmode, dimid[2], buf[NY][NX], no_fill, fill_value, old_mode;
    MPI_Offset  global_ny, global_nx;
    MPI_Offset start[2], count[2];

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
    if (argc == 1) strcpy(filename, argv[0]); /* optional argument */
    else strcpy(filename, "testfile.nc");

    MPI_Bcast(filename, 256, MPI_CHAR, 0, MPI_COMM_WORLD);

    /* create a new file for writing ----------------------------------------*/
    cmode = NC_CLOBBER;
    err = ncmpi_create(MPI_COMM_WORLD, filename, cmode, MPI_INFO_NULL, &ncid);
    ERR

    /* the global array is NY * (NX * nprocs) */
    global_ny = NY;
    global_nx = NX * nprocs;

    /* define dimensions x and y */
    err = ncmpi_def_dim(ncid, "REC_DIM", NC_UNLIMITED, &dimid[0]); ERR
    err = ncmpi_def_dim(ncid, "X", global_nx, &dimid[1]); ERR

    /* define a 2D variable of integer type */
    err = ncmpi_def_var(ncid, "rec_var", NC_INT, 2, dimid, &rec_varid); ERR

    err = ncmpi_def_dim(ncid, "Y", global_ny, &dimid[0]); ERR
    err = ncmpi_def_var(ncid, "fix_var", NC_INT, 2, dimid, &fix_varid); ERR

    /* set the fill mode to NC_FILL for entire file */
    err = ncmpi_set_fill(ncid, NC_FILL, &old_mode); ERR
    if (verbose) {
        if (old_mode == NC_FILL) printf("The old fill mode is NC_FILL\n");
        else                     printf("The old fill mode is NC_NOFILL\n");
    }

    /* set the fill mode back to NC_NOFILL for entire file */
    err = ncmpi_set_fill(ncid, NC_NOFILL, NULL); ERR

    /* set the variable's fill mode to NC_FILL with default fill value */
    err = ncmpi_def_var_fill(ncid, fix_varid, 0, NULL); ERR

    /* set a customized fill value -1 */
    fill_value = -1;
    err = ncmpi_put_att_int(ncid, rec_varid, "_FillValue", NC_INT, 1, &fill_value);
    ERR

    /* do not forget to exit define mode */
    err = ncmpi_enddef(ncid);
    ERR

    /* now we are in data mode */
    start[0] = 0;
    start[1] = NX * rank;
    count[0] = NY;
    count[1] = NX;

    /* initialize user buffer */
    for (i=0; i<NY; i++)
        for (j=0; j<NX; j++)
             buf[i][j] = rank;

    /* do not write the variable in full */
    count[1]--;
    err = ncmpi_put_vara_int_all(ncid, fix_varid, start, count, &buf[0][0]); ERR

    err = ncmpi_inq_var_fill(ncid, fix_varid, &no_fill, &fill_value); ERR
    if (no_fill != 0)
        printf("Error: expecting no_fill to be 0\n");
    if (fill_value != NC_FILL_INT)
        printf("Error: expecting no_fill to be %ld but got %d\n",NC_FILL_INT,fill_value);

    /* fill the 1st record of the record variable */
    start[0] = 0;
    err = ncmpi_fill_var_rec(ncid, rec_varid, start[0]); ERR

    /* write to the 1st record */
    count[0] = 1;
    err = ncmpi_put_vara_int_all(ncid, rec_varid, start, count, &buf[0][0]); ERR

    /* fill the 2nd record of the record variable */
    start[0] = 1;
    err = ncmpi_fill_var_rec(ncid, rec_varid, start[0]); ERR

    /* write to the 2nd record */
    err = ncmpi_put_vara_int_all(ncid, rec_varid, start, count, &buf[0][0]); ERR

    err = ncmpi_close(ncid);
    ERR

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


/*********************************************************************
 *
 *  Copyright (C) 2014, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *********************************************************************/
/* $Id: vard_int.c 2239 2015-12-18 18:21:09Z wkliao $ */

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * This example shows how to use the vard API ncmpi_put_vard() and
 * ncmpi_get_vard() to write and read 2D record and fixed-size variables.
 *
 *    To compile:
 *        mpicc -O2 vard_int.c -o vard_int -lpnetcdf
 *
 * Example commands for MPI run and outputs from running ncmpidump on the
 * NC file produced by this example program:
 *
 *    % mpiexec -n 4 ./vard_int /pvfs2/wkliao/testfile.nc
 *
 * The expected results from the output file contents are:
 *
 *  % ncmpidump /pvfs2/wkliao/testfile.nc
 *    netcdf testfile {
 *    // file format: CDF-1
 *    dimensions:
 *           REC_DIM = UNLIMITED ; // (2 currently)
 *           X = 12 ;
 *           FIX_DIM = 2 ;
 *    variables:
 *           int rec_var(REC_DIM, X) ;
 *           int fix_var(FIX_DIM, X) ;
 *    data:
 *
 *     rec_var =
 *       0, 1, 2, 100, 101, 102, 200, 201, 202, 300, 301, 302,
 *       10, 11, 12, 110, 111, 112, 210, 211, 212, 310, 311, 312 ;
 *
 *     fix_var =
 *       0, 1, 2, 100, 101, 102, 200, 201, 202, 300, 301, 302,
 *       10, 11, 12, 110, 111, 112, 210, 211, 212, 310, 311, 312 ;
 *    }
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h> /* getopt() */
#include <mpi.h>
#include <pnetcdf.h>

#define NY 2
#define NX 3
#define ERR if (err!=NC_NOERR) {printf("Error at line %d: %s\n", __LINE__,ncmpi_strerror(err));}

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

/*----< main() >------------------------------------------------------------*/
int main(int argc, char **argv) {

    extern int optind;
    char         filename[256];
    int          i, j, err, ncid, verbose=1, varid0, varid1, dimids[2];
    int          rank, nprocs, array_of_blocklengths[2], buf[NY][NX];
    int          array_of_sizes[2], array_of_subsizes[2], array_of_starts[2];
    MPI_Offset   start[2], count[2], recsize, bufcount, len;
    MPI_Aint     array_of_displacements[2];
    MPI_Datatype buftype, rec_filetype, fix_filetype;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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

    if (verbose && rank == 0) printf("%s: example of using vard APIs\n",__FILE__);

    start[0] = 0; start[1] = NX*rank;
    count[0] = 2; count[1] = NX;

    /* create a new file for write */
    err = ncmpi_create(MPI_COMM_WORLD, filename, NC_CLOBBER, MPI_INFO_NULL,
                       &ncid); ERR

    /* define a 2D array */
    err = ncmpi_def_dim(ncid, "REC_DIM", NC_UNLIMITED, &dimids[0]); ERR
    err = ncmpi_def_dim(ncid, "X",       NX*nprocs,    &dimids[1]); ERR
    err = ncmpi_def_var(ncid, "rec_var", NC_INT, 2, dimids, &varid0); ERR
    err = ncmpi_def_dim(ncid, "FIX_DIM", 2, &dimids[0]); ERR
    err = ncmpi_def_var(ncid, "fix_var", NC_INT, 2, dimids, &varid1); ERR
    err = ncmpi_enddef(ncid); ERR

    /* create a file type for the record variable */
    err = ncmpi_inq_recsize(ncid, &recsize);
    for (i=0; i<count[0]; i++) {
        array_of_blocklengths[i] = count[1];
        array_of_displacements[i] = start[1]*sizeof(int) + recsize * i;
    }
    MPI_Type_create_hindexed(2, array_of_blocklengths, array_of_displacements,
                             MPI_INT, &rec_filetype);
    MPI_Type_commit(&rec_filetype);

    /* create a file type for the fixed-size variable */
    array_of_sizes[0]    = 2;
    array_of_sizes[1]    = NX*nprocs;
    array_of_subsizes[0] = count[0];
    array_of_subsizes[1] = count[1];
    array_of_starts[0]   = start[0];
    array_of_starts[1]   = start[1];
    MPI_Type_create_subarray(2, array_of_sizes, array_of_subsizes,
                             array_of_starts, MPI_ORDER_C,
                             MPI_INT, &fix_filetype);
    MPI_Type_commit(&fix_filetype);

    buftype = MPI_INT;
    bufcount = count[0] * count[1];

    /* initialize the contents of the array */
    for (j=0; j<NY; j++) for (i=0; i<NX; i++) buf[j][i] = rank*100 + j*10 + i;

    /* write the record variable */
    err = ncmpi_put_vard_all(ncid, varid0, rec_filetype, buf, bufcount,buftype);
    ERR

    /* check if the number of records changed to 2 */
    err = ncmpi_inq_unlimdim(ncid, &dimids[0]); ERR
    err = ncmpi_inq_dimlen(ncid, dimids[0], &len); ERR
    if (len != 2)
        printf("Error: number of records should be 2 but got %lld\n", len);

    /* write the fixed-size variable */
    err = ncmpi_put_vard_all(ncid, varid1, fix_filetype, buf, bufcount,buftype);
    ERR

    err = ncmpi_close(ncid); ERR

    /* open the same file and read back for validate */
    err = ncmpi_open(MPI_COMM_WORLD, filename, NC_NOWRITE, MPI_INFO_NULL,
                     &ncid); ERR

    err = ncmpi_inq_varid(ncid, "rec_var", &varid0); ERR
    err = ncmpi_inq_varid(ncid, "fix_var", &varid1); ERR

    /* read back fixed-size variable */
    err = ncmpi_get_vard_all(ncid, varid1, fix_filetype, buf, bufcount,buftype);
    ERR

    /* read back record variable */
    err = ncmpi_get_vard_all(ncid, varid0, rec_filetype, buf, bufcount,buftype);
    ERR

    err = ncmpi_close(ncid); ERR
    MPI_Type_free(&rec_filetype);
    MPI_Type_free(&fix_filetype);

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

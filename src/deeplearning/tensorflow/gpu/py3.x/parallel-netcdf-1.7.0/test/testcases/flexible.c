/*
 *  Copyright (C) 2013, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *  $Id: flexible.c 2133 2015-09-26 19:16:01Z wkliao $
 */

/*
 * This program tests the use of flexible API.
 * The write buffer is a 2D array of size NY x NX
 * The MPI data type for the buffer is defined by swapping the 1st and 2nd
 * rows of the array. It uses MPI_Type_create_hindex(). After the write, this
 * test reads back the array using regular and flexible get APIs (blocking and
 * nonblokcing) and check the contents.
 *
 * The expected reults from the output file contents are:
 * (when running on 1 MPI process)
 *
 *  % ncmpidump testfile.nc
 *    netcdf testfile {
 *    // file format: CDF-1
 *    dimensions:
 * 	   Y = UNLIMITED ; // (2 currently)
 * 	   X = 5 ;
 *    variables:
 * 	   int VAR(Y, X) ;
 *    data:
 * 
 *    var =
 *      1, 1, 1, 1, 1,
 *      0, 0, 0, 0, 0 ;
 *    }
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <pnetcdf.h>

#include <testutils.h>

#define NY 2
#define NX 5
#define ERR if (err!=NC_NOERR) {printf("Error at line %d: %s\n", __LINE__,ncmpi_strerror(err)); nerrs++;}

/*----< main() >------------------------------------------------------------*/
int main(int argc, char **argv) {

    char         filename[256];
    int          i, j, err, ncid, varid, dimids[2], debug=0;
    int          rank, nprocs, blocklengths[2], buf[NY][NX], *bufptr;
    int         *ncbuf, req, st, nerrs=0;
    int          array_of_sizes[2], array_of_subsizes[2], array_of_starts[2];
    MPI_Offset   start[2], count[2];
    MPI_Aint     a0, a1, disps[2];
    MPI_Datatype buftype;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc > 2) {
        if (!rank) printf("Usage: %s [filename]\n",argv[0]);
        MPI_Finalize();
        return 0;
    }
    strcpy(filename, "testfile.nc");
    if (argc == 2) strcpy(filename, argv[1]);
    MPI_Bcast(filename, 256, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        char cmd_str[256];
        sprintf(cmd_str, "*** TESTING C   %s for flexible put and get ", argv[0]);
        printf("%-66s ------ ", cmd_str); fflush(stdout);
    }

    err = ncmpi_create(MPI_COMM_WORLD, filename, NC_CLOBBER, MPI_INFO_NULL,
                       &ncid); ERR

    /* define a 2D array */
    err = ncmpi_def_dim(ncid, "Y", NC_UNLIMITED, &dimids[0]); ERR
    err = ncmpi_def_dim(ncid, "X", NX*nprocs,    &dimids[1]); ERR
    err = ncmpi_def_var(ncid, "var", NC_INT, 2, dimids, &varid); ERR
    err = ncmpi_enddef(ncid); ERR

    /* initialize the contents of the array */
    for (j=0; j<NY; j++) for (i=0; i<NX; i++) buf[j][i] = j;

    /* construct an MPI derived data type for swapping 1st row with 2nd row */
    blocklengths[0] = blocklengths[1] = NX;
    MPI_Get_address(buf[1], &a0);
    MPI_Get_address(buf[0], &a1);
    disps[0] = 0;
    disps[1] = a1 - a0;
    bufptr = buf[1];
    err = MPI_Type_create_hindexed(2, blocklengths, disps, MPI_INT, &buftype);
    if (err != MPI_SUCCESS) printf("MPI error MPI_Type_create_hindexed\n");
    MPI_Type_commit(&buftype);

    start[0] = 0; start[1] = NX*rank;
    count[0] = 2; count[1] = NX;
    if (debug) printf("put start=%lld %lld count=%lld %lld\n",start[0],start[1],count[0],count[1]);

    /* call flexible API */
    err = ncmpi_put_vara_all(ncid, varid, start, count, bufptr, 1, buftype); ERR
    MPI_Type_free(&buftype);

    /* check if the contents of buf are altered */
    for (j=0; j<NY; j++)
        for (i=0; i<NX; i++)
            if (buf[j][i] != j)
                printf("buf[%d][%d] != %d\n",j,i,buf[j][i]);
 
    /* check if root process can write to file header in data mode */
    err = ncmpi_rename_var(ncid, varid, "VAR"); ERR

    err = ncmpi_close(ncid); ERR

    /* open the same file and read back for validate */
    err = ncmpi_open(MPI_COMM_WORLD, filename, NC_NOWRITE, MPI_INFO_NULL,
                     &ncid); ERR

    err = ncmpi_inq_varid(ncid, "VAR", &varid); ERR

    /* initialize the contents of the array to a different value */
    for (j=0; j<NY; j++) for (i=0; i<NX; i++) buf[j][i] = -1;

    /* read back variable */
    start[0] = 0; start[1] = NX*rank;
    count[0] = 2; count[1] = NX;
    if (debug) printf("get start=%lld %lld count=%lld %lld\n",start[0],start[1],count[0],count[1]);

    err = ncmpi_get_vara_int_all(ncid, varid, start, count, buf[0]); ERR

    /* check if the contents of buf are expected */
    for (j=0; j<2; j++) {
        int val = (j == 0) ? 1 : 0;
        for (i=0; i<NX; i++)
            if (buf[j][i] != val) {
                printf("Unexpected buf[%d][%d]=%d != %d\n",j,i,buf[j][i],val);
                nerrs++;
            }
    }

    /* create a buftype with ghost cells on each side */
    ncbuf = (int *) malloc((count[0]+4)*(count[1]+4)*sizeof(int));
    array_of_sizes[0] = count[0]+4;
    array_of_sizes[1] = count[1]+4;
    array_of_subsizes[0] = count[0];
    array_of_subsizes[1] = count[1];
    array_of_starts[0] = 2;
    array_of_starts[1] = 2;
    MPI_Type_create_subarray(2, array_of_sizes, array_of_subsizes,
                             array_of_starts, MPI_ORDER_C,
                             MPI_INT, &buftype);
    MPI_Type_commit(&buftype);
    err = ncmpi_get_vara_all(ncid, varid, start, count, ncbuf, 1, buftype); ERR

    for (j=0; j<count[0]; j++) {
        for (i=0; i<count[1]; i++)
            if (buf[j][i] != ncbuf[(j+2)*(count[1]+4)+(i+2)]) {
                printf("Error: expecting ncbuf[%d][%d]=%d but got %d\n",
                       j,i,buf[j][i],ncbuf[(j+2)*(count[1]+4)+(i+2)]);
                nerrs++;
            }
    }
    for (i=0; i<(count[0]+4)*(count[1]+4); i++) ncbuf[i] = -1;

    err = ncmpi_iget_vara(ncid, varid, start, count, ncbuf, 1, buftype, &req); ERR
    err = ncmpi_wait_all(ncid, 1, &req, &st); ERR

    for (j=0; j<count[0]; j++) {
        for (i=0; i<count[1]; i++)
            if (buf[j][i] != ncbuf[(j+2)*(count[1]+4)+(i+2)]) {
                printf("Error: expecting ncbuf[%d][%d]=%d but got %d\n",
                       j,i,buf[j][i],ncbuf[(j+2)*(count[1]+4)+(i+2)]);
                nerrs++;
            }
    }

    MPI_Type_free(&buftype);
    free(ncbuf);

    err = ncmpi_close(ncid); ERR

    /* check if PnetCDF freed all internal malloc */
    MPI_Offset malloc_size, sum_size;
    err = ncmpi_inq_malloc_size(&malloc_size);
    if (err == NC_NOERR) {
        MPI_Reduce(&malloc_size, &sum_size, 1, MPI_OFFSET, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0 && sum_size > 0)
            printf("heap memory allocated by PnetCDF internally has %lld bytes yet to be freed\n",
                   sum_size);
    }

    MPI_Allreduce(MPI_IN_PLACE, &nerrs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        if (nerrs) printf(FAIL_STR,nerrs);
        else       printf(PASS_STR);
    }

    MPI_Finalize();
    return 0;
}

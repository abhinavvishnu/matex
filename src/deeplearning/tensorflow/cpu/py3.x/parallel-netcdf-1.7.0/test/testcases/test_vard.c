/*
 *  Copyright (C) 2014, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *  $Id: test_vard.c 2219 2015-12-11 22:30:03Z wkliao $
 */

/*
 * This program tests the vard API.
 * The write buffer is a 2D array of size NY x NX
 * The MPI data type for the buffer is defined by swapping the 1st and 2nd
 * rows of the array using a butype constructed by MPI_Type_create_hindex().
 * It also writes a fixed-size variable using a buftype constructed by
 * MPI_Type_create_subarray(). Both record and foxed-size variables are read
 * back using various filetypes and buftypes and check the contents.
 *
 * The expected reults from the output file contents are:
 * (when running on 1 MPI process)
 *
 *  % ncmpidump testfile.nc
 *    netcdf testfile {
 *    // file format: CDF-1
 *    dimensions:
 *           REC_DIM = UNLIMITED ; // (2 currently)
 *           X = 5 ;
 *           FIX_DIM = 2 ;
 *    variables:
 *           int rec_var(REC_DIM, X) ;
 *           int dummy_rec(REC_DIM, X) ;
 *           int fix_var(FIX_DIM, X) ;
 *    data:
 * 
 *    rec_var =
 *      10, 11, 12, 13, 14,
 *      0, 1, 2, 3, 4 ;
 * 
 *    dummy_rec =
 *      0, 0, 0, 0, 0,
 *      0, 0, 0, 0, 0 ;
 *
 *    fix_var =
 *      10, 11, 12, 13, 14,
 *      0, 1, 2, 3, 4 ;
 * }
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

#define CHECK_VALUE_PERMUTED { \
    for (j=0; j<count[0]; j++) { \
        int val = (j == 0) ? 1 : 0; \
        val = rank * 100 + val * 10; \
        for (i=0; i<count[1]; i++) \
            if (buf[j][i] != val+i) { \
                printf("line %d: expecting buf[%d][%d]=%d but got %d\n",__LINE__,j,i,val+i,buf[j][i]); \
                nerrs++; \
            } \
    } \
} \

#define CHECK_VALUE { \
    for (j=0; j<count[0]; j++) { \
        for (i=0; i<count[1]; i++) \
            if (buf[j][i] != rank*100+j*10+i) { \
                printf("line %d: expecting buf[%d][%d]=%d but got %d\n",__LINE__,j,i,rank*100+j*10+i,buf[j][i]); \
                nerrs++; \
            } \
    } \
}

static
int get_var_and_verify(int ncid,
                       int varid,
                       MPI_Offset *start,
                       MPI_Offset *count,
                       int **buf,
                       MPI_Datatype buftype,
                       MPI_Datatype ghost_buftype,
                       MPI_Datatype filetype)
{
    int i, j, rank, err, *ncbuf, nerrs=0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    ncbuf = (int *) malloc((count[0]+4)*(count[1]+4)*sizeof(int));

    /* clear the contents of the read buffer */
    for (j=0; j<count[0]; j++) for (i=0; i<count[1]; i++) buf[j][i] = -1;

    /* read back using regular vara API */
    err = ncmpi_get_vara_int_all(ncid, varid, start, count, buf[0]); ERR

    /* check if the contents of buf are expected */
    CHECK_VALUE_PERMUTED

    /* clear the contents of the read buffer */
    for (j=0; j<count[0]; j++) for (i=0; i<count[1]; i++) buf[j][i] = -1;

    /* read back using flexible vara API */
    err = ncmpi_get_vara_all(ncid, varid, start, count, buf[1], 1, buftype); ERR

    /* check if the contents of buf are expected */
    CHECK_VALUE

    /* clear the contents of the read buffer */
    for (j=0; j<count[0]; j++) for (i=0; i<count[1]; i++) buf[j][i] = -1;

    /* read back using vard API and permuted buftype */
    err = ncmpi_get_vard_all(ncid, varid, filetype, buf[1], 1, buftype); ERR

    /* check if the contents of buf are expected */
    CHECK_VALUE

    /* clear the contents of the read buffer */
    for (j=0; j<count[0]; j++) for (i=0; i<count[1]; i++) buf[j][i] = -1;

    /* read back using vard API and no buftype */
    err = ncmpi_get_vard_all(ncid, varid, filetype, buf[0], 0, MPI_DATATYPE_NULL); ERR

    /* check if the contents of buf are expected */
    CHECK_VALUE_PERMUTED

    /* clear the contents of the read buffer */
    for (i=0; i<(count[0]+4)*(count[1]+4); i++) ncbuf[i] = -1;

    /* read back using ghost buftype */
    err = ncmpi_get_vard_all(ncid, varid, filetype, ncbuf, 1, ghost_buftype); ERR

    for (j=0; j<count[0]; j++) {
        for (i=0; i<count[1]; i++)
            if (buf[j][i] != ncbuf[(j+2)*(count[1]+4)+(i+2)]) {
                printf("Error at line %d: expecting ncbuf[%d][%d]=%d but got %d\n",
                       __LINE__,j,i,buf[j][i],ncbuf[(j+2)*(count[1]+4)+(i+2)]);
                nerrs++;
            }
    }
    free(ncbuf);
    return nerrs;
}

/*----< main() >------------------------------------------------------------*/
int main(int argc, char **argv) {

    char         filename[256];
    int          i, j, err, ncid, varid0, varid1, varid2, dimids[2], nerrs=0;
    int          rank, nprocs, debug=0, blocklengths[2], **buf, *bufptr;
    int          array_of_sizes[2], array_of_subsizes[2], array_of_starts[2];
    MPI_Offset   start[2], count[2];
    MPI_Aint     a0, a1, disps[2];
    MPI_Datatype buftype, ghost_buftype, rec_filetype, fix_filetype;

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

    buf = (int**)malloc(NY * sizeof(int*));
    buf[0] = (int*)malloc(NY * NX * sizeof(int));
    for (i=1; i<NY; i++) buf[i] = buf[i-1] + NX;

    /* construct various MPI derived data types */

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

    /* create a file type for the fixed-size variable */
    array_of_sizes[0] = 2;
    array_of_sizes[1] = NX*nprocs;
    array_of_subsizes[0] = count[0];
    array_of_subsizes[1] = count[1];
    array_of_starts[0] = start[0];
    array_of_starts[1] = start[1];
    MPI_Type_create_subarray(2, array_of_sizes, array_of_subsizes,
                             array_of_starts, MPI_ORDER_C,
                             MPI_INT, &fix_filetype);
    MPI_Type_commit(&fix_filetype);

    /* create a buftype with ghost cells on each side */
    array_of_sizes[0] = count[0]+4;
    array_of_sizes[1] = count[1]+4;
    array_of_subsizes[0] = count[0];
    array_of_subsizes[1] = count[1];
    array_of_starts[0] = 2;
    array_of_starts[1] = 2;
    MPI_Type_create_subarray(2, array_of_sizes, array_of_subsizes,
                             array_of_starts, MPI_ORDER_C,
                             MPI_INT, &ghost_buftype);
    MPI_Type_commit(&ghost_buftype);

    /* create a new file for write */
    err = ncmpi_create(MPI_COMM_WORLD, filename, NC_CLOBBER, MPI_INFO_NULL,
                       &ncid); ERR

    /* define a 2D array */
    err = ncmpi_def_dim(ncid, "REC_DIM", NC_UNLIMITED, &dimids[0]); ERR
    err = ncmpi_def_dim(ncid, "X",       NX*nprocs,    &dimids[1]); ERR
    err = ncmpi_def_var(ncid, "rec_var", NC_INT, 2, dimids, &varid0); ERR
    err = ncmpi_def_var(ncid, "dummy_rec", NC_INT, 2, dimids, &varid2); ERR
    err = ncmpi_def_dim(ncid, "FIX_DIM", 2, &dimids[0]); ERR
    err = ncmpi_def_var(ncid, "fix_var", NC_INT, 2, dimids, &varid1); ERR
    err = ncmpi_enddef(ncid); ERR

    /* create a file type for the record variable */
    int *array_of_blocklengths=(int*) malloc(count[0]*sizeof(int));
    MPI_Aint *array_of_displacements=(MPI_Aint*) malloc(count[0]*sizeof(MPI_Aint));
    MPI_Offset recsize;
    err = ncmpi_inq_recsize(ncid, &recsize);
    for (i=0; i<count[0]; i++) {
        array_of_blocklengths[i] = count[1];
        array_of_displacements[i] = start[1]*sizeof(int) + recsize * i;
    }
    MPI_Type_create_hindexed(2, array_of_blocklengths, array_of_displacements,
                             MPI_INT, &rec_filetype);
    MPI_Type_commit(&rec_filetype);
    free(array_of_blocklengths);
    free(array_of_displacements);

    /* initialize the contents of the array */
    for (j=0; j<NY; j++) for (i=0; i<NX; i++) buf[j][i] = rank*100 + j*10 + i;

    /* write the record variable */
    err = ncmpi_put_vard_all(ncid, varid0, rec_filetype, bufptr, 1, buftype); ERR

    /* check if the contents of buf are altered */
    CHECK_VALUE

    /* check if root process can write to file header in data mode */
    err = ncmpi_rename_var(ncid, varid0, "rec_VAR"); ERR

    /* write the fixed-size variable */
    err = ncmpi_put_vard_all(ncid, varid1, fix_filetype, bufptr, 1, buftype); ERR

    /* check if the contents of buf are altered */
    CHECK_VALUE
 
    /* check if root process can write to file header in data mode */
    err = ncmpi_rename_var(ncid, varid0, "rec_var"); ERR

    /* test the same routines in independent data mode */
    err = ncmpi_begin_indep_data(ncid); ERR
    err = ncmpi_put_vard(ncid, varid0, rec_filetype, bufptr, 1, buftype); ERR
    CHECK_VALUE
    err = ncmpi_rename_var(ncid, varid0, "rec_VAR"); ERR
    err = ncmpi_put_vard(ncid, varid1, fix_filetype, bufptr, 1, buftype); ERR
    CHECK_VALUE
    err = ncmpi_rename_var(ncid, varid0, "rec_var"); ERR
    err = ncmpi_end_indep_data(ncid); ERR

    err = ncmpi_close(ncid); ERR

    /* open the same file and read back for validate */
    err = ncmpi_open(MPI_COMM_WORLD, filename, NC_NOWRITE, MPI_INFO_NULL,
                     &ncid); ERR

    err = ncmpi_inq_varid(ncid, "rec_var", &varid0); ERR
    err = ncmpi_inq_varid(ncid, "fix_var", &varid1); ERR

    nerrs += get_var_and_verify(ncid, varid0, start, count, buf, buftype, ghost_buftype, rec_filetype);
    nerrs += get_var_and_verify(ncid, varid1, start, count, buf, buftype, ghost_buftype, fix_filetype);

    err = ncmpi_close(ncid); ERR

    MPI_Type_free(&rec_filetype);
    MPI_Type_free(&fix_filetype);
    MPI_Type_free(&buftype);
    MPI_Type_free(&ghost_buftype);
    free(buf[0]); free(buf);

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

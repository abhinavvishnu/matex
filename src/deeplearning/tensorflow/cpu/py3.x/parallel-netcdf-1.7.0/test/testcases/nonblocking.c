/*
 *  Copyright (C) 2013, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *  $Id: nonblocking.c 2133 2015-09-26 19:16:01Z wkliao $
 */

/*
 * This program tests the use of nonblocking API.
 * The write buffer is a 2D array of size NY x NX
 * It writes the 2nd row of the memory buffer to the 1st row of the variable
 * array in file. Then it writes the 1st row of the memory buffer to the
 * 2nd row of the variable array in file.
 *
 * The expected reults from the output file contents are:
 * (when running on 1 MPI process)
 *
 *  % ncmpidump testfile.nc
 *    netcdf testfile {
 *    // file format: CDF-1
 *    dimensions:
 *         Y = UNLIMITED ; // (2 currently)
 *         X = 5 ;
 *    variables:
 *         int VAR(Y, X) ;
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

#define NY 4
#define NX 5
#define ERR if (err!=NC_NOERR) {printf("Error at line %d: %s\n", __LINE__,ncmpi_strerror(err)); exit(-1);}

/*----< main() >------------------------------------------------------------*/
int main(int argc, char **argv) {

    char       filename[256];
    int        i, j, err, ncid, varid, dimids[2], req[2], st[2], nerrs=0;
    int        rank, nprocs, buf[NY+1][NX];
    MPI_Offset start[2], count[2];
    MPI_Info   info;

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
        sprintf(cmd_str, "*** TESTING C   %s for using ncmpi_iput_vara_int() ", argv[0]);
        printf("%-66s ------ ", cmd_str); fflush(stdout);
    }

    MPI_Info_create(&info);
    /* When using PVFS2, unexpected buffer value error message might occur.
     * This is due to  a possible bug in ADIOI_PVFS2_OldWriteStrided() when
     * filetype is contiguous and buftype is non-contiguous.
     * Fix: Add ROMIO hint to force ADIO driever to use POSIX I/O */
    /* MPI_Info_set(info, "romio_pvfs2_posix_write", "enable"); */

    err = ncmpi_create(MPI_COMM_WORLD, filename, NC_CLOBBER, info, &ncid); ERR
    MPI_Info_free(&info);

    /* define a 2D array */
    err = ncmpi_def_dim(ncid, "Y", NC_UNLIMITED, &dimids[0]); ERR
    err = ncmpi_def_dim(ncid, "X", NX,    &dimids[1]); ERR
    err = ncmpi_def_var(ncid, "var", NC_INT, 2, dimids, &varid); ERR
    err = ncmpi_enddef(ncid); ERR

    /* initialize the contents of the array */
    for (j=0; j<NY+1; j++) for (i=0; i<NX; i++) buf[j][i] = j;

    start[0] = 2*rank; start[1] = 0;
    count[0] = 1;      count[1] = NX;

    /* call nonblocking API */
    err = ncmpi_iput_vara_int(ncid, varid, start, count, buf[1], &req[0]); ERR

    start[0] += 1;
    err = ncmpi_iput_vara_int(ncid, varid, start, count, buf[0], &req[1]); ERR

    st[0] = st[1] = NC_NOERR;
    err = ncmpi_wait_all(ncid, 2, req, st); ERR
    err = st[0]; ERR
    err = st[1]; ERR

    /* check if the contents of buf are altered */
    for (j=0; j<NY; j++)
        for (i=0; i<NX; i++)
            if (buf[j][i] != j)
                printf("Error: buf[%d][%d]=%d != %d\n",j,i,buf[j][i],j);
 
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
    start[0] = 2*rank; start[1] = 0;
    count[0] = 2;      count[1] = NX;
    err = ncmpi_get_vara_int_all(ncid, varid, start, count, buf[0]); ERR

    err = ncmpi_close(ncid); ERR

    /* check if the contents of buf are expected */
    for (j=0; j<2; j++) {
        int val = (j == 0) ? 1 : 0;
        for (i=0; i<NX; i++)
            if (buf[j][i] != val) {
                printf("Unexpected read buf[%d][%d]=%d, should be %d\n",
                       j,i,buf[j][i],val);
                nerrs++;
            }
    }

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

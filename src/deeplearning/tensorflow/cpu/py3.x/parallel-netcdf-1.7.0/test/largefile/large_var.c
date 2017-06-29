/*********************************************************************
 *
 *  Copyright (C) 2014, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *********************************************************************/
/* $Id: large_var.c 2300 2016-01-09 06:16:29Z wkliao $ */

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * This program is to test if MPI filetypes are defined correctly for accessing
 * arrays with more than 2G elements.
 *
 * This program calls ncmpi_put_vara_int_all() and ncmpi_iput_vara_int()
 * to write multiple subarray of a large 3D 4-byte integer array. It first
 * defines a netCDF variable of size 4 x 10 x 4294967296.
 * 1st write: subarray of 1 x 2 x 10 at the start 1 x 8 x (2G + rank * 10)
 * 2nd write: subarray of 1 x 2 x 5  at the start 3 x 8 x (2G + rank * 10)
 * 3rd write: subarray of 1 x 1 x 5  at the start 3 x 8 x (2G + rank * 10 + 5)
 * 4th write: subarray of 1 x 1 x 5  at the start 3 x 9 x (2G + rank * 10 + 5)
 *
 * The written contents are read back by a different process to check
 * correctness. Two reads are performed: PnetCDF and MPI-IO.
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <string.h> /* strcpy() */
#include <mpi.h>
#include <pnetcdf.h>
#include <testutils.h>

#define FOUR_G 4294967296
#define TWO_G  2147483648
#define ONE_G  1073741824

#define NZ 4
#define NY 10
#define NX FOUR_G

#define ERR if (err!=NC_NOERR) {printf("Error at line %d: err=%s (%s)\n", __LINE__,nc_err_code_name(err),ncmpi_strerror(err)); exit(-1);}

#ifndef WORDS_BIGENDIAN
/* Endianness byte swap: done in-place */
#define SWAP(x,y) {tmp = (x); (x) = (y); (y) = tmp;}
static void
swapn(void       *buf,
           MPI_Offset  nelems)
{
    int  i;
    unsigned char tmp, *op = (unsigned char*)buf;

    while (nelems-- > 0) {
        for (i=0; i<2; i++)
            SWAP(op[i], op[3-i])
        op += 4;
    }
}
#endif

int main(int argc, char** argv)
{
    char filename[256];
    int i, j, rank, nprocs, err, nerrs=0, bufsize;
    int ncid, cmode, varid, dimid[3], req[3], st[3], *buf, *buf_ptr;
    MPI_Offset offset, var_offset, start[3], count[3];
    MPI_File fh;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /* get command-line arguments */
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
        sprintf(cmd_str, "*** TESTING C   %s for writing to a large variable ", argv[0]);
        printf("%-66s ------ ", cmd_str); fflush(stdout);
    }

    /* create a new file for writing ----------------------------------------*/
    cmode = NC_CLOBBER | NC_64BIT_DATA;
    err = ncmpi_create(MPI_COMM_WORLD, filename, cmode, MPI_INFO_NULL, &ncid);
    ERR

    /* define dimensions Z, Y, and X */
    err = ncmpi_def_dim(ncid, "Z", NZ, &dimid[0]);
    ERR
    err = ncmpi_def_dim(ncid, "Y", NY, &dimid[1]);
    ERR
    err = ncmpi_def_dim(ncid, "X", NX, &dimid[2]);
    ERR

    /* define a 3D variable of integer type */
    err = ncmpi_def_var(ncid, "var", NC_INT, 3, dimid, &varid);
    ERR

    /* do not forget to exit define mode */
    err = ncmpi_enddef(ncid);
    ERR

    /* get the beginning of file offset for the varaiable */
    err = ncmpi_inq_varoffset(ncid, varid, &var_offset);
    ERR

    /* now we are in data mode */
    start[0] = 1;
    start[1] = NY - 2;
    start[2] = TWO_G + 10 * rank;
    count[0] = 1;
    count[1] = 2;
    count[2] = 10;

    bufsize = count[0]*count[1]*count[2];
    buf = (int*) malloc(bufsize * sizeof(int));
    for (i=0; i<bufsize; i++) buf[i] = rank*100 + i;

    err = ncmpi_put_vara_int_all(ncid, varid, start, count, buf);
    ERR

    /* now test nonblocking put */
    /* rearrange buffer contents */
    for (i=5;  i<10; i++) buf[i] = rank*100 + i + 5;
    for (i=10; i<15; i++) buf[i] = rank*100 + i - 5;

    start[0] = NZ-1;
    count[2] = 5;
    err = ncmpi_iput_vara_int(ncid, varid, start, count, buf, &req[0]);
    ERR

    start[2] += 5;
    count[1]  = 1;
    err = ncmpi_iput_vara_int(ncid, varid, start, count, buf+10, &req[1]);
    ERR

    start[1]++;
    err = ncmpi_iput_vara_int(ncid, varid, start, count, buf+15, &req[2]);
    ERR

    err = ncmpi_wait_all(ncid, 3, req, st);
    ERR

    err = ncmpi_close(ncid);
    ERR

    /* open the same file and read back for validation */
    err = ncmpi_open(MPI_COMM_WORLD, filename, NC_NOWRITE, MPI_INFO_NULL,
                     &ncid); ERR

    err = ncmpi_inq_varid(ncid, "var", &varid); ERR

    /* initialize the contents of the array to a different value */
    for (i=0; i<bufsize; i++) buf[i] = -1;

    /* read back subarray written by the process (rank+1)%nprocs */
    start[0] = 1;
    start[1] = NY - 2;
    start[2] = TWO_G + ((rank == nprocs - 1) ? 0 : 10 * (rank + 1));
    count[0] = 1;
    count[1] = 2;
    count[2] = 10;

    err = ncmpi_get_vara_int_all(ncid, varid, start, count, buf); ERR

    /* check if the contents of buf are expected */
    for (i=0; i<bufsize; i++) {
        int expected = (rank == nprocs - 1) ? i : (rank+1)*100 + i;
        if (buf[i] != expected) {
            printf("%d (at line %d): Unexpected read buf[%d]=%d, should be %d\n",
                   rank, __LINE__, i, buf[i], expected);
            nerrs++;
        }
    }

    for (i=0; i<bufsize; i++) buf[i] = -1;

    start[0] = NZ-1;
    err = ncmpi_get_vara_int_all(ncid, varid, start, count, buf); ERR

    /* check if the contents of buf are expected */
    for (i=0; i<bufsize; i++) {
        int expected = (rank == nprocs - 1) ? i : (rank+1)*100 + i;
        if (buf[i] != expected) {
            printf("%d (at line %d): Unexpected read buf[%d]=%d, should be %d\n",
                   rank, __LINE__, i, buf[i], expected);
            nerrs++;
        }
    }

    err = ncmpi_close(ncid); ERR

    /* MPI file open the same file and read back for validation */
    err = MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if (err != MPI_SUCCESS) {
        int errorStringLen;
        char errorString[MPI_MAX_ERROR_STRING];
        MPI_Error_string(err, errorString, &errorStringLen);
        printf("MPI error MPI_File_open : %s\n", errorString);
    }

    /* initialize the contents of the array to a different value */
    for (i=0; i<bufsize; i++) buf[i] = -1;

    /* read back subarray written by the process (rank+1)%nprocs */
    start[0] = 1;
    start[1] = NY - 2;
    start[2] = TWO_G + ((rank == nprocs - 1) ? 0 : 10 * (rank + 1));
    count[0] = 1;
    count[1] = 2;
    count[2] = 10;

    buf_ptr = buf;
    for (i=0; i<count[0]; i++) {
        for (j=0; j<count[1]; j++) {
            offset = var_offset + ((start[0] + i) * NY * NX + (start[1] + j) * NX + start[2]) * sizeof(int);
            MPI_File_read_at(fh, offset, buf_ptr, count[2], MPI_INT, &status);
#ifndef WORDS_BIGENDIAN
            swapn(buf_ptr, count[2]);
#endif
            buf_ptr += count[2];
        }
    }

    /* check if the contents of buf are expected */
    for (i=0; i<bufsize; i++) {
        int expected = (rank == nprocs - 1) ? i : (rank+1)*100 + i;
        if (buf[i] != expected) {
            printf("%d (at line %d): Unexpected read buf[%d]=%d, should be %d\n",
                   rank, __LINE__, i, buf[i], expected);
            nerrs++;
        }
    }

    for (i=0; i<bufsize; i++) buf[i] = -1;

    start[0] = NZ-1;

    buf_ptr = buf;
    for (i=0; i<count[0]; i++) {
        for (j=0; j<count[1]; j++) {
            offset = var_offset + ((start[0] + i) * NY * NX + (start[1] + j) * NX + start[2]) * sizeof(int);
            MPI_File_read_at(fh, offset, buf_ptr, count[2], MPI_INT, &status);
#ifndef WORDS_BIGENDIAN
            swapn(buf_ptr, count[2]);
#endif
            buf_ptr += count[2];
        }
    }

    /* check if the contents of buf are expected */
    for (i=0; i<bufsize; i++) {
        int expected = (rank == nprocs - 1) ? i : (rank+1)*100 + i;
        if (buf[i] != expected) {
            printf("%d (at line %d): Unexpected read buf[%d]=%d, should be %d\n",
                   rank, __LINE__, i, buf[i], expected);
            nerrs++;
        }
    }

    MPI_File_close(&fh);

    free(buf);

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


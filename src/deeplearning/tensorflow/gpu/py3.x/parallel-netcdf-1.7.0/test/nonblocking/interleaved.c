/*********************************************************************
 *
 *  Copyright (C) 2014, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *********************************************************************/
/* $Id: interleaved.c 2133 2015-09-26 19:16:01Z wkliao $ */

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * This program tests nonblocking APIs for handling interleaved file types.
 * It makes 3 calls to ncmpi_iput_vara_int(), where the first 2 interleaved and
 * the third one does not.
 * It first defines a netCDF variable of size 10 x 20.
 * First  write: a subarray of size 3 x 5 at the start offsets of 6 x 8
 * Second write: a subarray of size 2 x 5 at the start offsets of 6 x 13
 * Third  write: a subarray of size 1 x 5 at the start offsets of 8 x 13
 *
 * % mpiexec -n 1 ./interleaved
 * % ncmpidump testfile.nc
 * netcdf testfile {
 * // file format: CDF-5 (big variables)
 * dimensions:
 * 	Y = 10 ;
 * 	X = 20 ;
 * variables:
 * 	int var(Y, X) ;
 * data:
 *
 *  var =
 *   -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
 *   -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
 *   -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
 *   -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
 *   -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
 *   -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
 *   -1, -1, -1, -1, -1, -1, -1, -1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, -1, -1,
 *   -1, -1, -1, -1, -1, -1, -1, -1, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, -1, -1,
 *   -1, -1, -1, -1, -1, -1, -1, -1, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, -1, -1,
 *   -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 ;
 * }
 *

 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <string.h> /* strcpy() */
#include <mpi.h>
#include <pnetcdf.h>

#include <testutils.h>

#define NY 10
#define NX 20

#define ERR if (err!=NC_NOERR) {printf("Error at line %d: %s\n", __LINE__,ncmpi_strerror(err)); exit(-1);}

int main(int argc, char** argv)
{
    char filename[256];
    int i, j, rank, nprocs, err, nerrs=0, expected;
    int ncid, cmode, varid, dimid[2], req[3], st[3], *buf;
    MPI_Offset start[2], count[2];
    MPI_Info info;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /* this program is intended to run on one process */
    if (rank) goto fn_exit;

    /* get command-line arguments */
    if (argc > 2) {
        if (!rank) printf("Usage: %s [filename]\n",argv[0]);
        MPI_Finalize();
        return 0;
    }
    strcpy(filename, "testfile.nc");
    if (argc == 2) strcpy(filename, argv[1]);

    if (rank == 0) {
        char cmd_str[256];
        sprintf(cmd_str, "*** TESTING C   %s for writing interleaved fileviews ", argv[0]);
        printf("%-66s ------ ", cmd_str);
    }

    MPI_Info_create(&info);
    MPI_Info_set(info, "romio_cb_write", "disable");
    MPI_Info_set(info, "ind_wr_buffer_size", "8");
    /* these 2 hints are required to cause a core dump if r1758 fix is not
     * presented */

    /* create a new file for writing ----------------------------------------*/
    cmode = NC_CLOBBER | NC_64BIT_DATA;
    err = ncmpi_create(MPI_COMM_SELF, filename, cmode, info, &ncid);
    ERR

    MPI_Info_free(&info);

    /* define dimensions Y and X */
    err = ncmpi_def_dim(ncid, "Y", NY, &dimid[0]);
    ERR
    err = ncmpi_def_dim(ncid, "X", NX, &dimid[1]);
    ERR

    /* define a 2D variable of integer type */
    err = ncmpi_def_var(ncid, "var", NC_INT, 2, dimid, &varid);
    ERR

    /* do not forget to exit define mode */
    err = ncmpi_enddef(ncid);
    ERR

    /* now we are in data mode */
    buf = (int*) malloc(NY*NX * sizeof(int));
    for (i=0; i<NY*NX; i++) buf[i] = -1;

    /* fill the entire array with -1s */
    err = ncmpi_put_var_int_all(ncid, varid, buf); ERR

    /* rearrange buffer contents, as buf is 2D */
    for (i=0;  i<5;  i++) buf[i] = 10 + i;
    for (i=5;  i<10; i++) buf[i] = 10 + i +  5;
    for (i=10; i<15; i++) buf[i] = 10 + i + 10;
    start[0] = 6; start[1] = 8;
    count[0] = 3; count[1] = 5;
    err = ncmpi_iput_vara_int(ncid, varid, start, count, buf, &req[0]);
    ERR

    for (i=15; i<20; i++) buf[i] = 10 + i - 10;
    for (i=20; i<25; i++) buf[i] = 10 + i -  5;
    start[0] = 6; start[1] = 13;
    count[0] = 2; count[1] = 5;
    err = ncmpi_iput_vara_int(ncid, varid, start, count, buf+15, &req[1]);
    ERR

    for (i=25; i<30; i++) buf[i] = 10 + i;
    start[0] = 8; start[1] = 13;
    count[0] = 1; count[1] = 5;
    err = ncmpi_iput_vara_int(ncid, varid, start, count, buf+25, &req[2]);
    ERR

    err = ncmpi_wait_all(ncid, 3, req, st);
    ERR

    err = ncmpi_close(ncid);
    ERR

    /* open the same file and read back for validate */
    err = ncmpi_open(MPI_COMM_SELF, filename, NC_NOWRITE, MPI_INFO_NULL,
                     &ncid); ERR

    err = ncmpi_inq_varid(ncid, "var", &varid); ERR

    /* initialize the contents of the array to a different value */
    for (i=0; i<NY*NX; i++) buf[i] = -1;

    /* read the entire array */
    err = ncmpi_get_var_int_all(ncid, varid, buf); ERR

    /* check if the contents of buf are expected */
    expected = 10;
    for (j=6; j<9; j++) {
        for (i=8; i<18; i++) {
            if (buf[j*NX+i] != expected) {
                printf("%d: Unexpected read buf[%d]=%d, should be %d\n",
                       rank, i, buf[j*NX+i], expected);
                nerrs++;
            }
            expected++;
        }
    }

    err = ncmpi_close(ncid); ERR

    free(buf);

    /* check if PnetCDF freed all internal malloc */
    MPI_Offset malloc_size;
    err = ncmpi_inq_malloc_size(&malloc_size);
    if (err == NC_NOERR && malloc_size > 0)
        printf("heap memory allocated by PnetCDF internally has %lld bytes yet to be freed\n", malloc_size);

fn_exit:
    MPI_Allreduce(MPI_IN_PLACE, &nerrs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        if (nerrs) printf(FAIL_STR,nerrs);
        else       printf(PASS_STR);
    }

    MPI_Finalize();
    return 0;
}


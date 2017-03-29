/*********************************************************************
 *
 *  Copyright (C) 2012, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *********************************************************************/
/* $Id: pnetcdf-write-buffered.c 1153 2013-02-03 17:45:55Z wkliao $ */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pnetcdf.h>

#define ERRCODE 2
#define ERR(e) {printf("Error at line %d: err=%d %s\n", __LINE__, e, ncmpi_strerror(e)); exit(ERRCODE);}

/*----< main() >------------------------------------------------------------*/
int main(int argc, char **argv) {
    int i, j, ncid, dimid[2], varid, err, rank, nprocs, cmode;
    int req[2], status[2]; 
    float  var[4][6];
    MPI_Offset start[2], count[2], stride[2], imap[2];
    MPI_Offset bufsize;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc != 2) {
        if (rank == 0) printf("Usage: %s filename\n", argv[0]);
        MPI_Finalize();
        exit(-1);
    }

    cmode = NC_CLOBBER | NC_64BIT_DATA;
    if (NC_NOERR != (err = ncmpi_create(MPI_COMM_WORLD, argv[1],
                                        cmode, MPI_INFO_NULL, &ncid)))
       ERR(err);

    /* define a variable of a 6 x (4*nprocs) int64 array in the nc file */
    if (NC_NOERR != (err = ncmpi_def_dim(ncid, "Y", 6, &dimid[0])))
        ERR(err);
    if (NC_NOERR != (err = ncmpi_def_dim(ncid, "X", 4*nprocs, &dimid[1])))
        ERR(err);
    if (NC_NOERR != (err = ncmpi_def_var(ncid, "var", NC_INT64, 2, dimid,
                                         &varid)))
        ERR(err);
    if (NC_NOERR != (err = ncmpi_enddef(ncid)))
        ERR(err);

    /* set the contents of the local write buffer var, a 4 x 6 float array
       for example, for rank == 2, var[4][6] =
          48, 49, 50, 51, 52, 53,
          54, 55, 56, 57, 58, 59,
          60, 61, 62, 63, 64, 65,
          66, 67, 68, 69, 70, 71
     */
    for (j=0; j<4; j++)
        for (i=0; i<6; i++)
            var[j][i] = j*6+i + rank*24;

    /* bufsize must be the max of data type converted before and after */
    bufsize = 4*6*sizeof(long long);  /* as var is of NC_INT64 */
    if (NC_NOERR != (err = ncmpi_buffer_attach(ncid, bufsize)))
        ERR(err);

    /* write var to the NC variable in the matrix transposed way */
    count[0]  = 6; count[1]  = 2;
    stride[0] = 1; stride[1] = 1;
    imap[0]   = 1; imap[1]   = 6;

    /* write to the 1st two columns of the variable in matrix transposed way */
    start[0]  = 0;
    start[1]  = rank*4;
    if (NC_NOERR != (err = ncmpi_bput_varm_float(ncid, varid, start, count,
                                           stride, imap, &var[0][0], &req[0])))
        ERR(err);

    /* write to the 2nd two columns of the variable in transposed way */
    start[0]  = 0;
    start[1]  = rank*4+2;
    if (NC_NOERR != (err = ncmpi_bput_varm_float(ncid, varid, start, count,
                                           stride, imap, &var[2][0], &req[1])))
        ERR(err);

    /* You are now free to change contents of the buffer, var.
       It will not change the data supposed to be written in the file.
     */
    for (j=0; j<4; j++)
        for (i=0; i<6; i++)
            var[j][i] = -1.1;  /* or any numbers */

    if (NC_NOERR != (err = ncmpi_wait_all(ncid, 2, req, status)))
        ERR(err);

    /* check each bput status */
    for (i=0; i<2; i++)
        if (status[i] != NC_NOERR) ERR(status[i]);

    if (NC_NOERR != (err = ncmpi_buffer_detach(ncid)))
        ERR(err);

    if (NC_NOERR != (err = ncmpi_close(ncid))) ERR(err);

    /* The output from command "ncmpidump test.nc" is shown below if run
       this example on 4 processes.

       netcdf test {
       // file format: CDF-5 (big variables)
       dimensions:
              Y = 6 ;
              X = 16 ;
       variables:
              int64 var(Y, X) ;
      data:

       var =
        0,  6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90,
        1,  7, 13, 19, 25, 31, 37, 43, 49, 55, 61, 67, 73, 79, 85, 91,
        2,  8, 14, 20, 26, 32, 38, 44, 50, 56, 62, 68, 74, 80, 86, 92,
        3,  9, 15, 21, 27, 33, 39, 45, 51, 57, 63, 69, 75, 81, 87, 93,
        4, 10, 16, 22, 28, 34, 40, 46, 52, 58, 64, 70, 76, 82, 88, 94,
        5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65, 71, 77, 83, 89, 95 ;
        <---- P0 ---->|<---- P1 ----->|<---- P2 ----->|<---- P3 ---->
     */
    MPI_Finalize();
    return 0;
}


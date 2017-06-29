/*
 *  Copyright (C) 2013, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id: test_inq_format.c 2250 2015-12-20 21:12:59Z wkliao $ */

/* This program tests if PnetCDF can report correct file formats */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <pnetcdf.h>
#include <testutils.h>

#define ERR {if(err!=NC_NOERR) {printf("Error(%d) at line %d: %s\n",err,__LINE__,ncmpi_strerror(err)); nerrs++; }}

int main(int argc, char **argv) {
    char dir_name[256], filename[256];
    int err, rank, nerrs=0, format, ncid;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc != 2) {
        if (!rank) printf("Usage: %s dir_name\n",argv[0]);
        MPI_Finalize();
        return 0;
    }
    strcpy(dir_name, argv[1]);
    MPI_Bcast(dir_name, 256, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        char cmd_str[256];
        sprintf(cmd_str, "*** TESTING C   %s for inquiring CDF file formats ", argv[0]);
        printf("%-66s ------ ", cmd_str);
    }

    /* test CDF-1 -----------------------------------------------------------*/
    sprintf(filename,"%s/test_cdf1.nc",dir_name);
    err = ncmpi_open(MPI_COMM_WORLD, filename, 0, MPI_INFO_NULL, &ncid); ERR
    err = ncmpi_inq_format(ncid, &format); ERR
    if (format != NC_FORMAT_CLASSIC) {
        printf("Error (line=%d): expecting CDF-1 format for file %s but got %d\n",
               __LINE__,filename,format);
        nerrs++;
    }
    err = ncmpi_close(ncid); ERR
  
    err = ncmpi_inq_file_format(filename, &format); ERR
    if (format != NC_FORMAT_CLASSIC) {
        printf("Error (line=%d): expecting CDF-1 format for file %s but got %d\n",
               __LINE__,filename,format);
        nerrs++;
    }

    /* test CDF-2 -----------------------------------------------------------*/
    sprintf(filename,"%s/test_cdf2.nc",dir_name);
    err = ncmpi_open(MPI_COMM_WORLD, filename, 0, MPI_INFO_NULL, &ncid); ERR
    err = ncmpi_inq_format(ncid, &format); ERR
    if (format != NC_FORMAT_CDF2) {
        printf("Error (line=%d): expecting CDF-2 format for file %s but got %d\n",
               __LINE__,filename,format);
        nerrs++;
    }
    err = ncmpi_close(ncid); ERR
  
    err = ncmpi_inq_file_format(filename, &format); ERR
    if (format != NC_FORMAT_CDF2) {
        printf("Error (line=%d): expecting CDF-2 format for file %s but got %d\n",
               __LINE__,filename,format);
        nerrs++;
    }

    /* test CDF-5 -----------------------------------------------------------*/
    sprintf(filename,"%s/test_cdf5.nc",dir_name);
    err = ncmpi_open(MPI_COMM_WORLD, filename, 0, MPI_INFO_NULL, &ncid); ERR
    err = ncmpi_inq_format(ncid, &format); ERR
    if (format != NC_FORMAT_CDF5) {
        printf("Error (line=%d): expecting CDF-5 format for file %s but got %d\n",
               __LINE__,filename,format);
        nerrs++;
    }
    err = ncmpi_close(ncid); ERR
  
    err = ncmpi_inq_file_format(filename, &format); ERR
    if (format != NC_FORMAT_CDF5) {
        printf("Error (line=%d): expecting CDF-5 format for file %s but got %d\n",
               __LINE__,filename,format);
        nerrs++;
    }

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

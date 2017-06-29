/*********************************************************************
 *
 *  Copyright (C) 2014, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *********************************************************************/
/* $Id: modes.c 2219 2015-12-11 22:30:03Z wkliao $ */

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * This program tests if the correct error codes are returns given various
 * create/open modes.
 *
 * NC_EINVAL_CMODE should be returned when creating a file using
 * comde with both NC_64BIT_OFFSET & NC_64BIT_DATA flags set.
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <string.h> /* strcpy() */
#include <unistd.h> /* unlink(), access() */
#include <mpi.h>
#include <pnetcdf.h>

#include <testutils.h>

#define ERR {if(err!=NC_NOERR)printf("Error at line=%d: %s\n", __LINE__, ncmpi_strerror(err));}

#define EXPECT_ERR(err_no) \
    if (err != err_no) { \
        nerrs++; \
        printf("Error at line %d: expect error code %s but got %s\n", \
               __LINE__,nc_err_code_name(err_no),nc_err_code_name(err)); \
    }

static
int check_modes(char *filename)
{
    int rank, err, nerrs=0, file_exist;
    int ncid, cmode;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* delete the file and ignore error */
    unlink(filename);
    MPI_Barrier(MPI_COMM_WORLD);

    /* create a new file and test various cmodes ----------------------------*/
    cmode = NC_CLOBBER;

    /* It is illegal to use both NC_64BIT_OFFSET and NC_64BIT_DATA together */
    cmode |= NC_64BIT_OFFSET | NC_64BIT_DATA;

    err = ncmpi_create(MPI_COMM_WORLD, filename, cmode, MPI_INFO_NULL, &ncid);
    EXPECT_ERR(NC_EINVAL_CMODE)

    /* The file should not be created */
    file_exist = 0;
    if (rank == 0 && access(filename, F_OK) == 0) file_exist = 1;
    MPI_Bcast(&file_exist, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (file_exist) {
        printf("Error at line %d: file (%s) should not be created\n", __LINE__, filename);
        nerrs++;
    }

    /* delete the file and ignore error */
    unlink(filename);
    MPI_Barrier(MPI_COMM_WORLD);

    /* Collectively opening a non-existing file for read, expect error code
     * NC_ENOENT on all processes */
    err = ncmpi_open(MPI_COMM_WORLD, filename, NC_NOWRITE, MPI_INFO_NULL, &ncid);
    EXPECT_ERR(NC_ENOENT)

    file_exist = 0;
    if (rank == 0 && access(filename, F_OK) == 0) file_exist = 1;
    MPI_Bcast(&file_exist, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (file_exist) {
        printf("Error at line %d: file (%s) should not be created\n", __LINE__, filename);
        nerrs++;
    }

    /* delete the file and ignore error */
    unlink(filename);
    MPI_Barrier(MPI_COMM_WORLD);

    /* Collectively opening a non-existing file for write, expect error code
     * NC_ENOENT on all processes */
    err = ncmpi_open(MPI_COMM_WORLD, filename, NC_WRITE, MPI_INFO_NULL, &ncid);
    EXPECT_ERR(NC_ENOENT)

    file_exist = 0;
    if (rank == 0 && access(filename, F_OK) == 0) file_exist = 1;
    MPI_Bcast(&file_exist, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (file_exist) {
        printf("Error at line %d: file (%s) should not be created\n", __LINE__, filename);
        nerrs++;
    }

    /* delete the file and ignore error */
    unlink(filename);

    return nerrs;
}

int main(int argc, char** argv)
{
    char filename[256];
    int rank, err, nerrs=0;

    MPI_Init(&argc, &argv);
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
        sprintf(cmd_str, "*** TESTING C   %s for file create/open modes ", argv[0]);
        printf("%-66s ------ ", cmd_str); fflush(stdout);
    }

    /* test under safe mode enabled */
    setenv("PNETCDF_SAFE_MODE", "1", 1);
    nerrs += check_modes(filename);

    /* test under safe mode disabled */
    setenv("PNETCDF_SAFE_MODE", "0", 1);
    nerrs += check_modes(filename);

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


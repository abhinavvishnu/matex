/*********************************************************************
 *
 *  Copyright (C) 2014, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *********************************************************************/
/* $Id: get_vara.c 2245 2015-12-20 18:39:52Z wkliao $ */

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * This example is the read counterpart of example put_vara.c. It shows how to
 * use ncmpi_get_vara_int_all() to read a 2D 4-byte integer array in parallel.
 * It also reads a global attribute and two attribute of variable named "var".
 * The data partitioning pattern is a column-wise partitioning across all
 * processes. Each process reads a subarray of size local_ny * local_nx.
 *
 *    To compile:
 *        mpicc -O2 get_vara.c -o get_vara -lpnetcdf
 *
 * Input file is the output file produced by put_vara.c. Here is the CDL dumped
 * from running ncmpidump.
 *
 *    % ncmpidump /pvfs2/wkliao/testfile.nc
 *    netcdf testfile {
 *    // file format: CDF-1
 *    dimensions:
 *            y = 10 ;
 *            x = 16 ;
 *    variables:
 *            int var(y, x) ;
 *                var:str_att_name = "example attribute of type text." ;
 *                var:float_att_name = 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f ;
 *    // global attributes:
 *                :history = "Wed Apr 30 11:18:58 2014\n",
 *       "" ;
 *    data:
 *
 *     var =
 *         0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
 *         0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
 *         0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
 *         0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
 *         0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
 *         0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
 *         0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
 *         0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
 *         0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
 *         0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3 ;
 *    }
 *
 * Example command for MPI run:
 *
 *    % mpiexec -n 4 ./get_vara /pvfs2/wkliao/testfile.nc
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <string.h> /* strcpy() */
#include <unistd.h> /* getopt() */
#include <mpi.h>
#include <pnetcdf.h>

#define CHECK_ERR { \
    if (err!=NC_NOERR) { \
        printf("Error at line=%d: %s\n", __LINE__, ncmpi_strerror(err)); \
        goto fn_exit; \
    } \
}

static void
usage(char *argv0)
{
    char *help =
    "Usage: %s [-h] | [-q] [file_name]\n"
    "       [-h] Print help\n"
    "       [-q] Quiet mode (reports when fail)\n"
    "       [filename] input netCDF file name\n";
    fprintf(stderr, help, argv0);
}

int main(int argc, char** argv)
{
    extern int optind;
    char *filename="testfile.nc", str_att[NC_MAX_NAME];
    int i, rank, nprocs, err, verbose=1, ncid, varid, dimid[2], *buf;
    float *float_att;
    MPI_Offset len, global_ny, global_nx, local_ny, local_nx;
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
    if (argc == 1) filename = argv[0]; /* optional argument */

    /* open an existing file for reading -------------------------------------*/
    err = ncmpi_open(MPI_COMM_WORLD, filename, NC_NOWRITE, MPI_INFO_NULL, &ncid);
    CHECK_ERR

    /* get global attribute named "history" */
    err = ncmpi_get_att_text(ncid, NC_GLOBAL, "history", str_att);
    CHECK_ERR
    err = ncmpi_inq_attlen(ncid, NC_GLOBAL, "history", &len);
    CHECK_ERR
    str_att[len] = '\0'; /* add a NULL char at the end */
    if (rank == 0 && verbose)
        printf("global attribute \"history\" of text: %s\n",str_att);

    /* get dimension IDs for dimensions Y and X */
    err = ncmpi_inq_dimid(ncid, "Y", &dimid[0]);
    CHECK_ERR
    err = ncmpi_inq_dimid(ncid, "X", &dimid[1]);
    CHECK_ERR

    /* get dimension lengths for dimensions Y and X */
    err = ncmpi_inq_dimlen(ncid, dimid[0], &global_ny);
    CHECK_ERR
    err = ncmpi_inq_dimlen(ncid, dimid[1], &global_nx);
    CHECK_ERR

    /* get the variable ID of a 2D variable of integer type */
    err = ncmpi_inq_varid(ncid, "var", &varid);
    CHECK_ERR

    /* get variable's attribute named "str_att_name" */
    err = ncmpi_get_att_text(ncid, varid, "str_att_name", str_att);
    CHECK_ERR
    err = ncmpi_inq_attlen(ncid, varid, "str_att_name", &len);
    CHECK_ERR
    str_att[len] = '\0'; /* add a NULL char at the end */
    if (rank == 0 && verbose)
        printf("variable attribute \"str_att_name\" of type text = \"%s\"\n",
               str_att);

    /* get the length of variable's attribute named "float_att_name" */
    err = ncmpi_inq_attlen(ncid, varid, "float_att_name", &len);
    CHECK_ERR

    /* get attribute contents */
    float_att = (float*) malloc(len * sizeof(float));
    err = ncmpi_get_att_float(ncid, varid, "float_att_name", float_att);
    CHECK_ERR

    /* the local array size */
    local_ny = global_ny;
    local_nx = global_nx / nprocs;
    buf = (int*) malloc(local_nx * local_ny * sizeof(int));

    /* prepare reading subarray */
    start[0] = 0;
    start[1] = local_nx * rank;
    count[0] = local_ny;
    count[1] = local_nx;

    /* read a subarray in collective mode */
    err = ncmpi_get_vara_int_all(ncid, varid, start, count, buf);
    CHECK_ERR

    err = ncmpi_close(ncid);
    CHECK_ERR

    /* check if there is any PnetCDF internal malloc residue */
    MPI_Offset malloc_size, sum_size;
    err = ncmpi_inq_malloc_size(&malloc_size);
    if (err == NC_NOERR) {
        MPI_Reduce(&malloc_size, &sum_size, 1, MPI_OFFSET, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0 && sum_size > 0)
            printf("heap memory allocated by PnetCDF internally has %lld bytes yet to be freed\n",
                   sum_size);
    }

fn_exit:
    MPI_Finalize();
    return 0;
}


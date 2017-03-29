/*********************************************************************
 *
 *  Copyright (C) 2013, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *********************************************************************/
/* $Id: put_vara.c 2245 2015-12-20 18:39:52Z wkliao $ */

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * This example shows how to use ncmpi_put_vara_int_all() to write a 2D
 * 4-byte integer array in parallel. It first defines a netCDF variable of
 * size global_nx * global_ny where
 *    global_ny == NY and
 *    global_nx == (NX * number of MPI processes).
 * The data partitioning pattern is a column-wise partitioning across all
 * processes. Each process writes a subarray of size ny * nx.
 *
 *    To compile:
 *        mpicc -O2 put_vara.c -o put_vara -lpnetcdf
 *
 * Example commands for MPI run and outputs from running ncmpidump on the
 * NC file produced by this example program:
 *
 *    % mpiexec -n 4 ./put_vara /pvfs2/wkliao/testfile.nc
 *
 *    % ncmpidump /pvfs2/wkliao/testfile.nc
 *    netcdf testfile {
 *    // file format: CDF-5 (big variables)
 *    dimensions:
 *            y = 10 ;
 *            x = 16 ;
 *    variables:
 *            int var(y, x) ;
 *                var:str_att_name = "example attribute of type text." ;
 *                var:float_att_name = 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f ;
 *                var:int64_att_name = 10000000000L ;
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
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <string.h> /* strcpy() */
#include <unistd.h> /* getopt() */
#include <time.h>   /* time() localtime(), asctime() */
#include <assert.h>
#include <mpi.h>
#include <pnetcdf.h>

#define NY 10
#define NX 4


#define ERR {if(err!=NC_NOERR)printf("Error at line=%d: %s\n", __LINE__, ncmpi_strerror(err));}

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

int main(int argc, char** argv)
{
    extern int optind;
    char filename[256];
    int i, j, rank, nprocs, verbose=1, err;
    int ncid, cmode, varid, dimid[2], buf[NY][NX];
    char str_att[128];
    float float_att[100];
    MPI_Offset  global_ny, global_nx;
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
    if (argc == 1) strcpy(filename, argv[0]); /* optional argument */
    else strcpy(filename, "testfile.nc");

    MPI_Bcast(filename, 256, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (verbose && rank == 0) printf("%s: example of using put_vara APIs\n",__FILE__);

    /* create a new file for writing ----------------------------------------*/
    cmode = NC_CLOBBER | NC_64BIT_DATA;
    err = ncmpi_create(MPI_COMM_WORLD, filename, cmode, MPI_INFO_NULL, &ncid);
    ERR

    /* the global array is NY * (NX * nprocs) */
    global_ny = NY;
    global_nx = NX * nprocs;

    for (i=0; i<NY; i++)
        for (j=0; j<NX; j++)
             buf[i][j] = rank;

    /* add a global attribute: a time stamp at rank 0 */
    time_t ltime = time(NULL); /* get the current calendar time */
    asctime_r(localtime(&ltime), str_att);

    /* make sure the time string are consistent among all processes */
    MPI_Bcast(str_att, strlen(str_att), MPI_CHAR, 0, MPI_COMM_WORLD);

    err = ncmpi_put_att_text(ncid, NC_GLOBAL, "history", strlen(str_att),
                             &str_att[0]);
    ERR

    /* define dimensions x and y */
    err = ncmpi_def_dim(ncid, "Y", global_ny, &dimid[0]);
    ERR
    err = ncmpi_def_dim(ncid, "X", global_nx, &dimid[1]);
    ERR

    /* define a 2D variable of integer type */
    err = ncmpi_def_var(ncid, "var", NC_INT, 2, dimid, &varid);
    ERR

    /* add attributes to the variable */
    strcpy(str_att, "example attribute of type text.");
    err = ncmpi_put_att_text(ncid, varid, "str_att_name", strlen(str_att),
                             &str_att[0]);
    ERR

    for (i=0; i<8; i++) float_att[i] = i;
    err = ncmpi_put_att_float(ncid, varid, "float_att_name", NC_FLOAT, 8,
                              &float_att[0]);
    ERR
    long long int64_att=10000000000LL;
    err = ncmpi_put_att_longlong(ncid, varid, "int64_att_name", NC_INT64, 1,
                              &int64_att);
    ERR

    /* do not forget to exit define mode */
    err = ncmpi_enddef(ncid);
    ERR

    /* now we are in data mode */
    start[0] = 0;
    start[1] = NX * rank;
    count[0] = NY;
    count[1] = NX;

    err = ncmpi_put_vara_int_all(ncid, varid, start, count, &buf[0][0]);
    ERR

    err = ncmpi_close(ncid);
    ERR

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


/*********************************************************************
 *
 *  Copyright (C) 2013, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *********************************************************************/
/* $Id: hints.c 1669 2014-05-24 18:28:16Z wkliao $ */

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * This example sets two PnetCDF hints:
 *    nc_header_align_size and nc_var_align_size
 * and prints the hint values as well as the header size, header extent, and
 * two variables' starting file offsets.
 *
 * The compile and run commands are given below.
 *
 *    % mpicc -O2 -o hints hints.c -lpnetcdf
 *
 *    % mpiexec -l -n 4 ./hints /pvfs2/wkliao/testfile.nc
 *
 *    nc_header_align_size      set to = 1024
 *    nc_var_align_size         set to = 512
 *    nc_header_read_chunk_size set to = 256
 *    header size                      = 252
 *    header extent                    = 1024
 *    var_zy start file offset         = 1024
 *    var_yx start file offset         = 3072
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <string.h> /* strcpy() */
#include <unistd.h> /* getopt() */
#include <assert.h>
#include <mpi.h>
#include <pnetcdf.h>

#define NZ 5
#define NY 5
#define NX 5

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

static
void print_hints(int ncid,
                 int varid0,
                 int varid1)
{
    char value[MPI_MAX_INFO_VAL];
    int err, len, flag;
    MPI_Offset header_size, header_extent, var_zy_start, var_yx_start;
    MPI_Offset h_align=-1, v_align=-1, h_chunk=-1;
    MPI_Info info_used;

    err = ncmpi_inq_header_size  (ncid, &header_size);      ERR
    err = ncmpi_inq_header_extent(ncid, &header_extent);    ERR
    err = ncmpi_inq_varoffset(ncid, varid0, &var_zy_start); ERR
    err = ncmpi_inq_varoffset(ncid, varid1, &var_yx_start); ERR

    err = ncmpi_inq_file_info(ncid, &info_used); ERR
    MPI_Info_get_valuelen(info_used, "nc_header_align_size", &len, &flag);
    if (flag) {
        MPI_Info_get(info_used, "nc_header_align_size", len+1, value, &flag);
        h_align = atoll(value);
    }
        MPI_Info_get_valuelen(info_used, "nc_var_align_size", &len, &flag);
    if (flag) {
        MPI_Info_get(info_used, "nc_var_align_size", len+1, value, &flag);
        v_align = atoll(value);
    }
    MPI_Info_get_valuelen(info_used, "nc_header_read_chunk_size", &len, &flag);
    if (flag) {
        MPI_Info_get(info_used, "nc_header_read_chunk_size", len+1, value,&flag);
        h_chunk = atoll(value);
    }
    MPI_Info_free(&info_used);

    if (h_align == -1)
        printf("nc_header_align_size      is NOT set\n");
    else
        printf("nc_header_align_size      set to = %lld\n", h_align);

    if (v_align == -1)
        printf("nc_var_align_size         is NOT set\n");
    else
        printf("nc_var_align_size         set to = %lld\n", v_align);
    if (h_chunk == -1)
        printf("nc_header_read_chunk_size is NOT set\n");
    else
        printf("nc_header_read_chunk_size set to = %lld\n", h_chunk);

    printf("header size                      = %lld\n", header_size);
    printf("header extent                    = %lld\n", header_extent);
    printf("var_zy start file offset         = %lld\n", var_zy_start);
    printf("var_yx start file offset         = %lld\n", var_yx_start);
}

int main(int argc, char** argv)
{
    extern int optind;
    char *filename="testfile.nc";
    int i, rank, nprocs, verbose=1, err;
    int ncid, cmode, varid0, varid1, dimid[3], *buf_zy;
    float *buf_yx;
    MPI_Offset start[2], count[2];
    MPI_Info info;

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

    MPI_Info_create(&info);
    MPI_Info_set(info, "nc_header_align_size",      "1024"); /* size in bytes */
    MPI_Info_set(info, "nc_var_align_size",         "512");  /* size in bytes */
    MPI_Info_set(info, "nc_header_read_chunk_size", "256");  /* size in bytes */
    /* note that set the above values to 1 to disable the alignment */

    /* create a new file for writing ----------------------------------------*/
    cmode = NC_CLOBBER | NC_64BIT_DATA;
    err = ncmpi_create(MPI_COMM_WORLD, filename, cmode, info, &ncid); ERR
    MPI_Info_free(&info);

    /* define 3 dimensions */
    err = ncmpi_def_dim(ncid, "Z", NZ*nprocs, &dimid[0]); ERR
    err = ncmpi_def_dim(ncid, "Y", NY*nprocs, &dimid[1]); ERR
    err = ncmpi_def_dim(ncid, "X", NX*nprocs, &dimid[2]); ERR

    /* define a variable of size (NZ * nprocs) * (NY * nprocs) */
    err = ncmpi_def_var(ncid, "var_zy", NC_INT,   2, &dimid[0], &varid0); ERR
    /* define a variable of size (NY * nprocs) * (NX * nprocs) */
    err = ncmpi_def_var(ncid, "var_yx", NC_FLOAT, 2, &dimid[1], &varid1); ERR
    err = ncmpi_enddef(ncid); ERR

    /* var_zy is partitioned along Z dimension */
    buf_zy = (int*) malloc(NZ * (NY * nprocs) * sizeof(int));
    for (i=0; i<NZ*(NY*nprocs); i++) buf_zy[i] = i;

    start[0] = NZ * rank; start[1] = 0;
    count[0] = NZ;        count[1] = NY * nprocs;
    err = ncmpi_put_vara_int_all(ncid, varid0, start, count, buf_zy); ERR

    /* var_yx is partitioned along X dimension */
    buf_yx = (float*) malloc((NY * nprocs) * NX * sizeof(float));
    for (i=0; i<(NY*nprocs)*NX; i++) buf_yx[i] = i;

    start[0] = 0;           start[1] = NX * rank;
    count[0] = NY * nprocs; count[1] = NX;
    err = ncmpi_put_vara_float_all(ncid, varid1, start, count, buf_yx); ERR

    if (rank == 0 && verbose) print_hints(ncid, varid0, varid1);

    err = ncmpi_close(ncid); ERR

    free(buf_zy);
    free(buf_yx);

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


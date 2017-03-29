/*********************************************************************
 *
 *  Copyright (C) 2014, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *********************************************************************/
/* $Id: varn_int.c 2219 2015-12-11 22:30:03Z wkliao $ */

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * This example tests using a single call of ncmpi_put_varn_int_all() to
 * write a sequence of requests with arbitrary array indices and lengths.
 *
 * The compile and run commands are given below, together with an ncmpidump of
 * the output file.
 *
 *    % mpicc -O2 -o varn_int varn_int.c -lpnetcdf
 *    % mpiexec -n 4 ./varn_int /pvfs2/wkliao/testfile.nc
 *    % ncmpidump /pvfs2/wkliao/testfile.nc
 *    netcdf testfile {
 *    // file format: CDF-5 (big variables)
 *    dimensions:
 *             Y = 4 ;
 *             X = 10 ;
 *             REC_DIM = UNLIMITED ; // (4 currently)
 *    variables:
 *             int var(Y, X) ;
 *             int rec_var(REC_DIM, X) ;
 *    data:
 *
 *     var =
 *       3, 3, 3, 1, 1, 0, 0, 2, 1, 1,
 *       0, 2, 2, 2, 3, 1, 1, 2, 2, 2,
 *       1, 1, 2, 3, 3, 3, 0, 0, 1, 1,
 *       0, 0, 0, 2, 1, 1, 1, 3, 3, 3 ;
 *
 *     rec_var =
 *       3, 3, 3, 1, 1, 0, 0, 2, 1, 1,
 *       0, 2, 2, 2, 3, 1, 1, 2, 2, 2,
 *       1, 1, 2, 3, 3, 3, 0, 0, 1, 1,
 *       0, 0, 0, 2, 1, 1, 1, 3, 3, 3 ;
 *    }
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <string.h> /* strcpy(), memset() */
#include <mpi.h>
#include <pnetcdf.h>

#include <testutils.h>

#define NY 4
#define NX 10
#define NDIMS 2

#define ERR {if(err!=NC_NOERR){printf("Error at %s line=%d: %s\n", __FILE__,__LINE__, ncmpi_strerror(err)); nerrs++;}}

static
int check_contents_for_fail(int *buffer)
{
    int i, nprocs;
    int expected[NY*NX] = {3, 3, 3, 1, 1, 0, 0, 2, 1, 1,
                           0, 2, 2, 2, 3, 1, 1, 2, 2, 2,
                           1, 1, 2, 3, 3, 3, 0, 0, 1, 1,
                           0, 0, 0, 2, 1, 1, 1, 3, 3, 3};

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /* check if the contents of buf are expected */
    for (i=0; i<NY*NX; i++) {
        if (expected[i] >= nprocs) continue;
        if (buffer[i] != expected[i]) {
            printf("Expected read buf[%d]=%d, but got %d\n",
                   i,expected[i],buffer[i]);
            return 1;
        }
    }
    return 0;
}

static
void permute(MPI_Offset a[NDIMS], MPI_Offset b[NDIMS])
{
    int i;
    MPI_Offset tmp;
    for (i=0; i<NDIMS; i++) {
        tmp = a[i]; a[i] = b[i]; b[i] = tmp;
    }
}

int main(int argc, char** argv)
{
    char filename[256];
    int i, j, rank, nprocs, err, verbose=0, nerrs=0;
    int ncid, cmode, varid[3], dimid[2], num_reqs, *buffer, *r_buffer;
    MPI_Offset w_len, **starts, **counts;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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
        sprintf(cmd_str, "*** TESTING C   %s for ncmpi_put_varn_int_all() ", argv[0]);
        printf("%-66s ------ ", cmd_str); fflush(stdout);
    }

    if (verbose && nprocs != 4 && rank == 0)
        printf("Warning: %s is intended to run on 4 processes\n",argv[0]);

    /* create a new file for writing ----------------------------------------*/
    cmode = NC_CLOBBER | NC_64BIT_DATA;
    err = ncmpi_create(MPI_COMM_WORLD, filename, cmode, MPI_INFO_NULL, &ncid);
    ERR

    /* create a global array of size NY * NX */
    err = ncmpi_def_dim(ncid, "Y", NY, &dimid[0]);
    ERR
    err = ncmpi_def_dim(ncid, "X", NX, &dimid[1]);
    ERR
    err = ncmpi_def_var(ncid, "var", NC_INT, NDIMS, dimid, &varid[0]);
    ERR
    err = ncmpi_def_dim(ncid, "REC_DIM", NC_UNLIMITED, &dimid[0]);
    ERR
    err = ncmpi_def_var(ncid, "rec_var", NC_INT, NDIMS, dimid, &varid[1]);
    ERR
    err = ncmpi_enddef(ncid);
    ERR

    /* pick arbitrary numbers of requests for 4 processes */
    num_reqs = 0;
    if (rank == 0)      num_reqs = 4;
    else if (rank == 1) num_reqs = 6;
    else if (rank == 2) num_reqs = 5;
    else if (rank == 3) num_reqs = 4;

    starts    = (MPI_Offset**) malloc(num_reqs *        sizeof(MPI_Offset*));
    counts    = (MPI_Offset**) malloc(num_reqs *        sizeof(MPI_Offset*));
    starts[0] = (MPI_Offset*)  calloc(num_reqs * NDIMS, sizeof(MPI_Offset));
    counts[0] = (MPI_Offset*)  calloc(num_reqs * NDIMS, sizeof(MPI_Offset));
    for (i=1; i<num_reqs; i++) {
        starts[i] = starts[i-1] + NDIMS;
        counts[i] = counts[i-1] + NDIMS;
    }

    /* assign arbitrary starts and counts */
    const int y=0, x=1;
    if (rank == 0) {
        starts[0][y] = 0; starts[0][x] = 5; counts[0][y] = 1; counts[0][x] = 2;
        starts[1][y] = 1; starts[1][x] = 0; counts[1][y] = 1; counts[1][x] = 1;
        starts[2][y] = 2; starts[2][x] = 6; counts[2][y] = 1; counts[2][x] = 2;
        starts[3][y] = 3; starts[3][x] = 0; counts[3][y] = 1; counts[3][x] = 3;
        /* rank 0 is writing the followings: ("-" means skip)
                  -  -  -  -  -  0  0  -  -  - 
                  0  -  -  -  -  -  -  -  -  - 
                  -  -  -  -  -  -  0  0  -  - 
                  0  0  0  -  -  -  -  -  -  - 
         */
    } else if (rank ==1) {
        starts[0][y] = 0; starts[0][x] = 3; counts[0][y] = 1; counts[0][x] = 2;
        starts[1][y] = 0; starts[1][x] = 8; counts[1][y] = 1; counts[1][x] = 2;
        starts[2][y] = 1; starts[2][x] = 5; counts[2][y] = 1; counts[2][x] = 2;
        starts[3][y] = 2; starts[3][x] = 0; counts[3][y] = 1; counts[3][x] = 2;
        starts[4][y] = 2; starts[4][x] = 8; counts[4][y] = 1; counts[4][x] = 2;
        starts[5][y] = 3; starts[5][x] = 4; counts[5][y] = 1; counts[5][x] = 3;
        /* rank 1 is writing the followings: ("-" means skip)
                  -  -  -  1  1  -  -  -  1  1 
                  -  -  -  -  -  1  1  -  -  - 
                  1  1  -  -  -  -  -  -  1  1 
                  -  -  -  -  1  1  1  -  -  - 
         */
    } else if (rank ==2) {
        starts[0][y] = 0; starts[0][x] = 7; counts[0][y] = 1; counts[0][x] = 1;
        starts[1][y] = 1; starts[1][x] = 1; counts[1][y] = 1; counts[1][x] = 3;
        starts[2][y] = 1; starts[2][x] = 7; counts[2][y] = 1; counts[2][x] = 3;
        starts[3][y] = 2; starts[3][x] = 2; counts[3][y] = 1; counts[3][x] = 1;
        starts[4][y] = 3; starts[4][x] = 3; counts[4][y] = 1; counts[4][x] = 1;
        /* rank 2 is writing the followings: ("-" means skip)
                  -  -  -  -  -  -  -  2  -  - 
                  -  2  2  2  -  -  -  2  2  2 
                  -  -  2  -  -  -  -  -  -  - 
                  -  -  -  2  -  -  -  -  -  - 
         */
    } else if (rank ==3) {
        starts[0][y] = 0; starts[0][x] = 0; counts[0][y] = 1; counts[0][x] = 3;
        starts[1][y] = 1; starts[1][x] = 4; counts[1][y] = 1; counts[1][x] = 1;
        starts[2][y] = 2; starts[2][x] = 3; counts[2][y] = 1; counts[2][x] = 3;
        starts[3][y] = 3; starts[3][x] = 7; counts[3][y] = 1; counts[3][x] = 3;
        /* rank 3 is writing the followings: ("-" means skip)
                  3  3  3  -  -  -  -  -  -  - 
                  -  -  -  -  3  -  -  -  -  - 
                  -  -  -  3  3  3  -  -  -  - 
                  -  -  -  -  -  -  -  3  3  3 
         */
    }

    w_len = 0; /* total write length for this process */
    for (i=0; i<num_reqs; i++) {
        MPI_Offset w_req_len=1;
        for (j=0; j<NDIMS; j++)
            w_req_len *= counts[i][j];
        w_len += w_req_len;
    }

    /* allocate I/O buffer and initialize its contents */
    r_buffer = (int*) malloc(NY*NX * sizeof(int));
    buffer   = (int*) malloc(w_len * sizeof(int));
    for (i=0; i<w_len; i++) buffer[i] = rank;

    /* check error code: NC_ENULLSTART */
    err = ncmpi_put_varn_int_all(ncid, varid[0], 1, NULL, NULL, NULL);
    if (err != NC_ENULLSTART) {
        printf("expecting error code NC_ENULLSTART but got %s\n",nc_err_code_name(err));
        nerrs++;
    }

    /* write using varn API */
    err = ncmpi_put_varn_int_all(ncid, varid[0], num_reqs, starts, counts, buffer);
    ERR

    if (nprocs > 4) MPI_Barrier(MPI_COMM_WORLD);

    /* read back and check contents */
    memset(r_buffer, 0, NY*NX*sizeof(int));
    err = ncmpi_get_var_int_all(ncid, varid[0], r_buffer);
    ERR
    nerrs += check_contents_for_fail(r_buffer);

    /* permute write order */
    if (num_reqs > 0) {
        permute(starts[1], starts[2]); permute(counts[1], counts[2]);
        permute(starts[2], starts[3]); permute(counts[2], counts[3]);
    }

    /* write using varn API */
    err = ncmpi_put_varn_int_all(ncid, varid[1], num_reqs, starts, counts, buffer);
    ERR

    /* read back using get_var API and check contents */
    memset(r_buffer, 0, NY*NX*sizeof(int));
    err = ncmpi_get_var_int_all(ncid, varid[1], r_buffer);
    ERR
    nerrs += check_contents_for_fail(r_buffer);

    /* read back using get_varn API and check contents */
    for (i=0; i<w_len; i++) buffer[i] = -1;
    err = ncmpi_get_varn_int_all(ncid, varid[0], num_reqs, starts, counts, buffer);
    ERR

    for (i=0; i<w_len; i++) {
        if (buffer[i] != rank) {
            printf("Error at line %d: expecting buffer[%d]=%d but got %d\n",
                   __LINE__,i,rank,buffer[i]);
            nerrs++;
        }
    }

    /* test flexible API, using a noncontiguous buftype */
    MPI_Datatype buftype;
    MPI_Type_vector(w_len, 1, 2, MPI_INT, &buftype);
    MPI_Type_commit(&buftype);
    free(buffer);
    buffer = (int*) malloc(w_len * 2 * sizeof(int));
    for (i=0; i<2*w_len; i++) buffer[i] = -1;
    err = ncmpi_get_varn_all(ncid, varid[0], num_reqs, starts, counts, buffer, 1, buftype);
    ERR
    MPI_Type_free(&buftype);

    for (i=0; i<w_len*2; i++) {
        if (i%2 && buffer[i] != -1) {
            printf("Error at line %d: expecting buffer[%d]=-1 but got %d\n",
                   __LINE__,i,buffer[i]);
            nerrs++;
        }
        if (i%2 == 0 && buffer[i] != rank) {
            printf("Error at line %d: expecting buffer[%d]=%d but got %d\n",
                   __LINE__,i,rank,buffer[i]);
            nerrs++;
        }
    }

    err = ncmpi_close(ncid);
    ERR

    free(buffer);
    free(r_buffer);
    free(starts[0]);
    free(counts[0]);
    free(starts);
    free(counts);

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


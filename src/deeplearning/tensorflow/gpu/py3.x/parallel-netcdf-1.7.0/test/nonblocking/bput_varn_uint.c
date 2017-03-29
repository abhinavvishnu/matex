/*********************************************************************
 *
 *  Copyright (C) 2014, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *********************************************************************/
/* $Id: bput_varn_uint.c 2219 2015-12-11 22:30:03Z wkliao $ */

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * This example tests nonblocking buffered write varn APIs, including
 * ncmpi_bput_varn_uint() and ncmpi_bput_varn(),
 * It first writes a sequence of requests with arbitrary array indices and
 * lengths to four variables of type NC_UINT, and reads back.
 *
 * The compile and run commands are given below, together with an ncmpidump of
 * the output file.
 *
 *    % mpicc -O2 -o bput_varn_uint bput_varn_uint.c -lpnetcdf
 *    % mpiexec -n 4 ./bput_varn_uint /pvfs2/wkliao/testfile.nc
 *    % ncmpidump /pvfs2/wkliao/testfile.nc
 *    netcdf testfile {
 *    // file format: CDF-5 (big variables)
 *    dimensions:
 *             Y = 4 ;
 *             X = 10 ;
 *    variables:
 *            uint var0(Y, X) ;
 *            uint var1(Y, X) ;
 *            uint var2(Y, X) ;
 *            uint var3(Y, X) ;
 *    data:
 *
 *     var0 =
 *      3, 3, 3, 1, 1, 0, 0, 2, 1, 1,
 *      0, 2, 2, 2, 3, 1, 1, 2, 2, 2,
 *      1, 1, 2, 3, 3, 3, 0, 0, 1, 1,
 *      0, 0, 0, 2, 1, 1, 1, 3, 3, 3 ;
 *
 *     var1 =
 *      2, 2, 2, 0, 0, 3, 3, 1, 0, 0,
 *      3, 1, 1, 1, 2, 0, 0, 1, 1, 1,
 *      0, 0, 1, 2, 2, 2, 3, 3, 0, 0,
 *      3, 3, 3, 1, 0, 0, 0, 2, 2, 2 ;
 *
 *     var2 =
 *      1, 1, 1, 3, 3, 2, 2, 0, 3, 3,
 *      2, 0, 0, 0, 1, 3, 3, 0, 0, 0,
 *      3, 3, 0, 1, 1, 1, 2, 2, 3, 3,
 *      2, 2, 2, 0, 3, 3, 3, 1, 1, 1 ;
 *
 *     var3 =
 *      0, 0, 0, 2, 2, 1, 1, 3, 2, 2,
 *      1, 3, 3, 3, 0, 2, 2, 3, 3, 3,
 *      2, 2, 3, 0, 0, 0, 1, 1, 2, 2,
 *      1, 1, 1, 3, 2, 2, 2, 0, 0, 0 ;
 *    }
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <string.h> /* strcpy() */
#include <mpi.h>
#include <pnetcdf.h>

#include <testutils.h>

#define NY 4
#define NX 10
#define NDIMS 2

#define ERR \
    if (err != NC_NOERR) { \
        printf("Error at line=%d: %s\n", __LINE__, ncmpi_strerror(err)); \
        nerrs++; \
    }

#define ERRS(n,a) { \
    int _i; \
    for (_i=0; _i<(n); _i++) { \
        if ((a)[_i] != NC_NOERR) { \
            printf("Error at line=%d: err[%d] %s\n", __LINE__, _i, \
                   ncmpi_strerror((a)[_i])); \
            nerrs++; \
        } \
    } \
}

static
void clear_file_contents(int ncid, int *varid)
{
    int i, err, rank;
    unsigned int *w_buffer = (unsigned int*) malloc(NY*NX * sizeof(unsigned int));
    for (i=0; i<NY*NX; i++) w_buffer[i] = 99999;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for (i=0; i<4; i++) {
        err = ncmpi_put_var_uint_all(ncid, varid[i], w_buffer);
        if (err != NC_NOERR) printf("Error at line=%d: %s\n", __LINE__, ncmpi_strerror(err));
    }
    free(w_buffer);
}

static
int check_contents_for_fail(int ncid, int *varid)
{
    /* all processes read entire variables back and check contents */
    int i, j, err, nprocs;
    unsigned int expected[4][NY*NX] = {{3, 3, 3, 1, 1, 0, 0, 2, 1, 1,
                                        0, 2, 2, 2, 3, 1, 1, 2, 2, 2,
                                        1, 1, 2, 3, 3, 3, 0, 0, 1, 1,
                                        0, 0, 0, 2, 1, 1, 1, 3, 3, 3},
                                       {2, 2, 2, 0, 0, 3, 3, 1, 0, 0,
                                        3, 1, 1, 1, 2, 0, 0, 1, 1, 1,
                                        0, 0, 1, 2, 2, 2, 3, 3, 0, 0,
                                        3, 3, 3, 1, 0, 0, 0, 2, 2, 2},
                                       {1, 1, 1, 3, 3, 2, 2, 0, 3, 3,
                                        2, 0, 0, 0, 1, 3, 3, 0, 0, 0,
                                        3, 3, 0, 1, 1, 1, 2, 2, 3, 3,
                                        2, 2, 2, 0, 3, 3, 3, 1, 1, 1},
                                       {0, 0, 0, 2, 2, 1, 1, 3, 2, 2,
                                        1, 3, 3, 3, 0, 2, 2, 3, 3, 3,
                                        2, 2, 3, 0, 0, 0, 1, 1, 2, 2,
                                        1, 1, 1, 3, 2, 2, 2, 0, 0, 0}};

    unsigned int *r_buffer = (unsigned int*) malloc(NY*NX * sizeof(unsigned int));

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    if (nprocs > 4) MPI_Barrier(MPI_COMM_WORLD);

    for (i=0; i<4; i++) {
        for (j=0; j<NY*NX; j++) r_buffer[j] = 99999;
        err = ncmpi_get_var_uint_all(ncid, varid[i], r_buffer);
        if (err != NC_NOERR) printf("Error at line=%d: %s\n", __LINE__, ncmpi_strerror(err));

        /* check if the contents of buf are expected */
        for (j=0; j<NY*NX; j++) {
            if (expected[i][j] >= nprocs) continue;
            if (r_buffer[j] != expected[i][j]) {
                printf("Expected read buf[%d][%d]=%u, but got %u\n",
                       i,j,expected[i][j],r_buffer[j]);
                free(r_buffer);
                return 1;
            }
        }
    }
    free(r_buffer);
    return 0;
}

static int
check_num_pending_reqs(int ncid, int expected, int lineno)
/* check if PnetCDF can reports expected number of pending requests */
{
    int err, n_pendings;
    err = ncmpi_inq_nreqs(ncid, &n_pendings);
    if (err != NC_NOERR) printf("Error at line=%d: %s\n", __LINE__, ncmpi_strerror(err));
    if (n_pendings != expected) {
        printf("Error at line %d: expect %d pending requests but got %d\n",
               lineno, expected, n_pendings);
        return 1;
    }
    return 0;
}

static
void check_attached_buffer_usage(int ncid,
                                 MPI_Offset expected_size,
                                 MPI_Offset expected_usage,
                                 int lineno)
/* check attached buf usage */
{
    int err, rank;
    MPI_Offset usage, buf_size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank >= 4) return;

    err = ncmpi_inq_buffer_size(ncid, &buf_size);
    if (err != NC_NOERR) printf("Error at line=%d: %s\n", __LINE__, ncmpi_strerror(err));
    if (expected_size != buf_size)
        printf("Error at line %d: expect buffer size %lld but got %lld\n",
               lineno, expected_size, buf_size);

    err = ncmpi_inq_buffer_usage(ncid, &usage);
    if (err != NC_NOERR) printf("Error at line=%d: %s\n", __LINE__, ncmpi_strerror(err));
    if (expected_usage != usage)
        printf("Error at line %d: expect buffer usage %lld but got %lld\n",
               lineno, expected_usage, usage);
}

/* swap two rows, a and b, of a 2D array */
static
void permute(MPI_Offset *a, MPI_Offset *b)
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
    int i, j, k, rank, nprocs, verbose=0, err, nerrs=0;
    int ncid, cmode, varid[4], dimid[2], nreqs, reqs[4], sts[4];
    unsigned int *buffer[4];
    int num_segs[4] = {4, 6, 5, 4};
    int req_lens[4], my_nsegs[4];
    MPI_Offset **starts[4], **counts[4];
    MPI_Offset n_starts[4][6][2] = {{{0,5}, {1,0}, {2,6}, {3,0}, {0,0}, {0,0}},
                                    {{0,3}, {0,8}, {1,5}, {2,0}, {2,8}, {3,4}},
                                    {{0,7}, {1,1}, {1,7}, {2,2}, {3,3}, {0,0}},
                                    {{0,0}, {1,4}, {2,3}, {3,7}, {0,0}, {0,0}}};
    MPI_Offset n_counts[4][6][2] = {{{1,2}, {1,1}, {1,2}, {1,3}, {0,0}, {0,0}},
                                    {{1,2}, {1,2}, {1,2}, {1,2}, {1,2}, {1,3}},
                                    {{1,1}, {1,3}, {1,3}, {1,1}, {1,1}, {0,0}},
                                    {{1,3}, {1,1}, {1,3}, {1,3}, {0,0}, {0,0}}};

    /* n_starts[0][][] n_counts[0][][] indicate the following: ("-" means skip)
              -  -  -  -  -  X  X  -  -  - 
              X  -  -  -  -  -  -  -  -  - 
              -  -  -  -  -  -  X  X  -  - 
              X  X  X  -  -  -  -  -  -  - 
       n_starts[1][][] n_counts[1][][] indicate the following pattern.
              -  -  -  X  X  -  -  -  X  X 
              -  -  -  -  -  X  X  -  -  - 
              X  X  -  -  -  -  -  -  X  X 
              -  -  -  -  X  X  X  -  -  - 
       n_starts[2][][] n_counts[2][][] indicate the following pattern.
              -  -  -  -  -  -  -  X  -  - 
              -  X  X  X  -  -  -  X  X  X 
              -  -  X  -  -  -  -  -  -  - 
              -  -  -  X  -  -  -  -  -  - 
       n_starts[3][][] n_counts[3][][] indicate the following pattern.
              X  X  X  -  -  -  -  -  -  - 
              -  -  -  -  X  -  -  -  -  - 
              -  -  -  X  X  X  -  -  -  - 
              -  -  -  -  -  -  -  X  X  X 
     */
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
        sprintf(cmd_str, "*** TESTING C   %s for bput_varn_uint ", argv[0]);
        printf("%-66s ------ ", cmd_str);
    }

    if (verbose && nprocs != 4 && rank == 0)
        printf("Warning: %s is intended to run on 4 processes\n",argv[0]);

    /* create a new file for writing ----------------------------------------*/
    cmode = NC_CLOBBER | NC_64BIT_DATA;
    err = ncmpi_create(MPI_COMM_WORLD, filename, cmode, MPI_INFO_NULL, &ncid);
    ERR

    /* create a global array of size NY * NX */
    err = ncmpi_def_dim(ncid, "Y", NY, &dimid[0]); ERR
    err = ncmpi_def_dim(ncid, "X", NX, &dimid[1]); ERR
    err = ncmpi_def_var(ncid, "var0", NC_UINT, NDIMS, dimid, &varid[0]); ERR
    err = ncmpi_def_var(ncid, "var1", NC_UINT, NDIMS, dimid, &varid[1]); ERR
    err = ncmpi_def_var(ncid, "var2", NC_UINT, NDIMS, dimid, &varid[2]); ERR
    err = ncmpi_def_var(ncid, "var3", NC_UINT, NDIMS, dimid, &varid[3]); ERR
    err = ncmpi_enddef(ncid); ERR

    /* allocate space for starts and counts */
    starts[0] = (MPI_Offset**) malloc(4 * 6 * sizeof(MPI_Offset*));
    counts[0] = (MPI_Offset**) malloc(4 * 6 * sizeof(MPI_Offset*));
    starts[0][0] = (MPI_Offset*) calloc(4 * 6 * NDIMS, sizeof(MPI_Offset));
    counts[0][0] = (MPI_Offset*) calloc(4 * 6 * NDIMS, sizeof(MPI_Offset));
    for (i=1; i<4; i++) {
        starts[i] = starts[i-1] + 6;
        counts[i] = counts[i-1] + 6;
        starts[i][0] = starts[i-1][0] + 6 * NDIMS;
        counts[i][0] = counts[i-1][0] + 6 * NDIMS;
    }
    for (i=0; i<4; i++) {
        for (j=1; j<6; j++) {
            starts[i][j] = starts[i][j-1] + NDIMS;
            counts[i][j] = counts[i][j-1] + NDIMS;
        }
    }

    /* set values for starts and counts */
    for (i=0; i<4; i++) {
        int n = (i + rank) % 4;
        my_nsegs[i] = num_segs[n]; /* number of segments for this request */
        for (j=0; j<6; j++) {
            for (k=0; k<NDIMS; k++) {
                starts[i][j][k] = n_starts[n][j][k];
                counts[i][j][k] = n_counts[n][j][k];
            }
        }
    }

    /* test error code: NC_ENULLABUF */
    err = ncmpi_bput_varn_uint(ncid, varid[0], 1, NULL, NULL, NULL, &reqs[0]);
    if (err != NC_ENULLABUF) {
        printf("Error at line %d: expecting error code NC_ENULLABUF but got %s\n",
               __LINE__, nc_err_code_name(err));
        nerrs++;
    }

    /* only rank 0, 1, 2, and 3 do I/O:
     * each of ranks 0 to 3 write 4 nonblocking requests */
    nreqs = 4;
    if (rank >= 4) nreqs = 0;

    /* bufsize must be max of data type converted before and after */
    MPI_Offset bufsize = 0;

    /* calculate length of each varn request and allocate write buffer */
    for (i=0; i<nreqs; i++) {
        req_lens[i] = 0; /* total length this request */
        for (j=0; j<my_nsegs[i]; j++) {
            MPI_Offset req_len=1;
            for (k=0; k<NDIMS; k++)
                req_len *= counts[i][j][k];
            req_lens[i] += req_len;
        }
        if (verbose) printf("req_lens[%d]=%d\n",i,req_lens[i]);

        /* allocate I/O buffer and initialize its contents */
        buffer[i] = (unsigned int*) malloc(req_lens[i] * sizeof(unsigned int));
        for (j=0; j<req_lens[i]; j++) buffer[i][j] = rank;
        bufsize += req_lens[i];
    }
    bufsize *= sizeof(unsigned int);

    /* give PnetCDF a space to buffer the nonblocking requests */
    if (bufsize > 0) {
        err = ncmpi_buffer_attach(ncid, bufsize); ERR
    }
    if (verbose) printf("%d: Attach buffer size %lld\n", rank, bufsize);

    /* test error code: NC_ENULLSTART */
    err = ncmpi_bput_varn_uint(ncid, varid[0], 1, NULL, NULL, NULL, &reqs[0]);
    if (rank < 4 && err != NC_ENULLSTART) {
        printf("Error at line %d: expecting error code NC_ENULLSTART but got %s\n",
               __LINE__, nc_err_code_name(err));
        nerrs++;
    }

    /* write usning varn API, one bput call per variable */
    clear_file_contents(ncid, varid);
    for (i=0; i<nreqs; i++) {
        err = ncmpi_bput_varn_uint(ncid, varid[i], my_nsegs[i], starts[i],
                                   counts[i], buffer[i], &reqs[i]);
        ERR
    }
    nerrs += check_num_pending_reqs(ncid, nreqs, __LINE__);
    check_attached_buffer_usage(ncid, bufsize, bufsize, __LINE__);
    err = ncmpi_wait_all(ncid, nreqs, reqs, sts);
    ERRS(nreqs, sts)

    check_attached_buffer_usage(ncid, bufsize, 0, __LINE__);

    /* all processes read entire variables back and check contents */
    nerrs += check_contents_for_fail(ncid, varid);

    /* permute write order: so starts[*] are not in an increasing order:
     * swap segment 0 with segment 2 and swap segment 1 with segment 3
     */
    for (i=0; i<nreqs; i++) {
        permute(starts[i][0],starts[i][2]); permute(counts[i][0],counts[i][2]);
        permute(starts[i][1],starts[i][3]); permute(counts[i][1],counts[i][3]);
    }

    /* write using varn API, one bput call per variable */
    clear_file_contents(ncid, varid);
    for (i=0; i<nreqs; i++) {
        err = ncmpi_bput_varn_uint(ncid, varid[i], my_nsegs[i], starts[i],
                                   counts[i], buffer[i], &reqs[i]);
        ERR
    }
    nerrs += check_num_pending_reqs(ncid, nreqs, __LINE__);
    check_attached_buffer_usage(ncid, bufsize, bufsize, __LINE__);
    err = ncmpi_wait_all(ncid, nreqs, reqs, sts);
    ERRS(nreqs, sts)

    check_attached_buffer_usage(ncid, bufsize, 0, __LINE__);

    /* all processes read entire variables back and check contents */
    nerrs += check_contents_for_fail(ncid, varid);

    for (i=0; i<nreqs; i++) free(buffer[i]);

    /* test flexible API, using a noncontiguous buftype */
    clear_file_contents(ncid, varid);
    for (i=0; i<nreqs; i++) {
        MPI_Datatype buftype;
        MPI_Type_vector(req_lens[i], 1, 2, MPI_UNSIGNED, &buftype);
        MPI_Type_commit(&buftype);
        buffer[i] = (unsigned int*)malloc(req_lens[i]*2*sizeof(unsigned int));
        for (j=0; j<req_lens[i]*2; j++) buffer[i][j] = rank;

        err = ncmpi_bput_varn(ncid, varid[i], my_nsegs[i], starts[i],
                              counts[i], buffer[i], 1, buftype, &reqs[i]);
        ERR
        MPI_Type_free(&buftype);
    }
    nerrs += check_num_pending_reqs(ncid, nreqs, __LINE__);
    check_attached_buffer_usage(ncid, bufsize, bufsize, __LINE__);
    err = ncmpi_wait_all(ncid, nreqs, reqs, sts);
    ERRS(nreqs, sts)

    check_attached_buffer_usage(ncid, bufsize, 0, __LINE__);

    /* all processes read entire variables back and check contents */
    nerrs += check_contents_for_fail(ncid, varid);

    /* permute back to original order */
    for (i=0; i<nreqs; i++) {
        permute(starts[i][0],starts[i][2]); permute(counts[i][0],counts[i][2]);
        permute(starts[i][1],starts[i][3]); permute(counts[i][1],counts[i][3]);
    }

    /* test flexible API, using a noncontiguous buftype, one bput call per
     * variable */
    clear_file_contents(ncid, varid);
    for (i=0; i<nreqs; i++) {
        MPI_Datatype buftype;
        MPI_Type_vector(req_lens[i], 1, 2, MPI_UNSIGNED, &buftype);
        MPI_Type_commit(&buftype);
        for (j=0; j<req_lens[i]*2; j++) buffer[i][j] = rank;

        err = ncmpi_bput_varn(ncid, varid[i], my_nsegs[i], starts[i],
                              counts[i], buffer[i], 1, buftype, &reqs[i]);
        ERR
        MPI_Type_free(&buftype);
    }
    nerrs += check_num_pending_reqs(ncid, nreqs, __LINE__);
    check_attached_buffer_usage(ncid, bufsize, bufsize, __LINE__);
    err = ncmpi_wait_all(ncid, nreqs, reqs, sts);
    ERRS(nreqs, sts)

    check_attached_buffer_usage(ncid, bufsize, 0, __LINE__);

    /* all processes read entire variables back and check contents */
    nerrs += check_contents_for_fail(ncid, varid);

    /* free the buffer space for bput */
    if (bufsize > 0) {
        err = ncmpi_buffer_detach(ncid); ERR
    }

    /* test error code: NC_ENULLABUF */
    err = ncmpi_inq_buffer_usage(ncid, NULL);
    if (err != NC_ENULLABUF) {
        printf("expecting error code NC_ENULLABUF but got %s\n",
               nc_err_code_name(err));
        nerrs++;
    }

    err = ncmpi_close(ncid); ERR

    for (i=0; i<nreqs; i++) free(buffer[i]);
    free(starts[0][0]);
    free(counts[0][0]);
    free(starts[0]);
    free(counts[0]);

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


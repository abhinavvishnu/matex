/*********************************************************************
 *
 *  Copyright (C) 2013, Northwestern University
 *  See COPYRIGHT notice in top-level directory.
 *
 *********************************************************************/
/* $Id: aggregation.c 2325 2016-02-28 07:49:13Z wkliao $ */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>
#include <pnetcdf.h>

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * This program writes a series of 2D variables with data partitioning patterns
 * of block-block, *-cyclic, block-*, and *-block, round-robinly. The block-*
 * partitioning case writes 1st half followed by 2nd half. The same partitioning
 * patterns are used for read. In both cases, nonblocking APIs are used to
 * evaluate the performance.
 * 
 * The compile and run commands are given below, together with an ncmpidump of
 * the output file. In this example, NVARS = 5.
 *
 *    % mpicc -O2 -o aggregation aggregation.c -lpnetcdf
 *
 *    % mpiexec -l -n 4 ./aggregation 5 /pvfs2/wkliao/testfile.nc
 *
 *    % ncmpidump /pvfs2/wkliao/testfile.nc
 *      netcdf testfile {
 *      // file format: CDF-5 (big variables)
 *      dimensions:
 *              Block_BLOCK_Y = 10 ;
 *              Block_BLOCK_X = 10 ;
 *              Star_Y = 5 ;
 *              Cyclic_X = 20 ;
 *              Block_Y = 20 ;
 *              Star_X = 5 ;
 *      variables:
 *              int block_block_var_0(Block_BLOCK_Y, Block_BLOCK_X) ;
 *              float star_cyclic_var_1(Star_Y, Cyclic_X) ;
 *              short block_star_var_2(Block_Y, Star_X) ;
 *              double star_block_var_3(Star_Y, Cyclic_X) ;
 *              int block_block_var_4(Block_BLOCK_Y, Block_BLOCK_X) ;
 *      data:
 *
 *       block_block_var_0 =
 *        0, 0, 0, 0, 0, 2, 2, 2, 2, 2,
 *        0, 0, 0, 0, 0, 2, 2, 2, 2, 2,
 *        0, 0, 0, 0, 0, 2, 2, 2, 2, 2,
 *        0, 0, 0, 0, 0, 2, 2, 2, 2, 2,
 *        0, 0, 0, 0, 0, 2, 2, 2, 2, 2,
 *        1, 1, 1, 1, 1, 3, 3, 3, 3, 3,
 *        1, 1, 1, 1, 1, 3, 3, 3, 3, 3,
 *        1, 1, 1, 1, 1, 3, 3, 3, 3, 3,
 *        1, 1, 1, 1, 1, 3, 3, 3, 3, 3,
 *        1, 1, 1, 1, 1, 3, 3, 3, 3, 3 ;
 *
 *       star_cyclic_var_1 =
 *        0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
 *        0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
 *        0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
 *        0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
 *        0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3 ;
 *
 *       block_star_var_2 =
 *        0, 0, 0, 0, 0,
 *        0, 0, 0, 0, 0,
 *        0, 0, 0, 0, 0,
 *        0, 0, 0, 0, 0,
 *        0, 0, 0, 0, 0,
 *        1, 1, 1, 1, 1,
 *        1, 1, 1, 1, 1,
 *        1, 1, 1, 1, 1,
 *        1, 1, 1, 1, 1,
 *        1, 1, 1, 1, 1,
 *        2, 2, 2, 2, 2,
 *        2, 2, 2, 2, 2,
 *        2, 2, 2, 2, 2,
 *        2, 2, 2, 2, 2,
 *        2, 2, 2, 2, 2,
 *        3, 3, 3, 3, 3,
 *        3, 3, 3, 3, 3,
 *        3, 3, 3, 3, 3,
 *        3, 3, 3, 3, 3,
 *        3, 3, 3, 3, 3 ;
 *
 *       star_block_var_3 =
 *        0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
 *        0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
 *        0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
 *        0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
 *        0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3 ;
 *
 *       block_block_var_4 =
 *        0, 0, 0, 0, 0, 2, 2, 2, 2, 2,
 *        0, 0, 0, 0, 0, 2, 2, 2, 2, 2,
 *        0, 0, 0, 0, 0, 2, 2, 2, 2, 2,
 *        0, 0, 0, 0, 0, 2, 2, 2, 2, 2,
 *        0, 0, 0, 0, 0, 2, 2, 2, 2, 2,
 *        1, 1, 1, 1, 1, 3, 3, 3, 3, 3,
 *        1, 1, 1, 1, 1, 3, 3, 3, 3, 3,
 *        1, 1, 1, 1, 1, 3, 3, 3, 3, 3,
 *        1, 1, 1, 1, 1, 3, 3, 3, 3, 3,
 *        1, 1, 1, 1, 1, 3, 3, 3, 3, 3 ;
 *      }
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#define NVARS 5

#define ERR(e) {if((e)!=NC_NOERR)printf("Error at line=%d: %s\n", __LINE__, ncmpi_strerror(e));}

/*----< print_info() >------------------------------------------------------*/
static
void print_info(MPI_Info *info_used)
{
    int  i, nkeys;

    MPI_Info_get_nkeys(*info_used, &nkeys);
    printf("MPI File Info: nkeys = %d\n",nkeys);
    for (i=0; i<nkeys; i++) {
        char key[MPI_MAX_INFO_KEY], value[MPI_MAX_INFO_VAL];
        int  valuelen, flag;

        MPI_Info_get_nthkey(*info_used, i, key);
        MPI_Info_get_valuelen(*info_used, key, &valuelen, &flag);
        MPI_Info_get(*info_used, key, valuelen+1, value, &flag);
        printf("MPI File Info: [%2d] key = %24s, value = %s\n",i,key,value);
    }
}

/*----< benchmark_write() >---------------------------------------------------*/
static
int benchmark_write(char       *filename,
                    MPI_Offset  len,
                    MPI_Offset *w_size,
                    MPI_Info   *w_info_used,
                    double     *timing)  /* [6] */
{
    int i, j, k, verbose, rank, nprocs, err, num_reqs;
    int ncid, cmode, varid[NVARS], dimid[6], *reqs, *sts, psizes[2];
    void *buf[NVARS];
    double start_t, end_t;
    MPI_Comm comm=MPI_COMM_WORLD;
    MPI_Offset gsizes[2], start[2], count[2];
    MPI_Info info=MPI_INFO_NULL;

    verbose = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    /* set PnetCDF I/O hints */
    MPI_Info_create(&info);
    /* disable the header extent alignments
    MPI_Info_set(info, "nc_header_align_size", "1");    size in bytes
    */
    /* disable the fixed-size variable alignments */
    MPI_Info_set(info, "nc_var_align_size", "1");

    /* initialize I/O buffer with random numbers */
    srand(rank);
    for (i=0; i<NVARS; i++) {
        if (i % 4 == 0) {
            int *int_b = (int*) malloc(len * len * sizeof(int));
            for (j=0; j<len*len; j++) int_b[j] = rank; /* rand(); */
            buf[i] = (void*)int_b;
        }
        else if (i % 4 == 1) {
            float *flt_b = (float*) malloc(len * len * sizeof(float));
            for (j=0; j<len*len; j++) flt_b[j] = rank; /* rand(); */
            buf[i] = (void*)flt_b;
        }
        else if (i % 4 == 2) {
            short *shr_b = (short*) malloc(len * len * sizeof(short));
            for (j=0; j<len*len; j++) shr_b[j] = rank; /* rand(); */
            buf[i] = (void*)shr_b;
        }
        else {
            double *dbl_b = (double*) malloc(len * len * sizeof(double));
            for (j=0; j<len*len; j++) dbl_b[j] = rank; /* rand(); */
            buf[i] = (void*)dbl_b;
        }
    }
    MPI_Barrier(comm);
    timing[0] = MPI_Wtime();

    /* create a new file for writing -----------------------------------------*/
    cmode = NC_CLOBBER | NC_64BIT_DATA;
    err = ncmpi_create(comm, filename, cmode, info, &ncid); ERR(err)
    start_t = MPI_Wtime();
    timing[1] = start_t - timing[0];
    MPI_Info_free(&info);

    psizes[0] = psizes[1] = 0;
    MPI_Dims_create(nprocs, 2, psizes);

    gsizes[0] = len * psizes[0];
    gsizes[1] = len * psizes[1];

    err = ncmpi_def_dim(ncid, "Block_BLOCK_Y",  gsizes[0],  &dimid[0]); ERR(err)
    err = ncmpi_def_dim(ncid, "Block_BLOCK_X",  gsizes[1],  &dimid[1]); ERR(err)
    err = ncmpi_def_dim(ncid, "Star_Y",         len,        &dimid[2]); ERR(err)
    err = ncmpi_def_dim(ncid, "Cyclic_X",       len*nprocs, &dimid[3]); ERR(err)
    err = ncmpi_def_dim(ncid, "Block_Y",        len*nprocs, &dimid[4]); ERR(err)
    err = ncmpi_def_dim(ncid, "Star_X",         len,        &dimid[5]); ERR(err)

    /* define variables */
    num_reqs = 0;
    for (i=0; i<NVARS; i++) {
        char var_name[32];
        if (i % 4 == 0) {
            /* variables are block-block partitioned */
            sprintf(var_name,"block_block_var_%d",i);
            err = ncmpi_def_var(ncid, var_name, NC_INT, 2, dimid, &varid[i]);
            ERR(err)
            num_reqs++; /* complete in 1 nonblocking call */
        }
        else if (i % 4 == 1) {
            /* variables are *-cyclic partitioned */
            sprintf(var_name,"star_cyclic_var_%d",i);
            err = ncmpi_def_var(ncid, var_name, NC_FLOAT, 2, dimid+2, &varid[i]);
            ERR(err)
            num_reqs += len; /* complete in len nonblocking calls */
        }
        else if (i % 4 == 2) {
            /* variables are block-* partitioned */
            sprintf(var_name,"block_star_var_%d",i);
            err = ncmpi_def_var(ncid, var_name, NC_SHORT, 2, dimid+4, &varid[i]);
            ERR(err)
            num_reqs += 2; /* write 1st half followed by 2nd half */
        }
        else {
            /* variables are *-block partitioned */
            sprintf(var_name,"star_block_var_%d",i);
            err = ncmpi_def_var(ncid, var_name, NC_DOUBLE, 2, dimid+2, &varid[i]);
            ERR(err)
            num_reqs++; /* complete in 1 nonblocking call */
        }
    }
    reqs = (int*) malloc(num_reqs * sizeof(int));

    err = ncmpi_enddef(ncid); ERR(err)
    end_t = MPI_Wtime();
    timing[2] = end_t - start_t;
    start_t = end_t;

    k = 0;
    for (i=0; i<NVARS; i++) {
        if (i % 4 == 0) {
            int *int_b = (int*) buf[i];
            start[0] = len * (rank % psizes[0]);
            start[1] = len * ((rank / psizes[1]) % psizes[1]);
            count[0] = len;
            count[1] = len;
            err = ncmpi_iput_vara_int(ncid, varid[i], start, count, int_b,
                                      &reqs[k++]);
            ERR(err)
            if (verbose) printf("block-block %d: start=%lld %lld count=%lld %lld\n",i,start[0],start[1],count[0],count[1]);
        }
        else if (i % 4 == 1) {
            float *flt_b = (float*) buf[i];
            start[0] = 0;
            count[0] = len;
            count[1] = 1;
            for (j=0; j<len; j++) {
                start[1] = rank + j * nprocs;
                err = ncmpi_iput_vara_float(ncid, varid[i], start, count,
                                            flt_b, &reqs[k++]);
                ERR(err)
                flt_b += len;
                if (verbose) printf("*-cyclic i=%d j=%d: start=%lld %lld count=%lld %lld\n",i,j,start[0],start[1],count[0],count[1]);
            }
        }
        else if (i % 4 == 2) {
            short *shr_b = (short*) buf[i];
            start[0] = len * rank;
            start[1] = 0;
            count[0] = len;
            count[1] = len/2;
            err = ncmpi_iput_vara_short(ncid, varid[i], start, count,
                                        shr_b, &reqs[k++]);
            ERR(err)
            if (verbose) printf("block-* i=0 start=%lld %lld count=%lld %lld\n",start[0],start[1],count[0],count[1]);

            shr_b += len * (len/2);
            start[1] = len/2;
            count[1] = len - len/2;
            err = ncmpi_iput_vara_short(ncid, varid[i], start, count,
                                        shr_b, &reqs[k++]);
            ERR(err)
            if (verbose) printf("block-* i=1 start=%lld %lld count=%lld %lld\n",start[0],start[1],count[0],count[1]);
        }
        else {
            double *dbl_b = (double*) buf[i];
            start[0] = 0;
            start[1] = len * rank;
            count[0] = len;
            count[1] = len;
            err = ncmpi_iput_vara_double(ncid, varid[i], start, count, dbl_b,
                                         &reqs[k++]);
            ERR(err)
            if (verbose) printf("*-block %d: start=%lld %lld count=%lld %lld\n",i,start[0],start[1],count[0],count[1]);
        }
    }
    num_reqs = k;

    end_t = MPI_Wtime();
    timing[3] = end_t - start_t;
    start_t = end_t;

    sts = (int*) malloc(num_reqs * sizeof(int));

#ifdef USE_INDEP_MODE
    err = ncmpi_begin_indep_data(ncid);          ERR(err)
    err = ncmpi_wait(ncid, num_reqs, reqs, sts); ERR(err)
    err = ncmpi_end_indep_data(ncid);            ERR(err)
#else
    err = ncmpi_wait_all(ncid, num_reqs, reqs, sts); ERR(err)
#endif
    /* check status of all requests */
    for (i=0; i<num_reqs; i++) ERR(sts[i])

    end_t = MPI_Wtime();
    timing[4] = end_t - start_t;
    start_t = end_t;

    /* get the true I/O amount committed */
    err = ncmpi_inq_put_size(ncid, w_size); ERR(err)

    /* get all the hints used */
    err = ncmpi_get_file_info(ncid, w_info_used); ERR(err)

    err = ncmpi_close(ncid); ERR(err)

    end_t = MPI_Wtime();
    timing[5] = end_t - start_t;
    timing[0] = end_t - timing[0];

    free(sts);
    free(reqs);
    for (i=0; i<NVARS; i++) free(buf[i]);

    return 1;
}

/*----< benchmark_read() >---------------------------------------------------*/
static
int benchmark_read(char       *filename,
                   MPI_Offset  len,
                   MPI_Offset *r_size,
                   MPI_Info   *r_info_used,
                   double     *timing)  /* [5] */
{
    int i, j, k, verbose, rank, nprocs, s_rank, err, num_reqs;
    int ncid, omode, varid[NVARS], *reqs, *sts, psizes[2];
    void *buf[NVARS];
    double start_t, end_t;
    MPI_Comm comm=MPI_COMM_WORLD;
    MPI_Offset start[2], count[2];
    MPI_Info info=MPI_INFO_NULL;

    verbose = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
    s_rank = (rank + nprocs / 2 ) % nprocs;

    psizes[0] = psizes[1] = 0;
    MPI_Dims_create(nprocs, 2, psizes);

    /* allocate I/O buffer */
    for (i=0; i<NVARS; i++) {
        if (i % 4 == 0)
            buf[i] = malloc(len * len * sizeof(int));
        else if (i % 4 == 1)
            buf[i] = (float*) malloc(len * len * sizeof(float));
        else if (i % 4 == 2)
            buf[i] = (short*) malloc(len * len * sizeof(short));
        else
            buf[i] = (double*) malloc(len * len * sizeof(double));
    }
    MPI_Barrier(comm);
    timing[0] = MPI_Wtime();

    /* open file for reading -----------------------------------------*/
    omode = NC_NOWRITE;
    err = ncmpi_open(comm, filename, omode, info, &ncid); ERR(err)
    start_t = MPI_Wtime();
    timing[1] = start_t - timing[0];

    /* Note that PnetCDF read the file in chunks of size 256KB, thus the read
     * amount may be more than the file header size
     */
    if (verbose) {
        MPI_Offset h_size, h_extent;
        ncmpi_inq_header_size(ncid, &h_size);
        ncmpi_inq_header_extent(ncid, &h_extent);
        printf("File header size=%lld extent=%lld\n",h_size,h_extent);
    }

    num_reqs = 0;
    for (i=0; i<NVARS; i++) {
        varid[i] = i;
        if (i % 4 == 0)
            num_reqs++; /* complete in 1 nonblocking call */
        else if (i % 4 == 1)
            num_reqs += len; /* complete in len nonblocking calls */
        else if (i % 4 == 2)
            num_reqs += 2; /* write 1st half followed by 2nd half */
        else
            num_reqs++; /* complete in 1 nonblocking call */
    }
    reqs = (int*) malloc(num_reqs * sizeof(int));

    k = 0;
    for (i=0; i<NVARS; i++) {
        if (i % 4 == 0) {
            int *int_b = (int*) buf[i];
            start[0] = len * (s_rank % psizes[0]);
            start[1] = len * ((s_rank / psizes[1]) % psizes[1]);
            count[0] = len;
            count[1] = len;
            err = ncmpi_iget_vara_int(ncid, varid[i], start, count, int_b,
                                      &reqs[k++]);
            ERR(err)
        }
        else if (i % 4 == 1) {
            float *flt_b = (float*) buf[i];
            start[0] = 0;
            count[0] = len;
            count[1] = 1;
            for (j=0; j<len; j++) {
                start[1] = s_rank + j * nprocs;
                err = ncmpi_iget_vara_float(ncid, varid[i], start, count,
                                            flt_b, &reqs[k++]);
                ERR(err)
                flt_b += len;
            }
        }
        else if (i % 4 == 2) {
            short *shr_b = (short*) buf[i];
            start[0] = len * s_rank;
            start[1] = 0;
            count[0] = len;
            count[1] = len/2;
            err = ncmpi_iget_vara_short(ncid, varid[i], start, count,
                                        shr_b, &reqs[k++]);
            ERR(err)

            shr_b += len * (len/2);
            start[1] = len/2;
            count[1] = len - len/2;
            err = ncmpi_iget_vara_short(ncid, varid[i], start, count,
                                        shr_b, &reqs[k++]);
            ERR(err)
        }
        else {
            double *dbl_b = (double*) buf[i];
            start[0] = 0;
            start[1] = len * s_rank;
            count[0] = len;
            count[1] = len;
            err = ncmpi_iget_vara_double(ncid, varid[i], start, count, dbl_b,
                                         &reqs[k++]);
            ERR(err)
        }
    }
    num_reqs = k;

    end_t = MPI_Wtime();
    timing[2] = end_t - start_t;
    start_t = end_t;

    sts = (int*) malloc(num_reqs * sizeof(int));

#ifdef USE_INDEP_MODE
    err = ncmpi_begin_indep_data(ncid);          ERR(err)
    err = ncmpi_wait(ncid, num_reqs, reqs, sts); ERR(err)
    err = ncmpi_end_indep_data(ncid);            ERR(err)
#else
    err = ncmpi_wait_all(ncid, num_reqs, reqs, sts); ERR(err)
#endif
    /* check status of all requests */
    for (i=0; i<num_reqs; i++) ERR(sts[i])

    end_t = MPI_Wtime();
    timing[3] = end_t - start_t;
    start_t = end_t;

    /* get the true I/O amount committed */
    err = ncmpi_inq_get_size(ncid, r_size); ERR(err)

    /* get all the hints used */
    err = ncmpi_get_file_info(ncid, r_info_used); ERR(err)

    err = ncmpi_close(ncid); ERR(err)

    end_t = MPI_Wtime();
    timing[4] = end_t - start_t;
    timing[0] = end_t - timing[0];

    free(sts);
    free(reqs);
    for (i=0; i<NVARS; i++) free(buf[i]);

    return 1;
}

/*----< main() >--------------------------------------------------------------*/
int main(int argc, char** argv) {
    int rank, nprocs;
    double timing[11], max_t[11];
    MPI_Offset len, w_size=0, r_size=0, sum_w_size, sum_r_size;
    MPI_Comm comm=MPI_COMM_WORLD;
    MPI_Info w_info_used, r_info_used;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    if (argc != 3) {
        if (!rank) printf("Usage: %s len filename\n",argv[0]);
        MPI_Finalize();
        return 0;
    }
    len = atoll(argv[1]);

    benchmark_write(argv[2], len, &w_size, &w_info_used, timing);
    benchmark_read (argv[2], len, &r_size, &r_info_used, timing+6);

    MPI_Reduce(&timing, &max_t,     11, MPI_DOUBLE, MPI_MAX, 0, comm);
#ifdef MPI_OFFSET
    MPI_Reduce(&w_size, &sum_w_size, 1, MPI_OFFSET, MPI_SUM, 0, comm);
    MPI_Reduce(&r_size, &sum_r_size, 1, MPI_OFFSET, MPI_SUM, 0, comm);
#else
    MPI_Reduce(&w_size, &sum_w_size, 1, MPI_LONG_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(&r_size, &sum_r_size, 1, MPI_LONG_LONG, MPI_SUM, 0, comm);
#endif
    if (rank == 0) {
        double bw = sum_w_size;
        bw /= 1048576.0;
        print_info(&w_info_used);
        printf("-----------------------------------------------------------\n");
        printf("Write %d variables using nonblocking APIs\n", NVARS);
        printf("In each process, the local variable size is %lld x %lld\n", len,len);
        printf("Total write amount        = %13lld    B\n", sum_w_size);
        printf("            amount        = %16.4f MiB\n", bw);
        printf("            amount        = %16.4f GiB\n", bw/1024);
        printf("Max file open/create time = %16.4f sec\n", max_t[1]);
        printf("Max PnetCDF define   time = %16.4f sec\n", max_t[2]);
        printf("Max nonblocking post time = %16.4f sec\n", max_t[3]);
        printf("Max nonblocking wait time = %16.4f sec\n", max_t[4]);
        printf("Max file close       time = %16.4f sec\n", max_t[5]);
        printf("Max open-to-close    time = %16.4f sec\n", max_t[0]);
        printf("Write bandwidth           = %16.4f MiB/s\n", bw/max_t[0]);
        bw /= 1024.0;
        printf("Write bandwidth           = %16.4f GiB/s\n", bw/max_t[0]);

        bw = sum_r_size;
        bw /= 1048576.0;
        printf("-----------------------------------------------------------\n");
        printf("Read  %d variables using nonblocking APIs\n", NVARS);
        printf("In each process, the local variable size is %lld x %lld\n", len,len);
        printf("Total read  amount        = %13lld    B\n", sum_r_size);
        printf("            amount        = %16.4f MiB\n", bw);
        printf("            amount        = %16.4f GiB\n", bw/1024);
        printf("Max file open/create time = %16.4f sec\n", max_t[7]);
        printf("Max nonblocking post time = %16.4f sec\n", max_t[8]);
        printf("Max nonblocking wait time = %16.4f sec\n", max_t[9]);
        printf("Max file close       time = %16.4f sec\n", max_t[10]);
        printf("Max open-to-close    time = %16.4f sec\n", max_t[6]);
        printf("Read  bandwidth           = %16.4f MiB/s\n", bw/max_t[6]);
        bw /= 1024.0;
        printf("Read  bandwidth           = %16.4f GiB/s\n", bw/max_t[6]);
    }
    MPI_Info_free(&w_info_used);
    MPI_Info_free(&r_info_used);

    /* check if there is any PnetCDF internal malloc residue */
    MPI_Offset malloc_size, sum_size;
    int err = ncmpi_inq_malloc_size(&malloc_size);
    if (err == NC_NOERR) {
        MPI_Reduce(&malloc_size, &sum_size, 1, MPI_OFFSET, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0 && sum_size > 0)
            printf("heap memory allocated by PnetCDF internally has %lld bytes yet to be freed\n",
                   sum_size);
    }

    MPI_Finalize();
    return 0;
}


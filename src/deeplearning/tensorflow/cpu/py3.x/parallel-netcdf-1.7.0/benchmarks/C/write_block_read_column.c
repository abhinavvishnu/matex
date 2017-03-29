/*********************************************************************
 *
 *  Copyright (C) 2013, Northwestern University
 *  See COPYRIGHT notice in top-level directory.
 *
 *********************************************************************/
/* $Id: write_block_read_column.c 2215 2015-12-08 04:58:04Z wkliao $ */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>
#include <pnetcdf.h>

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * This program writes a series of 2D variables with a block-block data
 * partitioning pattern. The 2D variables are read back in a *-block pattern.
 * 
 * The compile and run commands are given below, together with an ncmpidump of
 * the output file. In this example, NVARS = 4.
 *
 *    % mpicc -O2 -o write_block_read_column write_block_read_column.c -lpnetcdf
 *
 *    % mpiexec -l -n 4 ./write_block_read_column 128 /pvfs2/wkliao/testfile.nc
 *
 *    % ncmpidump -h /pvfs2/wkliao/testfile.nc
 *      netcdf testfile {
 *      // file format: CDF-5 (big variables)
 *      dimensions:
 *              Y = 256 ;
 *              X = 256 ;
 *      variables:
 *              int block_block_var_0(Y, X) ;
 *              float block_block_var_1(Y, X) ;
 *              short block_block_var_2(Y, X) ;
 *              double block_block_var_3(Y, X) ;
 *      }
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#define NVARS 4

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
        printf("MPI File Info: [%2d] key = %25s, value = %s\n",i,key,value);
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
    int i, j, rank, nprocs, err;
    int ncid, cmode, varid[NVARS], dimid[2], psizes[2];
    void *buf[NVARS];
    double start_t, end_t;
    MPI_Comm comm=MPI_COMM_WORLD;
    MPI_Offset gsizes[2], start[2], count[2];
    MPI_Info info=MPI_INFO_NULL;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    /* set PnetCDF I/O hints */
    MPI_Info_create(&info);
    MPI_Info_set(info, "nc_header_align_size",      "1");   /* size in bytes */
    MPI_Info_set(info, "nc_var_align_size",         "1");   /* size in bytes */
    MPI_Info_set(info, "nc_header_read_chunk_size", "512"); /* size in bytes */
    /* note that set the above values to 1 to disable the alignment */

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

    err = ncmpi_def_dim(ncid, "Y", gsizes[0], &dimid[0]); ERR(err)
    err = ncmpi_def_dim(ncid, "X", gsizes[1], &dimid[1]); ERR(err)

    /* define variables */
    for (i=0; i<NVARS; i++) {
        char var_name[32];
        sprintf(var_name,"block_block_var_%d",i);
        if (i % 4 == 0) {
            err = ncmpi_def_var(ncid, var_name, NC_INT, 2, dimid, &varid[i]);
            ERR(err)
        }
        else if (i % 4 == 1) {
            err = ncmpi_def_var(ncid, var_name, NC_FLOAT, 2, dimid, &varid[i]);
            ERR(err)
        }
        else if (i % 4 == 2) {
            err = ncmpi_def_var(ncid, var_name, NC_SHORT, 2, dimid, &varid[i]);
            ERR(err)
        }
        else {
            err = ncmpi_def_var(ncid, var_name, NC_DOUBLE, 2, dimid, &varid[i]);
            ERR(err)
        }
    }

    err = ncmpi_enddef(ncid); ERR(err)
    end_t = MPI_Wtime();
    timing[2] = end_t - start_t;
    start_t = end_t;

    start[0] = len * (rank % psizes[0]);
    start[1] = len * ((rank / psizes[1]) % psizes[1]);
    count[0] = len;
    count[1] = len;

    for (i=0; i<NVARS; i++) {
        if (i % 4 == 0) {
            err = ncmpi_put_vara_int_all(ncid, varid[i], start, count, (int*)buf[i]);
            ERR(err)
        }
        else if (i % 4 == 1) {
            err = ncmpi_put_vara_float_all(ncid, varid[i], start, count, (float*)buf[i]);
            ERR(err)
        }
        else if (i % 4 == 2) {
            err = ncmpi_put_vara_short_all(ncid, varid[i], start, count, (short*)buf[i]);
            ERR(err)
        }
        else {
            err = ncmpi_put_vara_double_all(ncid, varid[i], start, count, (double*)buf[i]);
            ERR(err)
        }
    }

    end_t = MPI_Wtime();
    timing[3] = end_t - start_t;
    start_t = end_t;

    /* get the true I/O amount committed */
    err = ncmpi_inq_put_size(ncid, w_size); ERR(err)

    /* get all the hints used */
    err = ncmpi_get_file_info(ncid, w_info_used); ERR(err)

    err = ncmpi_close(ncid); ERR(err)

    end_t = MPI_Wtime();
    timing[4] = end_t - start_t;
    timing[0] = end_t - timing[0];

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
    int i, rank, nprocs, err;
    int ncid, omode, nvars, dimid[2], *varid;
    void **buf;
    double start_t, end_t;
    MPI_Offset gsizes[2], start[2], count[2];
    MPI_Comm comm=MPI_COMM_WORLD;
    MPI_Info info=MPI_INFO_NULL;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    /* set PnetCDF I/O hints */
    MPI_Info_create(&info);
    MPI_Info_set(info, "nc_header_read_chunk_size", "512"); /* size in bytes */

    MPI_Barrier(comm);
    timing[0] = MPI_Wtime();

    /* open file for reading -----------------------------------------*/
    omode = NC_NOWRITE;
    err = ncmpi_open(comm, filename, omode, info, &ncid); ERR(err)
    start_t = MPI_Wtime();
    timing[1] = start_t - timing[0];
    MPI_Info_free(&info);

    err = ncmpi_inq_nvars(ncid, &nvars); ERR(err)
    err = ncmpi_inq_dimid(ncid, "Y", &dimid[0]); ERR(err)
    err = ncmpi_inq_dimid(ncid, "X", &dimid[1]); ERR(err)
    err = ncmpi_inq_dimlen(ncid, dimid[0], &gsizes[0]); ERR(err)
    err = ncmpi_inq_dimlen(ncid, dimid[1], &gsizes[1]); ERR(err)

    varid = (int*)   malloc(nvars * sizeof(int));
    buf   = (void**) malloc(nvars * sizeof(void*));

    /* construct *-block partitioning pattern */
    start[0] = 0;
    start[1] = (gsizes[1] / nprocs) * rank;
    count[0] = gsizes[0];
    count[1] = gsizes[1] / nprocs;
    if (rank < gsizes[1] % nprocs) {
        start[1] += rank;
        count[1]++;
    }
    else
        start[1] += gsizes[1] % nprocs;

    /* allocate I/O buffer */
    len = count[0] * count[1];
    for (i=0; i<nvars; i++) {
        varid[i] = i;
        if (i % 4 == 0)
            buf[i] = malloc(len * sizeof(int));
        else if (i % 4 == 1)
            buf[i] = (float*) malloc(len * sizeof(float));
        else if (i % 4 == 2)
            buf[i] = (short*) malloc(len * sizeof(short));
        else
            buf[i] = (double*) malloc(len * sizeof(double));
    }
    end_t = MPI_Wtime();
    timing[2] = end_t - start_t;
    start_t = end_t;

    for (i=0; i<nvars; i++) {
        if (i % 4 == 0) {
            err = ncmpi_get_vara_int_all(ncid, varid[i], start, count, (int*)buf[i]);
            ERR(err)
        }
        else if (i % 4 == 1) {
            err = ncmpi_get_vara_float_all(ncid, varid[i], start, count, (float*)buf[i]);
            ERR(err)
        }
        else if (i % 4 == 2) {
            err = ncmpi_get_vara_short_all(ncid, varid[i], start, count, (short*)buf[i]);
            ERR(err)
        }
        else {
            err = ncmpi_get_vara_double_all(ncid, varid[i], start, count, (double*)buf[i]);
            ERR(err)
        }
    }

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

    for (i=0; i<nvars; i++) free(buf[i]);
    free(buf);
    free(varid);

    return 1;
}

/*----< main() >--------------------------------------------------------------*/
int main(int argc, char** argv) {
    int rank, nprocs;
    double timing[10], max_t[10];
    MPI_Offset len, w_size, r_size, sum_w_size, sum_r_size;
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
    benchmark_read (argv[2], len, &r_size, &r_info_used, timing+5);

    MPI_Reduce(&timing, &max_t,     10, MPI_DOUBLE, MPI_MAX, 0, comm);
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
        printf("Write %d variables using blocking APIs\n", NVARS);
        printf("In each process, the local variable size is %lld x %lld\n", len,len);
        printf("Total write amount        = %13lld    B\n", sum_w_size);
        printf("            amount        = %16.4f MiB\n", bw);
        printf("            amount        = %16.4f GiB\n", bw/1024);
        printf("Max file open/create time = %16.4f sec\n", max_t[1]);
        printf("Max PnetCDF define   time = %16.4f sec\n", max_t[2]);
        printf("Max PnetCDF put      time = %16.4f sec\n", max_t[3]);
        printf("Max file close       time = %16.4f sec\n", max_t[4]);
        printf("Max open-to-close    time = %16.4f sec\n", max_t[0]);
        printf("Write bandwidth           = %16.4f MiB/s\n", bw/max_t[0]);
        bw /= 1024.0;
        printf("Write bandwidth           = %16.4f GiB/s\n", bw/max_t[0]);

        bw = sum_r_size;
        bw /= 1048576.0;
        printf("-----------------------------------------------------------\n");
        printf("Read  %d variables using blocking APIs\n", NVARS);
        printf("In each process, the local variable size is %lld x %lld\n", len,len);
        printf("Total read  amount        = %13lld    B\n", sum_r_size);
        printf("            amount        = %16.4f MiB\n", bw);
        printf("            amount        = %16.4f GiB\n", bw/1024);
        printf("Max file open/create time = %16.4f sec\n", max_t[6]);
        printf("Max PnetCDF inquire  time = %16.4f sec\n", max_t[7]);
        printf("Max PnetCDF get      time = %16.4f sec\n", max_t[8]);
        printf("Max file close       time = %16.4f sec\n", max_t[9]);
        printf("Max open-to-close    time = %16.4f sec\n", max_t[5]);
        printf("Read  bandwidth           = %16.4f MiB/s\n", bw/max_t[5]);
        bw /= 1024.0;
        printf("Read  bandwidth           = %16.4f GiB/s\n", bw/max_t[5]);
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


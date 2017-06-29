/*********************************************************************
 *
 *  Copyright (C) 2015, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *********************************************************************/
/* $Id: ivarn.c 2219 2015-12-11 22:30:03Z wkliao $ */

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * This example tests using calls to ncmpi_iput_varn_int(),
 * ncmpi_iput_varn_float(), ncmpi_iput_varn_double() to write a sequence of
 * requests with arbitrary array indices and lengths. Note that the request IDs
 * in the argument array_of_requests[] of ncmpi_wait_all() are in an arbitray
 * order (instead in an increasing order).
 *
 * The compile and run commands are given below, together with an ncmpidump of
 * the output file.
 *
 *    % mpicc -O2 -o ivarn ivarn.c -lpnetcdf
 *    % mpiexec -n 4 ./ivarn /pvfs2/wkliao/testfile.nc
 *    % ncmpidump /pvfs2/wkliao/testfile.nc
 *    netcdf testfile {
 *    // file format: CDF-5 (big variables)
 *    dimensions:
 *             netcdf testfile {
 *    dimensions:
 *            dim000001 = 16 ;
 *            time = UNLIMITED ; // (1 currently)
 *    variables:
 *            int vari0001(time, dim000001) ;
 *            float varr0001(time, dim000001) ;
 *            double vard0001(time, dim000001) ;
 *            int vari0002(time, dim000001) ;
 *            float varr0002(time, dim000001) ;
 *            double vard0002(time, dim000001) ;
 *    data:
 *    
 *     vari0001 =
 *      1, _, 3, 4, 5, 6, 7, _, 9, 10, 11, 12, 13, 14, _, 16 ;
 *    
 *     varr0001 =
 *      1.1, _, 3.1, 4.1, 5.1, 6.1, 7.1, _, 9.1, 10.1, 11.1, 12.1, 13.1, 14.1, _, 16.1 ;
 *    
 *     vard0001 =
 *      1.3, _, 3.3, 4.3, 5.3, 6.3, 7.3, _, 9.3, 10.3, 11.3, 12.3, 13.3, 14.3, _, 16.3 ;
 *    
 *     vari0002 =
 *      1, _, 3, 4, 5, 6, 7, _, 9, 10, 11, 12, 13, 14, _, 16 ;
 *    
 *     varr0002 =
 *      1.2, _, 3.2, 4.2, 5.2, 6.2, 7.2, _, 9.2, 10.2, 11.2, 12.2, 13.2, 14.2, _, 16.2 ;
 *    
 *     vard0002 =
 *      1.4, _, 3.4, 4.4, 5.4, 6.4, 7.4, _, 9.4, 10.4, 11.4, 12.4, 13.4, 14.4, _, 16.4 ;
 *    }
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <string.h> /* strcpy() */
#include <mpi.h>
#include <pnetcdf.h>

#include <testutils.h>

#define LEN 16
#define ERR {if(err!=NC_NOERR)printf("Error at line=%d: %s\n", __LINE__, ncmpi_strerror(err));}

static
int check_int_buf(int *buffer)
{
    int i, nprocs;
    int expected[LEN];

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    for (i=0; i<LEN; i++) expected[i] = i+1;
    expected[1] = expected[7] = expected[14] = NC_FILL_INT;

    /* check if the contents of buf are expected */
    for (i=0; i<LEN; i++) {
        if (nprocs == 1) { if (i == 4) break; }
        else if (nprocs == 2) {
            if (3 < i && i < 7) continue;
            if (i == 12) break;
        }
        else if (nprocs == 3) { if (i == 12) break; }

        if (buffer[i] != expected[i]) {
            printf("Expected read buf[%d]=%d, but got %d\n",
                   i,expected[i],buffer[i]);
            return 1;
        }
    }
    return 0;
}

static
int check_flt_buf(float *buffer, float extra)
{
    int i, nprocs;
    float expected[LEN];

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    for (i=0; i<LEN; i++) expected[i] = extra+i+1;
    expected[1] = expected[7] = expected[14] = NC_FILL_FLOAT;

    /* check if the contents of buf are expected */
    for (i=0; i<LEN; i++) {
        if (nprocs == 1) { if (i == 4) break; }
        else if (nprocs == 2) {
            if (3 < i && i < 7) continue;
            if (i == 12) break;
        }
        else if (nprocs == 3) { if (i == 12) break; }

        if (buffer[i] != expected[i]) {
            printf("Expected read buf[%d]=%.1f, but got %.1f\n",
                   i,expected[i],buffer[i]);
            return 1;
        }
    }
    return 0;
}

static
int check_dbl_buf(double *buffer, double extra)
{
    int i, nprocs;
    double expected[LEN];

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    for (i=0; i<LEN; i++) expected[i] = extra+i+1;
    expected[1] = expected[7] = expected[14] = NC_FILL_DOUBLE;

    /* check if the contents of buf are expected */
    for (i=0; i<LEN; i++) {
        if (nprocs == 1) { if (i == 4) break; }
        else if (nprocs == 2) {
            if (3 < i && i < 7) continue;
            if (i == 12) break;
        }
        else if (nprocs == 3) { if (i == 12) break; }

        if (buffer[i] != expected[i]) {
            printf("Expected read buf[%d]=%.1f, but got %.1f\n",
                   i,expected[i],buffer[i]);
            return 1;
        }
    }
    return 0;
}

int main(int argc, char** argv)
{
    char filename[256];
    int i, rank, nprocs, err, verbose=0, nerrs=0;
    int ncid, cmode, dimid[2];
    int vari0001, vari0002, varr0001, varr0002, vard0001, vard0002;
    MPI_Offset **starts, **counts;
    int req[LEN], st[LEN], num_reqs=0;
    int ibuf1[LEN], ibuf2[LEN];
    float rbuf1[LEN], rbuf2[LEN];
    double dbuf1[LEN], dbuf2[LEN];

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
        sprintf(cmd_str, "*** TESTING C   %s for ncmpi_iput_varn_<type>() ", argv[0]);
        printf("%-66s ------ ", cmd_str); fflush(stdout);
    }

    if (verbose && nprocs != 4 && rank == 0)
        printf("Warning: %s is intended to run on 4 processes\n",argv[0]);

    /* create a new file for writing ----------------------------------------*/
    cmode = NC_CLOBBER | NC_64BIT_DATA;
    err = ncmpi_create(MPI_COMM_WORLD, filename, cmode, MPI_INFO_NULL, &ncid); ERR

    err = ncmpi_def_dim(ncid, "dim000001", LEN, &dimid[1]); ERR
    err = ncmpi_def_dim(ncid, "time", NC_UNLIMITED, &dimid[0]); ERR
    err = ncmpi_def_var(ncid, "vari0001", NC_INT, 2, dimid, &vari0001); ERR
    err = ncmpi_def_var(ncid, "varr0001", NC_FLOAT, 2, dimid, &varr0001); ERR
    err = ncmpi_def_var(ncid, "vard0001", NC_DOUBLE, 2, dimid, &vard0001); ERR
    err = ncmpi_def_var(ncid, "vari0002", NC_INT, 2, dimid, &vari0002); ERR
    err = ncmpi_def_var(ncid, "varr0002", NC_FLOAT, 2, dimid, &varr0002); ERR
    err = ncmpi_def_var(ncid, "vard0002", NC_DOUBLE, 2, dimid, &vard0002); ERR
    err = ncmpi_enddef(ncid); ERR

    starts    = (MPI_Offset**) malloc(2 *    sizeof(MPI_Offset*));
    counts    = (MPI_Offset**) malloc(2 *    sizeof(MPI_Offset*));
    starts[0] = (MPI_Offset*)  calloc(2 * 2, sizeof(MPI_Offset));
    counts[0] = (MPI_Offset*)  calloc(2 * 2, sizeof(MPI_Offset));
    for (i=1; i<2; i++) {
        starts[i] = starts[i-1] + 2;
        counts[i] = counts[i-1] + 2;
    }

    /* assign arbitrary starts and counts */
    if (rank == 0) {
        /* vari0001 and vari0002 */
        starts[0][0] = 0; starts[0][1] = 1; counts[0][0] = 1; counts[0][1] = 1;
        ibuf1[0] = NC_FILL_INT;
        err = ncmpi_iput_varn_int(ncid, vari0001, 1, starts, counts, ibuf1, &req[1]); ERR
        ibuf2[0] = NC_FILL_INT;
        err = ncmpi_iput_varn_int(ncid, vari0002, 1, starts, counts, ibuf2, &req[7]); ERR

        starts[0][0] = 0; starts[0][1] = 0; counts[0][0] = 1; counts[0][1] = 1;
        starts[1][0] = 0; starts[1][1] = 2; counts[1][0] = 1; counts[1][1] = 2;
        ibuf1[1] = 1; ibuf1[2] = 3; ibuf1[3] = 4;
        err = ncmpi_iput_varn_int(ncid, vari0001, 2, starts, counts, ibuf1+1, &req[0]); ERR
        ibuf2[1] = 1; ibuf2[2] = 3; ibuf2[3] = 4;
        err = ncmpi_iput_varn_int(ncid, vari0002, 2, starts, counts, ibuf2+1, &req[6]); ERR

        /* varr0001 and varr0002 */
        starts[0][0] = 0; starts[0][1] = 1; counts[0][0] = 1; counts[0][1] = 1;
        rbuf1[0] = NC_FILL_FLOAT;
        err = ncmpi_iput_varn_float(ncid, varr0001, 1, starts, counts, rbuf1, &req[3]); ERR
        rbuf2[0] = NC_FILL_FLOAT;
        err = ncmpi_iput_varn_float(ncid, varr0002, 1, starts, counts, rbuf2, &req[9]); ERR

        starts[0][0] = 0; starts[0][1] = 0; counts[0][0] = 1; counts[0][1] = 1;
        starts[1][0] = 0; starts[1][1] = 2; counts[1][0] = 1; counts[1][1] = 2;
        rbuf1[1] = 1.1; rbuf1[2] = 3.1; rbuf1[3] = 4.1;
        err = ncmpi_iput_varn_float(ncid, varr0001, 2, starts, counts, rbuf1+1, &req[2]); ERR
        rbuf2[1] = 1.2; rbuf2[2] = 3.2; rbuf2[3] = 4.2;
        err = ncmpi_iput_varn_float(ncid, varr0002, 2, starts, counts, rbuf2+1, &req[8]); ERR

        /* vard0001 and vard0002 */
        starts[0][0] = 0; starts[0][1] = 1; counts[0][0] = 1; counts[0][1] = 1;
        dbuf1[0] = NC_FILL_DOUBLE;
        err = ncmpi_iput_varn_double(ncid, vard0001, 1, starts, counts, dbuf1, &req[5]); ERR
        dbuf2[0] = NC_FILL_DOUBLE;
        err = ncmpi_iput_varn_double(ncid, vard0002, 1, starts, counts, dbuf2, &req[11]); ERR

        starts[0][0] = 0; starts[0][1] = 0; counts[0][0] = 1; counts[0][1] = 1;
        starts[1][0] = 0; starts[1][1] = 2; counts[1][0] = 1; counts[1][1] = 2;
        dbuf1[1] = 1.3; dbuf1[2] = 3.3; dbuf1[3] = 4.3;
        err = ncmpi_iput_varn_double(ncid, vard0001, 2, starts, counts, dbuf1+1, &req[4]); ERR
        dbuf2[1] = 1.4; dbuf2[2] = 3.4; dbuf2[3] = 4.4;
        err = ncmpi_iput_varn_double(ncid, vard0002, 2, starts, counts, dbuf2+1, &req[10]); ERR

        num_reqs = 12;
        /* rank 0 is writing the followings: ("x" means skip, "-" means fill value)
                  1  -  3  4  x  x  x  x  x  x  x  x  x  x  x  x 
         */
    } else if (rank ==1) {
        /* vari0001 and vari0002 */
        starts[0][0] = 0; starts[0][1] = 7; counts[0][0] = 1; counts[0][1] = 1;
        ibuf1[0] = NC_FILL_INT;
        err = ncmpi_iput_varn_int(ncid, vari0001, 1, starts, counts, ibuf1, &req[1]); ERR
        ibuf2[0] = NC_FILL_INT;
        err = ncmpi_iput_varn_int(ncid, vari0002, 1, starts, counts, ibuf2, &req[7]); ERR

        starts[0][0] = 0; starts[0][1] = 8; counts[0][0] = 1; counts[0][1] = 4;
        ibuf1[1] = 9; ibuf1[2] = 10; ibuf1[3] = 11; ibuf1[4] = 12;
        err = ncmpi_iput_varn_int(ncid, vari0001, 1, starts, counts, ibuf1+1, &req[0]); ERR
        ibuf2[1] = 9; ibuf2[2] = 10; ibuf2[3] = 11; ibuf2[4] = 12;
        err = ncmpi_iput_varn_int(ncid, vari0002, 1, starts, counts, ibuf2+1, &req[6]); ERR

        /* varr0001 and varr0002 */
        starts[0][0] = 0; starts[0][1] = 7; counts[0][0] = 1; counts[0][1] = 1;
        rbuf1[0] = NC_FILL_FLOAT;
        err = ncmpi_iput_varn_float(ncid, varr0001, 1, starts, counts, rbuf1, &req[3]); ERR
        rbuf2[0] = NC_FILL_FLOAT;
        err = ncmpi_iput_varn_float(ncid, varr0002, 1, starts, counts, rbuf2, &req[9]); ERR

        starts[0][0] = 0; starts[0][1] = 8; counts[0][0] = 1; counts[0][1] = 4;
        rbuf1[1] = 9.1; rbuf1[2] = 10.1; rbuf1[3] = 11.1; rbuf1[4] = 12.1;
        err = ncmpi_iput_varn_float(ncid, varr0001, 1, starts, counts, rbuf1+1, &req[2]); ERR
        rbuf2[1] = 9.2; rbuf2[2] = 10.2; rbuf2[3] = 11.2; rbuf2[4] = 12.2;
        err = ncmpi_iput_varn_float(ncid, varr0002, 1, starts, counts, rbuf2+1, &req[8]); ERR

        /* vard0001 and vard0002 */
        starts[0][0] = 0; starts[0][1] = 7; counts[0][0] = 1; counts[0][1] = 1;
        dbuf1[0] = NC_FILL_DOUBLE;
        err = ncmpi_iput_varn_double(ncid, vard0001, 1, starts, counts, dbuf1, &req[5]); ERR
        dbuf2[0] = NC_FILL_DOUBLE;
        err = ncmpi_iput_varn_double(ncid, vard0002, 1, starts, counts, dbuf2, &req[11]); ERR

        starts[0][0] = 0; starts[0][1] = 8; counts[0][0] = 1; counts[0][1] = 4;
        dbuf1[1] = 9.3; dbuf1[2] = 10.3; dbuf1[3] = 11.3; dbuf1[4] = 12.3;
        err = ncmpi_iput_varn_double(ncid, vard0001, 1, starts, counts, dbuf1+1, &req[4]); ERR
        dbuf2[1] = 9.4; dbuf2[2] = 10.4; dbuf2[3] = 11.4; dbuf2[4] = 12.4;
        err = ncmpi_iput_varn_double(ncid, vard0002, 1, starts, counts, dbuf2+1, &req[10]); ERR

        num_reqs = 12;
        /* rank 1 is writing the followings: ("x" means skip, "-" means fill value)
                  vari0001:
                  x  x  x  x  x  x  x  x  -  9  10 11 12 x  x  x  x 
         */
    } else if (rank ==2) {
        /* vari0001 and vari0002 */
        starts[0][0] = 0; starts[0][1] = 4; counts[0][0] = 1; counts[0][1] = 3;
        ibuf1[0] = 5; ibuf1[1] = 6; ibuf1[2] = 7;
        err = ncmpi_iput_varn_int(ncid, vari0001, 1, starts, counts, ibuf1, &req[0]); ERR
        ibuf2[0] = 5; ibuf2[1] = 6; ibuf2[2] = 7;
        err = ncmpi_iput_varn_int(ncid, vari0002, 1, starts, counts, ibuf2, &req[1]); ERR

        /* varr0001 and varr0002 */
        rbuf1[0] = 5.1; rbuf1[1] = 6.1; rbuf1[2] = 7.1;
        err = ncmpi_iput_varn_float(ncid, varr0001, 1, starts, counts, rbuf1, &req[2]); ERR
        rbuf2[0] = 5.2; rbuf2[1] = 6.2; rbuf2[2] = 7.2;
        err = ncmpi_iput_varn_float(ncid, varr0002, 1, starts, counts, rbuf2, &req[3]); ERR

        /* vard0001 and vard0002 */
        dbuf1[0] = 5.3; dbuf1[1] = 6.3; dbuf1[2] = 7.3;
        err = ncmpi_iput_varn_double(ncid, vard0001, 1, starts, counts, dbuf1, &req[4]); ERR
        dbuf2[0] = 5.4; dbuf2[1] = 6.4; dbuf2[2] = 7.4;
        err = ncmpi_iput_varn_double(ncid, vard0002, 1, starts, counts, dbuf2, &req[5]); ERR

        num_reqs = 6;
        /* rank 2 is writing the followings: ("x" means skip, "-" means fill value)
                  vari0001:
                  x  x  x  x  x  5  6  7  x  x  x  x  x  x  x  x  x 
         */
    } else if (rank ==3) {
        /* vari0001 and vari0002 */
        starts[0][0] = 0; starts[0][1] = 14; counts[0][0] = 1; counts[0][1] = 1;
        ibuf1[0] = NC_FILL_INT;
        err = ncmpi_iput_varn_int(ncid, vari0001, 1, starts, counts, ibuf1, &req[0]); ERR
        ibuf2[0] = NC_FILL_INT;
        err = ncmpi_iput_varn_int(ncid, vari0002, 1, starts, counts, ibuf2, &req[1]); ERR

        starts[0][0] = 0; starts[0][1] = 12; counts[0][0] = 1; counts[0][1] = 2;
        starts[1][0] = 0; starts[1][1] = 15; counts[1][0] = 1; counts[1][1] = 1;
        ibuf1[1] = 13; ibuf1[2] = 14; ibuf1[3] = 16;
        err = ncmpi_iput_varn_int(ncid, vari0001, 2, starts, counts, ibuf1+1, &req[2]); ERR
        ibuf2[1] = 13; ibuf2[2] = 14; ibuf2[3] = 16;
        err = ncmpi_iput_varn_int(ncid, vari0002, 2, starts, counts, ibuf2+1, &req[3]); ERR

        /* varr0001 and varr0002 */
        starts[0][0] = 0; starts[0][1] = 14; counts[0][0] = 1; counts[0][1] = 1;
        rbuf1[0] = NC_FILL_FLOAT;
        err = ncmpi_iput_varn_float(ncid, varr0001, 1, starts, counts, rbuf1, &req[4]); ERR
        rbuf2[0] = NC_FILL_FLOAT;
        err = ncmpi_iput_varn_float(ncid, varr0002, 1, starts, counts, rbuf2, &req[5]); ERR

        starts[0][0] = 0; starts[0][1] = 12; counts[0][0] = 1; counts[0][1] = 2;
        starts[1][0] = 0; starts[1][1] = 15; counts[1][0] = 1; counts[1][1] = 1;
        rbuf1[1] = 13.1; rbuf1[2] = 14.1; rbuf1[3] = 16.1;
        err = ncmpi_iput_varn_float(ncid, varr0001, 2, starts, counts, rbuf1+1, &req[6]); ERR
        rbuf2[1] = 13.2; rbuf2[2] = 14.2; rbuf2[3] = 16.2;
        err = ncmpi_iput_varn_float(ncid, varr0002, 2, starts, counts, rbuf2+1, &req[7]); ERR

        /* vard0001 and vard0002 */
        starts[0][0] = 0; starts[0][1] = 14; counts[0][0] = 1; counts[0][1] = 1;
        dbuf1[0] = NC_FILL_DOUBLE;
        err = ncmpi_iput_varn_double(ncid, vard0001, 1, starts, counts, dbuf1, &req[8]); ERR
        dbuf2[0] = NC_FILL_DOUBLE;
        err = ncmpi_iput_varn_double(ncid, vard0002, 1, starts, counts, dbuf2, &req[9]); ERR

        starts[0][0] = 0; starts[0][1] = 12; counts[0][0] = 1; counts[0][1] = 2;
        starts[1][0] = 0; starts[1][1] = 15; counts[1][0] = 1; counts[1][1] = 1;
        dbuf1[1] = 13.3; dbuf1[2] = 14.3; dbuf1[3] = 16.3;
        err = ncmpi_iput_varn_double(ncid, vard0001, 2, starts, counts, dbuf1+1, &req[10]); ERR
        dbuf2[1] = 13.4; dbuf2[2] = 14.4; dbuf2[3] = 16.4;
        err = ncmpi_iput_varn_double(ncid, vard0002, 2, starts, counts, dbuf2+1, &req[11]); ERR

        num_reqs = 12;
        /* rank 3 is writing the followings: ("x" means skip, "-" means fill value)
                  vari0001:
                  x  x  x  x  x  x  x  x  x  x  x  x 13 14  - 16
         */
    }

    err = ncmpi_wait_all(ncid, num_reqs, req, st); ERR
    for (i=0; i<num_reqs; i++) {
        err = st[i]; ERR
    }

    err = ncmpi_close(ncid); ERR

    free(starts[0]);
    free(counts[0]);
    free(starts);
    free(counts);

    err = ncmpi_open(MPI_COMM_WORLD, filename, NC_NOWRITE, MPI_INFO_NULL, &ncid); ERR

    err = ncmpi_inq_varid(ncid, "vari0001", &vari0001); ERR
    err = ncmpi_inq_varid(ncid, "varr0001", &varr0001); ERR
    err = ncmpi_inq_varid(ncid, "vard0001", &vard0001); ERR
    err = ncmpi_inq_varid(ncid, "vari0002", &vari0002); ERR
    err = ncmpi_inq_varid(ncid, "varr0002", &varr0002); ERR
    err = ncmpi_inq_varid(ncid, "vard0002", &vard0002); ERR

    for (i=0; i<LEN; i++) ibuf1[i] = -1;
    err = ncmpi_get_var_int_all(ncid, vari0001, ibuf1); ERR
    nerrs += check_int_buf(ibuf1);

    for (i=0; i<LEN; i++) ibuf2[i] = -1;
    err = ncmpi_get_var_int_all(ncid, vari0002, ibuf2); ERR
    nerrs += check_int_buf(ibuf2);

    for (i=0; i<LEN; i++) rbuf1[i] = -1;
    err = ncmpi_get_var_float_all(ncid, varr0001, rbuf1); ERR
    nerrs += check_flt_buf(rbuf1, 0.1);

    for (i=0; i<LEN; i++) rbuf2[i] = -1;
    err = ncmpi_get_var_float_all(ncid, varr0002, rbuf2); ERR
    nerrs += check_flt_buf(rbuf2, 0.2);

    for (i=0; i<LEN; i++) dbuf1[i] = -1;
    err = ncmpi_get_var_double_all(ncid, vard0001, dbuf1); ERR
    nerrs += check_dbl_buf(dbuf1, 0.3);

    for (i=0; i<LEN; i++) dbuf2[i] = -1;
    err = ncmpi_get_var_double_all(ncid, vard0002, dbuf2); ERR
    nerrs += check_dbl_buf(dbuf2, 0.4);

    err = ncmpi_close(ncid); ERR

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


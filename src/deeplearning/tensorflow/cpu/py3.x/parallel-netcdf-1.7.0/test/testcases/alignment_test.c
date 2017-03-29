/*
 *  Copyright (C) 2003, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *  $Id: alignment_test.c 2133 2015-09-26 19:16:01Z wkliao $
 */

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * This program tests if the file header size and variable offsets are properly
 * set when using a different set of alignment hints to open an existing file
 * and entering the redef mode to add more dimensions, attributes, and
 * variables, causing the expansion of the header.
 *
 * The compile and run commands are given below.
 *
 *    % mpicc -g -o alignment_test alignment_test.c -lpnetcdf
 *
 *    % mpiexec -l -n 4 alignment_test testfile.nc
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>
#include <pnetcdf.h>

#include <testutils.h>

#define NVARS 8
#define NX 5

#define ERR {if(err!=NC_NOERR){nerrs++;printf("Error at line=%d: %s\n", __LINE__, ncmpi_strerror(err));}}

int main(int argc, char** argv) {
    char *filename="redef1.nc";
    int i, j, rank, nprocs, err, verbose=0, nerrs=0;
    int ncid, cmode, varid[NVARS], dimid[2], *buf;
    char str[32];
    MPI_Offset start[2], count[2];
    MPI_Offset new_var_off[NVARS*2], old_var_off[NVARS*2];
    MPI_Offset header_size[2], header_extent[2];
    MPI_Info info=MPI_INFO_NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc > 2) {
        if (!rank) printf("Usage: %s [filename]\n",argv[0]);
        MPI_Finalize();
        return 0;
    }
    if (argc == 2) filename = argv[1];

    if (rank == 0) {
        char cmd_str[256];
        sprintf(cmd_str, "*** TESTING C   %s for alignment ", argv[0]);
        printf("%-66s ------ ", cmd_str); fflush(stdout);
    }

    /* create a new file for writing ----------------------------------------*/
    cmode = NC_CLOBBER | NC_64BIT_DATA;
    err = ncmpi_create(MPI_COMM_WORLD, filename, cmode, info, &ncid); ERR

    /* define dimension */
    err = ncmpi_def_dim(ncid, "Y", NC_UNLIMITED, &dimid[0]); ERR
    err = ncmpi_def_dim(ncid, "X", NX*nprocs, &dimid[1]); ERR

#define TEST_FIXED_VAR
#define TEST_RECORD_VAR
    /* Odd numbers are fixed variables, even numbers are record variables */
    for (i=0; i<NVARS; i++) {
#ifdef TEST_FIXED_VAR
        if (i%2) {
            sprintf(str,"fixed_var_%d",i);
            err = ncmpi_def_var(ncid, str, NC_INT, 1, dimid+1, &varid[i]); ERR
        }
#endif
#ifdef TEST_RECORD_VAR
        if (i%2 == 0) {
            sprintf(str,"record_var_%d",i);
            err = ncmpi_def_var(ncid, str, NC_INT, 2, dimid, &varid[i]); ERR
        }
#endif
    }
    err = ncmpi_enddef(ncid); ERR

    /* write all variables */
    buf = (int*) malloc(NX * sizeof(int));
    for (i=0; i<NVARS; i++) {
        for (j=0; j<NX; j++) buf[j] = rank*1000 + i*10 + j;
#ifdef TEST_FIXED_VAR
        if (i%2) {
            start[0] = NX*rank;
            count[0] = NX;
            err = ncmpi_put_vara_int_all(ncid, varid[i], start, count, buf); ERR
        }
#endif
#ifdef TEST_RECORD_VAR
        if (i%2 == 0) {
            start[0] = 0; start[1] = NX*rank;
            count[0] = 1; count[1] = NX;
            err = ncmpi_put_vara_int_all(ncid, varid[i], start, count, buf); ERR
            for (j=0; j<NX; j++) buf[j] = rank*1000 + 100 + i*10 + j;
            start[0] = 1; /* write 2nd record */
            err = ncmpi_put_vara_int_all(ncid, varid[i], start, count, buf); ERR
        }
#endif
    }
    err = ncmpi_close(ncid); ERR

    /* Now, reopen the file and grow the header and read data back */

    /* mimic netCDF that does not do alignments */
    MPI_Info_create(&info);
    MPI_Info_set(info, "nc_header_align_size", "1"); /* size in bytes */
    MPI_Info_set(info, "nc_var_align_size",    "197"); /* size in bytes */

    /* open the file for adding more metadata */
    err = ncmpi_open(MPI_COMM_WORLD, filename, NC_WRITE, info, &ncid); ERR

    /* get header size and extent, and offsets of all variables */
    err = ncmpi_inq_header_size(ncid, &header_size[0]); ERR
    err = ncmpi_inq_header_extent(ncid, &header_extent[0]); ERR
    for (i=0; i<NVARS; i++) {
#ifdef TEST_FIXED_VAR
        if (i%2)
            err = ncmpi_inq_varoffset(ncid, varid[i], &old_var_off[i]);
#endif
#ifdef TEST_RECORD_VAR
        if (i%2==0)
            err = ncmpi_inq_varoffset(ncid, varid[i], &old_var_off[i]);
#endif
        ERR
    }

    /* enter redef mode */
    err = ncmpi_redef(ncid); ERR

    /* add attributes to make header grow */
    for (i=0; i<NVARS; i++) {
        sprintf(str, "annotation_for_var_%d",i);
#ifdef TEST_FIXED_VAR
        if (i%2)
            err = ncmpi_put_att_text(ncid, varid[i], "text_attr", strlen(str), str);
#endif
#ifdef TEST_RECORD_VAR
        if (i%2==0)
            err = ncmpi_put_att_text(ncid, varid[i], "text_attr", strlen(str), str);
#endif
        ERR
    }

    /* add new dimensions */
    int new_dimid[3];
    err = ncmpi_def_dim(ncid, "new_dim_a", 5,         &new_dimid[0]); ERR
    err = ncmpi_def_dim(ncid, "new_dim_b", 4,         &new_dimid[1]); ERR
    err = ncmpi_def_dim(ncid, "new_dim_c", NX*nprocs, &new_dimid[2]); ERR

    /* add new variables */
    int new_varid[NVARS];
    for (i=0; i<NVARS; i++) {
#ifdef TEST_FIXED_VAR
        if (i%2 == 0) {
            sprintf(str,"fixed_var_%d",i+NVARS);
            err = ncmpi_def_var(ncid, str, NC_INT, 1, new_dimid+2, &new_varid[i]); ERR
        }
#endif
#ifdef TEST_RECORD_VAR
        if (i%2 == 1) {
            sprintf(str,"record_var_%d",i+NVARS);
            err = ncmpi_def_var(ncid, str, NC_INT, 2, dimid, &new_varid[i]); ERR
        }
#endif
    }
    err = ncmpi_enddef(ncid); ERR

    /* get the new header size and extent, also all variables' starting
       file offsets */
    err = ncmpi_inq_header_size(ncid, &header_size[1]); ERR
    err = ncmpi_inq_header_extent(ncid, &header_extent[1]); ERR
    if (rank == 0 && verbose) {
        printf("NX = %d (integer type)\n",NX);
        printf("old header_size  =%lld new header_size  =%lld\n",header_size[0],header_size[1]);
        printf("old header_extent=%lld new header_extent=%lld\n",header_extent[0],header_extent[1]);

#ifdef TEST_FIXED_VAR
        for (i=1; i<NVARS; i+=2) {
            err = ncmpi_inq_varoffset(ncid, varid[i], &new_var_off[i]); ERR
            printf("old fixed  var[%2d] old offset=%4lld new offset=%4lld\n",i,old_var_off[i],new_var_off[i]);
        }
        for (i=NVARS; i<2*NVARS; i++) {
            if (i%2 == 0) {
                err = ncmpi_inq_varoffset(ncid, new_varid[i-NVARS], &new_var_off[i]); ERR
                printf("new fixed  var[%2d]                 new offset=%4lld\n",i,new_var_off[i]);
            }
        }
#endif
#ifdef TEST_RECORD_VAR
        for (i=0; i<NVARS; i+=2) {
            err = ncmpi_inq_varoffset(ncid, varid[i], &new_var_off[i]); ERR
            printf("old record var[%2d] old offset=%4lld new offset=%4lld\n",i,old_var_off[i],new_var_off[i]);
        }
        for (i=NVARS; i<2*NVARS; i++) {
            if (i%2) {
                err = ncmpi_inq_varoffset(ncid, new_varid[i-NVARS], &new_var_off[i]); ERR
                printf("new record var[%2d]                 new offset=%4lld\n",i,new_var_off[i]);
            }
        }
#endif
    }

    /* write to the new variables */
    for (i=0; i<NVARS; i++) {
        for (j=0; j<NX; j++) buf[j] = -1 * (i*10 + j);
#ifdef TEST_FIXED_VAR
        if (i%2 == 0) {
            start[0] = NX*rank;
            count[0] = NX;
            err = ncmpi_put_vara_int_all(ncid, new_varid[i], start, count, buf); ERR
        }
#endif
#ifdef TEST_RECORD_VAR
        if (i%2 == 1) {
            start[0] = 0; start[1] = NX*rank;
            count[0] = 1; count[1] = NX;
            err = ncmpi_put_vara_int_all(ncid, new_varid[i], start, count, buf); ERR
            for (j=0; j<NX; j++) buf[j] = -1 * (100 + i*10 + j);
            start[0] = 1; /* write 2nd record */
            err = ncmpi_put_vara_int_all(ncid, new_varid[i], start, count, buf); ERR
        }
#endif
    }

    /* read old variables and check their contents */
    for (i=0; i<NVARS; i++) {
#ifdef TEST_FIXED_VAR
        if (i%2) {
            start[0] = NX*rank;
            count[0] = NX;
            err = ncmpi_get_vara_int_all(ncid, varid[i], start, count, buf); ERR
            for (j=0; j<NX; j++)
                if (buf[j] != rank*1000 + i*10 + j) {
                    printf("read error i=%d buf[j=%d]=%d != %d\n",i,j,buf[j],rank*1000+i*10+j);
                    nerrs++;
                }
        }
#endif
#ifdef TEST_RECORD_VAR
        if (i%2 == 0) {
            start[0] = 0; start[1] = NX*rank;
            count[0] = 1; count[1] = NX;
            err = ncmpi_get_vara_int_all(ncid, varid[i], start, count, buf); ERR
            for (j=0; j<NX; j++)
                if (buf[j] != rank*1000+i*10+j) {
                    printf("read error i=%d buf[j=%d]=%d != %d\n",i,j,buf[j],rank*1000+i*10+j);
                    nerrs++;
                }
            start[0] = 1;
            err = ncmpi_get_vara_int_all(ncid, varid[i], start, count, buf); ERR
            for (j=0; j<NX; j++)
                if (buf[j] != rank*1000 + 100 + i*10 + j) {
                    printf("read error i=%d buf[j=%d]=%d != %d\n",i,j,buf[j],rank*1000+100+i*10+j);
                    nerrs++;
                }
        }
#endif
    }
    err = ncmpi_close(ncid); ERR
    MPI_Info_free(&info);
    free(buf);

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


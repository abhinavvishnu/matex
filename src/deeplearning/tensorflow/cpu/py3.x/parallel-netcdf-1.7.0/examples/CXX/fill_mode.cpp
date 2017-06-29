/*********************************************************************
 *
 *  Copyright (C) 2015, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *********************************************************************/
/* $Id: fill_mode.cpp 2012 2015-02-16 05:52:44Z wkliao $ */

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * This example shows how to use 
 * 1. NcmpiVar::setFill() to define the variable's fill value
 * 2. NcmpiVar::getFillModeParameters() to inquire the variable's fill mode
 *    information
 * 3. NcmpiVar::fillVar() to write two 2D 4-byte integer array in parallel.
 * It first defines a netCDF record variable of size NC_UNLIMITED * global_ny
 * where
 *    global_nx == (NX * number of MPI processes).
 * It then defines a netCDF fixed-size variable of size global_nx * global_ny
 * where
 *    global_ny == NY and
 *    global_nx == (NX * number of MPI processes).
 *
 *    To compile:
 *        mpicxx -O2 fill_mode.cpp -o fill_mode -lpnetcdf
 *
 * Example commands for MPI run and outputs from running ncmpidump on the
 * NC file produced by this example program:
 *
 *    % mpiexec -n 4 ./fill_mode /pvfs2/wkliao/testfile.nc
 *
 *    % ncmpidump /pvfs2/wkliao/testfile.nc
 *    netcdf testfile {
 *    // file format: CDF-5 (big variables)
 *    dimensions:
 *            REC_DIM = UNLIMITED ; // (2 currently)
 *            X = 16 ;
 *            Y = 3 ;
 *    variables:
 *            int rec_var(REC_DIM, X) ;
 *                    rec_var:_FillValue = -1 ;
 *            int fix_var(Y, X) ;
 *
 *    // global attributes:
 *                    :history = "Thu Feb 12 01:28:34 2015\n",
 *       "" ;
 *    data:
 *
 *    rec_var =
 *           0, 0, 0, _, 1, 1, 1, _, 2, 2, 2, _, 3, 3, 3, _,
 *           0, 0, 0, _, 1, 1, 1, _, 2, 2, 2, _, 3, 3, 3, _ ;
 *
 *    fix_var =
 *           0, 0, 0, _, 1, 1, 1, _, 2, 2, 2, _, 3, 3, 3, _,
 *           0, 0, 0, _, 1, 1, 1, _, 2, 2, 2, _, 3, 3, 3, _,
 *           0, 0, 0, _, 1, 1, 1, _, 2, 2, 2, _, 3, 3, 3, _ ;
 *    }
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>

#include <iostream>
using namespace std;

#include <string.h> /* strcpy() */
#include <unistd.h> /* getopt() */
#include <time.h>   /* time() localtime(), asctime() */

#include <pnetcdf>

using namespace PnetCDF;
using namespace PnetCDF::exceptions;

#define NY 3
#define NX 4

static void
usage(char *argv0)
{
    cerr <<
    "Usage: %s [-h] | [-q] [file_name]\n"
    "       [-h] Print help\n"
    "       [-q] Quiet mode (reports when fail)\n"
    "       [filename] output netCDF file name\n"
    << argv0;
}

int main(int argc, char** argv)
{
    extern int optind;
    char filename[128], str_att[128];
    int i, j, verbose=1, rank, nprocs, buf[NY][NX];
    MPI_Offset  global_ny, global_nx;

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
    else           strcpy(filename, "testfile.nc");

    try {
        /* create a new file for writing ------------------------------------*/
        NcmpiFile ncFile(MPI_COMM_WORLD, filename, NcmpiFile::replace,
                         NcmpiFile::classic5);

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
        MPI_Bcast(str_att, 128, MPI_CHAR, 0, MPI_COMM_WORLD);

        ncFile.putAtt(string("history"), string(str_att));

        /* define dimensions Y and X */
        vector<NcmpiDim> dimid(2);

        dimid[0] = ncFile.addDim("REC_DIM", NC_UNLIMITED);
        dimid[1] = ncFile.addDim("X", global_nx);

        /* define a 2D record variable of integer type */
        NcmpiVar rec_var = ncFile.addVar("rec_var", ncmpiInt, dimid);

        dimid[0] = ncFile.addDim("Y", global_ny);

        /* define a 2D fixed-size variable of integer type */
        NcmpiVar fix_var = ncFile.addVar("fix_var", ncmpiInt, dimid);

        /* set the variable's fill mode to NC_FILL with default fill value */
        fix_var.setFill(true);

        /* set a customized fill value -1 */
        int fill_value = -1;
        rec_var.setFill(true, &fill_value);

        /* now we are in data mode */
        vector<MPI_Offset> start(2), count(2);
        start[0] = 0;
        start[1] = NX * rank;
        count[0] = NY;
        count[1] = NX;
        if (verbose)
            printf("%d: start=%lld %lld count=%lld %lld\n",rank,
                   start[0],start[1],count[0],count[1]);

        /* do not write the variable in full */
        count[1]--;
        fix_var.putVar_all(start, count, &buf[0][0]);

        bool fillMode=true;
        fix_var.getFillModeParameters(fillMode, &fill_value);
        if (!fillMode)
            printf("Error: expecting fillMode to be false\n");
        if (fill_value != NC_FILL_INT)
            printf("Error: expecting fillMode to be %ld but got %d\n",NC_FILL_INT,fill_value);

        /* fill the 1st record of the record variable */
        start[0] = 0;
        rec_var.fillRec(start[0]);

        /* write to the 1st record */
        count[0] = 1;
        rec_var.putVar_all(start, count, &buf[0][0]);

        /* fill the 2nd record of the record variable */
        start[0] = 1;
        rec_var.fillRec(start[0]);

        /* write to the 2nd record */
        rec_var.putVar_all(start, count, &buf[0][0]);

        /* file is close implicitly */
    }
    catch(NcmpiException& e) {
       cout << e.what() << " error code=" << e.errorCode() << " Error!\n";
    }

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


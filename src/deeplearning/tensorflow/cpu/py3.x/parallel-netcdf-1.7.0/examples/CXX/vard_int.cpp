/*********************************************************************
 *
 *  Copyright (C) 2014, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *********************************************************************/
/* $Id: vard_int.cpp 2239 2015-12-18 18:21:09Z wkliao $ */

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * This example shows how to use the vard API ncmpi_put_vard() and
 * ncmpi_get_vard() to write and read 2D record and fixed-size variables.
 *
 *    To compile:
 *        mpicxx -O2 vard_int.cpp -o vard_int -lpnetcdf
 *
 * Example commands for MPI run and outputs from running ncmpidump on the
 * NC file produced by this example program:
 *
 *    % mpiexec -n 4 ./vard_int /pvfs2/wkliao/testfile.nc
 *
 * The expected results from the output file contents are:
 *
 *  % ncmpidump /pvfs2/wkliao/testfile.nc
 *    netcdf testfile {
 *    // file format: CDF-1
 *    dimensions:
 *           REC_DIM = UNLIMITED ; // (2 currently)
 *           X = 12 ;
 *           FIX_DIM = 2 ;
 *    variables:
 *           int rec_var(REC_DIM, X) ;
 *           int fix_var(FIX_DIM, X) ;
 *    data:
 *
 *     rec_var =
 *       0, 1, 2, 100, 101, 102, 200, 201, 202, 300, 301, 302,
 *       10, 11, 12, 110, 111, 112, 210, 211, 212, 310, 311, 312 ;
 *
 *     fix_var =
 *       0, 1, 2, 100, 101, 102, 200, 201, 202, 300, 301, 302,
 *       10, 11, 12, 110, 111, 112, 210, 211, 212, 310, 311, 312 ;
 *    }
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>

#include <iostream>
using namespace std;

#include <string.h> /* strcpy() */
#include <unistd.h> /* getopt() */
#include <pnetcdf>

using namespace PnetCDF;
using namespace PnetCDF::exceptions;

#define NY 2
#define NX 3

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
    char filename[128];
    int i, j, verbose=1;
    int          rank, nprocs, array_of_blocklengths[2], buf[NY][NX];
    int          array_of_sizes[2], array_of_subsizes[2], array_of_starts[2];
    MPI_Offset   recsize, bufcount, len;
    MPI_Aint     array_of_displacements[2];
    MPI_Datatype buftype, rec_filetype, fix_filetype;

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

    vector <MPI_Offset> start(2), count(2);
    start[0] = 0; start[1] = NX*rank;
    count[0] = 2; count[1] = NX;
    if (verbose)
        printf("%d: start=%lld %lld count=%lld %lld\n",rank,
               start[0],start[1],count[0],count[1]);

    /* create a file type for the fixed-size variable */
    array_of_sizes[0]    = 2;
    array_of_sizes[1]    = NX*nprocs;
    array_of_subsizes[0] = count[0];
    array_of_subsizes[1] = count[1];
    array_of_starts[0]   = start[0];
    array_of_starts[1]   = start[1];
    MPI_Type_create_subarray(2, array_of_sizes, array_of_subsizes,
                             array_of_starts, MPI_ORDER_C,
                             MPI_INT, &fix_filetype);
    MPI_Type_commit(&fix_filetype);

    buftype = MPI_INT;
    bufcount = count[0] * count[1];

    try {
        /* create a new file for writing ------------------------------------*/
        NcmpiFile nc(MPI_COMM_WORLD, filename, NcmpiFile::replace);

        /* define 3 dimensions */
        vector<NcmpiDim> recdimid(2);
        recdimid[0] = nc.addDim("REC_DIM", NC_UNLIMITED);
        recdimid[1] = nc.addDim("X", NX*nprocs);

        vector<NcmpiDim> fixdimid(2);
        fixdimid[0] = nc.addDim("FIX_DIM", 2);
        fixdimid[1] = recdimid[1];

        /* define a variable of size (NZ * nprocs) * NY */
        NcmpiVar var0 = nc.addVar("rec_var", ncmpiInt, recdimid);

        /* define a variable of size NY * (NX * nprocs) */
        NcmpiVar var1 = nc.addVar("fix_var", ncmpiInt, fixdimid);

        nc.enddef();

        /* create a file type for the record variable */
        recsize = nc.getRecSize();
        for (i=0; i<count[0]; i++) {
            array_of_blocklengths[i] = count[1];
            array_of_displacements[i] = start[1]*sizeof(int) + recsize * i;
        }
        MPI_Type_create_hindexed(2, array_of_blocklengths,
                        array_of_displacements, MPI_INT, &rec_filetype);
        MPI_Type_commit(&rec_filetype);

        /* initialize the contents of the array */
        for (j=0; j<NY; j++) for (i=0; i<NX; i++)
            buf[j][i] = rank*100 + j*10 + i;

        /* write the record variable */
        var0.putVard_all(rec_filetype, buf, bufcount, buftype);

        /* check if the number of records changed to 2 */
        len = recdimid[0].getSize();
        if (len != 2)
            cout << "Error: number of records should be 2 but got " << len << "\n";

        /* write the fixed-size variable */
        var1.putVard_all(fix_filetype, buf, bufcount, buftype);

        /* file is close implicitly */
    }
    catch(NcmpiException& e) {
       cout << e.what() << " error code=" << e.errorCode() << " Error!\n";
    }

    try {
        /* open file for reading -----------------------------------------*/
        NcmpiFile nc(MPI_COMM_WORLD, filename, NcmpiFile::read);

        NcmpiVar var0 = nc.getVar("rec_var");
        NcmpiVar var1 = nc.getVar("fix_var");

        /* read the record variable */
        var0.getVard_all(rec_filetype, buf, bufcount, buftype);

        /* read the fixed-size variable */
        var1.getVard_all(fix_filetype, buf, bufcount, buftype);
    }
    catch(NcmpiException& e) {
       cout << e.what() << " error code=" << e.errorCode() << " Error!\n";
    }

    MPI_Type_free(&rec_filetype);
    MPI_Type_free(&fix_filetype);

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


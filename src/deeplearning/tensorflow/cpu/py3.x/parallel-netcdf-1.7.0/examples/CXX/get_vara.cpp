/*********************************************************************
 *
 *  Copyright (C) 2014, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *********************************************************************/
/* $Id: get_vara.cpp 2245 2015-12-20 18:39:52Z wkliao $ */

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * This example is the read counterpart of example put_vara.cpp. It shows how to
 * use NcmpiVar::getVar_all() to read a 2D 4-byte integer array in parallel.
 * It also reads a global attribute and two attribute of variable named "var".
 * The data partitioning pattern is a column-wise partitioning across all
 * processes. Each process reads a subarray of size local_ny * local_nx.
 *
 *    To compile:
 *        mpicxx -O2 get_vara.cpp -o get_vara -lpnetcdf
 *
 * Input file is the output file produced by put_vara.cpp. Here is the CDL
 * dumped from running ncmpidump.
 *
 *    % ncmpidump /pvfs2/wkliao/testfile.nc
 *    netcdf testfile {
 *    // file format: CDF-1
 *    dimensions:
 *            y = 10 ;
 *            x = 16 ;
 *    variables:
 *            int var(y, x) ;
 *                var:str_att_name = "example attribute of type text." ;
 *                var:float_att_name = 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f ;
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
 *
 * Example command for MPI run:
 *
 *    % mpiexec -n 4 ./get_vara /pvfs2/wkliao/testfile.nc
 *
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

static void
usage(char *argv0)
{
    cerr <<
    "Usage: %s [-h] | [-q] [file_name]\n"
    "       [-h] Print help\n"
    "       [-q] Quiet mode (reports when fail)\n"
    "       [filename] input netCDF file name\n"
    << argv0;
}

int main(int argc, char** argv)
{
    extern int optind;
    char filename[128], str_att[NC_MAX_NAME];
    int i, rank, nprocs, err, verbose=1;
    MPI_Offset len, global_ny, global_nx, local_ny, local_nx;

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
        /* open an existing file for reading --------------------------------*/
        NcmpiFile ncFile(MPI_COMM_WORLD, filename, NcmpiFile::read);

        // Check the format.
        if (ncFile.getFormat() != NcmpiFile::classic5) {
            cout << "unexpected file format"<<endl;
            throw NcmpiException("read Error ",__FILE__,__LINE__);
        }

        /* get global attribute named "history" */
        NcmpiGroupAtt att = ncFile.getAtt("history");
        att.getValues(str_att);
        len = att.getAttLength();
        str_att[len] = '\0'; /* add a NULL char at the end */
        if (rank == 0 && verbose)
            printf("global attribute \"history\" of text: %s\n",str_att);

        /* get dimension IDs for dimensions Y and X */
        NcmpiDim yDim = ncFile.getDim("Y");
        NcmpiDim xDim = ncFile.getDim("X");

        /* get dimension lengths for dimensions Y and X */
        global_ny = yDim.getSize();
        global_nx = xDim.getSize();

        /* get the variable ID of a 2D variable of integer type */
        NcmpiVar var = ncFile.getVar("var");

        /* get variable's attribute named "str_att_name" */
        NcmpiVarAtt vatt = var.getAtt("str_att_name");
        vatt.getValues(str_att);
        len = vatt.getAttLength();
        str_att[len] = '\0'; /* add a NULL char at the end */
        if (rank == 0 && verbose)
            printf("variable attribute \"str_att_name\" of type text = \"%s\"\n",
                   str_att);

        /* get the length of variable's attribute named "float_att_name" */
        NcmpiVarAtt flt_vatt = var.getAtt("float_att_name");
        len = flt_vatt.getAttLength();

        /* get attribute contents */
        float *float_att = (float*) malloc(len * sizeof(float));
        flt_vatt.getValues(float_att);

        /* the local array size */
        local_ny = global_ny;
        local_nx = global_nx / nprocs;
        int *buf = (int*) malloc(local_nx * local_ny * sizeof(int));

        /* prepare reading subarray */
        vector<MPI_Offset> start(2), count(2);
        start[0] = 0;
        start[1] = local_nx * rank;
        count[0] = local_ny;
        count[1] = local_nx;

        /* read a subarray in collective mode */

        // var.getVar_all(start, count, &buf[0][0]);
        var.getVar_all(start, count, buf);

        free(buf);
        free(float_att);
        /* file is close implicitly */
    }
    catch(NcmpiException& e) {
       cout << e.what() << " error code=" << e.errorCode() << " Error!\n";
    }

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


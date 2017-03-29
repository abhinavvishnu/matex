/*
 *  Copyright (C) 2014, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id: pnetcdf_version.c 2264 2015-12-22 15:42:59Z wkliao $ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h> /* getopt */

typedef enum {
    Version_number = 0,
    Date           = 1, 
    Patches        = 2,
    Configure_args = 3,
    Compilers      = 4,
    LastField
} fields;

/*
 *   pnetcdf_version - Report on the PnetCDF version
 */

/*---< usage() >-------------------------------------------------------------*/
static void usage(char *argv0) {
    char *help =
        "Usage: %s [switches]\n"
        "       -v  : version number\n"
        "       -d  : release date\n"
        "       -c  : configure arguments used to build PnetCDF\n"
        "       -b  : MPI compilers used\n"
        "       -h  : print this help (available command-line options)\n";
    fprintf(stderr, help, argv0);
    exit(-1);
}

/*----< main() >-------------------------------------------------------------*/
int main( int argc, char *argv[] )
{
           int     opt;

    int i, flags[10];

    if (argc <= 1) {
        /* Show all values */
        for (i=0; i<LastField; i++) flags[i] = 1;
    }
    else {
        /* Show only requested values */
        for (i=0; i<LastField; i++) flags[i] = 0;
    }

    while ( (opt=getopt(argc,argv,"vdcbh"))!= EOF) {
        switch (opt) {
            case 'v': flags[Version_number] = 1;
                      break;
            case 'd': flags[Date] = 1;
                      break;
            case 'c': flags[Configure_args] = 1;
                      break;
            case 'b': flags[Compilers] = 1;
                      break;
            case 'h':
            default: usage(argv[0]);
                      break;
        }
    }

    /* Print out the information, one item per line */
    if (flags[Version_number]) {
        printf( "PnetCDF Version:    \t%s\n", PNETCDF_VERSION);
    }
    if (flags[Date]) {
        printf( "PnetCDF Release date:\t%s\n", PNETCDF_RELEASE_DATE);
    }
    if (flags[Configure_args]) {
        printf( "PnetCDF configure: \t%s\n", CONFIGURE_ARGS_CLEAN);
    }
    if (flags[Compilers]) {
        if (strcmp(CFLAGS, ""))
            printf( "MPICC:  %s %s\n", MPICC, CFLAGS);
        else
            printf( "MPICC:  %s\n", MPICC);
#ifdef MPICXX
        if (strcmp(CXXFLAGS, ""))
            printf( "MPICXX: %s %s\n", MPICXX, CXXFLAGS);
        else
            printf( "MPICXX: %s\n", MPICXX);
#endif
#ifdef MPIF77
        if (strcmp(F77FLAGS, ""))
            printf( "MPIF77: %s %s\n", MPIF77, F77FLAGS);
        else
            printf( "MPIF77: %s\n", MPIF77);
#endif
#ifdef MPIF90
        if (strcmp(F90FLAGS, ""))
            printf( "MPIF90: %s %s\n", MPIF90, F90FLAGS);
        else
            printf( "MPIF90: %s\n", MPIF90);
#endif
    }

    return 0;
}


/*
 *  Copyright (C) 2013, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id: subfile.h 2021 2015-04-09 20:29:40Z wkliao $ */
#ifndef SUBFILE_H
#define SUBFILE_H

#include "pnetcdf.h"
#include "nc.h"
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "macro.h"

/* structure for storing access info of this process's request
   to the subfiles on all other processes, and vice-versa. used
   as array of structures indexed by subfile index. */
typedef struct {
    MPI_Offset *start;
    MPI_Offset *count;
    MPI_Offset *start_org;
} NC_subfile_access;

#define CEIL(x) ( (x - (int)x)==0 ? (int)x : (int)x+1 )
#define FLOOR(x) ( (x - (int)x)==0 ? (int)x : (int)x-1 )
#define ROUND(x) ( x >= 0 ? (int)(x+0.5) : (int)(x-0.5) )
#define ABS(a) (((a) < 0) ? -(a) : (a))

#define TEST_HANDLE_ERR(status)                                        \
    if ((status) != NC_NOERR) {                                        \
        printf("Error at file %s line %d (%s)\n", __FILE__, __LINE__,  \
               ncmpi_strerror((status)) );                             \
        return status;                                                 \
    }

extern int ncmpii_subfile_create(NC *ncp, int *ncidp);

extern int ncmpii_subfile_open(NC *ncp, int *ncidp);

extern int ncmpii_subfile_close(NC *ncp);

extern int ncmpii_subfile_partition(NC *ncp, int *ncidp);

extern int ncmpii_subfile_getput_vars(NC *ncp, NC_var *varp, const MPI_Offset start[],
                               const MPI_Offset count[], const MPI_Offset  stride[],
                               void *buf, MPI_Offset bufcount,
                               MPI_Datatype buftype, int rw_flag, int io_method);

#endif /* end of SUBFILE_H */

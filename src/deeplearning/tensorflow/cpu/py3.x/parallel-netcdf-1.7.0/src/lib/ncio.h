/*
 *  Copyright (C) 2003, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id: ncio.h 2017 2015-04-01 21:27:21Z wkliao $ */

#ifndef _NCIO_H_
#define _NCIO_H_

#include <stddef.h>       /* size_t */
#include <sys/types.h>    /* off_t */
#include <mpi.h>
#include "pnetcdf.h"

#define MPI_COLLECTIVE_FH 2
#define MPI_INDEPENDENT_FH 8

#define NC_collectiveFhOpened(nciop) \
    (fIsSet(nciop->mpioflags, MPI_COLLECTIVE_FH))

#define NC_independentFhOpened(nciop) \
    (fIsSet(nciop->mpioflags, MPI_INDEPENDENT_FH))

#define set_NC_collectiveFh(nciop) \
    fSet((nciop)->mpioflags, MPI_COLLECTIVE_FH)

#define set_NC_independentFh(nciop) \
    fSet((nciop)->mpioflags, MPI_INDEPENDENT_FH)


/*
 * I/O hints used by PnetCDF, but MPI-IO
 */
typedef struct {
    MPI_Offset h_align; /* file alignment size for header */
    MPI_Offset v_align; /* file alignment size for each fixed variable */
    MPI_Offset r_align; /* file alignment size for record variable section */
    MPI_Offset header_read_chunk_size;
#ifdef ENABLE_SUBFILING
    int subfile_mode;
    int num_subfiles;
#endif
} nc_hints;

/*
 * netcdf i/o abstraction
 */
typedef struct {
    /*
     * A copy of the ioflags argument passed in to ncmpiio_open()
     * or ncmpiio_create().
     */
    int ioflags;

    /*
     * The file descriptor of the netcdf file.
     * This gets handed to the user as the netcdf id.
     */
    int fd;

    /*
     * The MPI File handle and the communicator
     */
    MPI_File collective_fh;
    MPI_File independent_fh;
    MPI_Comm comm;
    MPI_Info mpiinfo;

    int mpiomode;        /* mode used in MPI_File_open */
    int mpioflags;       /* MPI_COLLECTIVE_FH or MPI_INDEPENDENT_FH */
    int striping_unit;   /* file stripe size of the file */

    nc_hints hints;      /* I/O hints used by PnetCDF only */

    /*
     * A copy of the 'path' argument passed in to ncmpiio_open()
     * or ncmpiio_create(). Used by ncabort() to remove (unlink)
     * the file and by error messages.
     */
    const char *path;

    MPI_Offset put_size;  /* amount of writes committed so far in bytes */
    MPI_Offset get_size;  /* amount of reads  committed so far in bytes */
} ncio;

extern ncio *
ncmpiio_new(const char *path, int ioflags);

extern void
ncmpiio_free(ncio *nciop);

extern int
ncmpiio_close(ncio *nciop, int doUnlink);

#endif /* _NCIO_H_ */

/*
 *  Copyright (C) 2003, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id: mpincio.c 2323 2016-02-26 23:04:46Z wkliao $ */

#if HAVE_CONFIG_H
# include "ncconfig.h"
#endif

#ifdef HAVE_ACCESS
#include <unistd.h>  /* access() */
#endif
#include <assert.h>
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#include <stdio.h>
#include <errno.h>
#include <string.h>  /* strchr() */
#ifdef _MSC_VER /* Microsoft Compilers */
#include <io.h>
#else
#include <unistd.h>
#endif

/* #define INSTRUMENT 1 */
#ifdef INSTRUMENT /* debugging */
#undef NDEBUG
#include <stdio.h>
#include "instr.h"
#endif

#include <mpi.h>

#include "nc.h"
#include "ncio.h"
#include "fbits.h"
#include "rnd.h"
#include "macro.h"

#if !defined(NDEBUG) && !defined(X_INT_MAX)
#define  X_INT_MAX INT_MAX
#endif

#if 0 /* !defined(NDEBUG) && !defined(X_ALIGN) */
#define  X_ALIGN 4
#else
#undef X_ALIGN
#endif

#define MAX_NC_ID 1024

static unsigned char IDalloc[MAX_NC_ID];

inline void
ncmpiio_free(ncio *nciop) {
    if (nciop != NULL) {
#ifdef HAVE_MPI_INFO_FREE
        if (nciop->mpiinfo != MPI_INFO_NULL)
            MPI_Info_free(&(nciop->mpiinfo));
#endif
        if (nciop->comm != MPI_COMM_NULL)
            MPI_Comm_free(&(nciop->comm));

        NCI_Free(nciop);
    }
}

inline ncio *
ncmpiio_new(const char *path, int ioflags)
{
    size_t sz_ncio = M_RNDUP(sizeof(ncio));
    size_t sz_path = M_RNDUP(strlen(path) +1);
    ncio *nciop;

    nciop = (ncio *) NCI_Malloc(sz_ncio + sz_path);
    if (nciop == NULL) return NULL;

    nciop->ioflags  = ioflags;
    nciop->comm     = MPI_COMM_NULL;
    nciop->mpiinfo  = MPI_INFO_NULL;
    nciop->put_size = 0;
    nciop->get_size = 0;

    nciop->path = (char *) ((char *)nciop + sz_ncio);
    (void) strcpy((char *)nciop->path, path);

    return nciop;
}

/*----< ncmpiio_extract_hints() >--------------------------------------------*/
/* this is where the I/O hints designated to pnetcdf are extracted */
static
void ncmpiio_extract_hints(ncio     *nciop,
                           MPI_Info  info)
{
    /* value 0 indicates the hint is not set */
    nciop->hints.h_align                = 0;
    nciop->hints.v_align                = 0;
    nciop->hints.r_align                = 0;
    nciop->hints.header_read_chunk_size = 0;
#ifdef ENABLE_SUBFILING
    nciop->hints.subfile_mode           = 1;
    nciop->hints.num_subfiles           = 0;
#endif

    /* extract NC hints */
    if (info != MPI_INFO_NULL) {
        char value[MPI_MAX_INFO_VAL];
        int  flag;

        MPI_Info_get(info, "nc_header_align_size", MPI_MAX_INFO_VAL-1, value,
                     &flag);
        if (flag) nciop->hints.h_align = atoll(value);

        MPI_Info_get(info, "nc_var_align_size",    MPI_MAX_INFO_VAL-1, value,
                     &flag);
        if (flag) nciop->hints.v_align = atoll(value);

        MPI_Info_get(info, "nc_record_align_size", MPI_MAX_INFO_VAL-1,
                     value, &flag);
        if (flag) nciop->hints.r_align = atoll(value);

        MPI_Info_get(info, "nc_header_read_chunk_size", MPI_MAX_INFO_VAL-1,
                     value, &flag);
        if (flag) nciop->hints.header_read_chunk_size = atoll(value);

#ifdef ENABLE_SUBFILING
        MPI_Info_get(info, "pnetcdf_subfiling", MPI_MAX_INFO_VAL-1,
                     value, &flag);
        if (flag && strcasecmp(value, "disable") == 0)
            nciop->hints.subfile_mode = 0;

        MPI_Info_get(info, "nc_num_subfiles", MPI_MAX_INFO_VAL-1,
                     value, &flag);
        if (flag) nciop->hints.num_subfiles = atoi(value);
#endif

        /* nc_header_align_size, nc_var_align_size, and nciop->hints.r_align
         * take effect when a file is created or opened and later adding more
         * header or variable data */

        if (nciop->hints.h_align < 0)
            nciop->hints.h_align = 0;
        if (nciop->hints.v_align < 0)
            nciop->hints.v_align = 0;
        if (nciop->hints.r_align < 0)
            nciop->hints.r_align = 0;
        if (nciop->hints.header_read_chunk_size < 0)
            nciop->hints.header_read_chunk_size = 0;
#ifdef ENABLE_SUBFILING
        if (nciop->hints.num_subfiles < 0)
            nciop->hints.num_subfiles = 0;
        /* override subfile hints if env var is set */
        char *num_sf_env;
        num_sf_env = getenv("NC_NUM_SUBFILES");
        if (num_sf_env != NULL)
            nciop->hints.num_subfiles = atoi(num_sf_env);
        if (nciop->hints.subfile_mode == 0)
            nciop->hints.num_subfiles = 0;
#endif
    }
}

/*----< ncmpiio_create() >---------------------------------------------------*/
int
ncmpiio_create(MPI_Comm     comm,
               const char  *path,
               int          ioflags,
               MPI_Info     info,
               NC          *ncp)
{
    ncio *nciop;
    int i, rank, mpireturn, err;
    int mpiomode = MPI_MODE_RDWR | MPI_MODE_CREATE;

    assert(path != NULL);
    assert(*path != '\0');
    /* checking path consistency is expected done in MPI-IO */

    MPI_Comm_rank(comm, &rank);

    /* NC_CLOBBER is the default mode, even if it is not used in cmode */
    if (fIsSet(ioflags, NC_NOCLOBBER)) {
        /* check if file exists: NetCDF requires NC_EEXIST returned if the file
         * already exists and NC_NOCLOBBER mode is used in create
         */
#ifdef HAVE_ACCESS
        int file_exist;
        /* if access() is available, use it to check whether file already exists
         * rank 0 calls access() and broadcasts file_exist */
        if (rank == 0) {
            /* remove the file system type prefix name if there is any.
             * For example, path=="lustre:/home/foo/testfile.nc",
             * use "/home/foo/testfile.nc" when calling access()
             */
            char *filename = strchr(path, ':');
            if (filename == NULL) /* no prefix */
                filename = (char*)path;
            else
                filename++;

            if (access(filename, F_OK) == 0) file_exist = 1;
            else                             file_exist = 0;
        }
        TRACE_COMM(MPI_Bcast)(&file_exist, 1, MPI_INT, 0, comm);
        if (file_exist) DEBUG_RETURN_ERROR(NC_EEXIST)
#else
        /* use MPI_MODE_EXCL mode in MPI_File_open and check returned error */
        fSet(mpiomode, MPI_MODE_EXCL);
#endif
    }
    else { /* NC_CLOBBER is the default mode in create */
        /* rank 0 deletes the file and ignores error code for file not exist
         * Note calling MPI_File_set_size is expensive as it calls truncate()
         */
        if (rank == 0) {
#ifdef HAVE_UNLINK
            err = unlink(path);
            if (err < 0 && errno != ENOENT) /* ignore ENOENT: file not exist */
                DEBUG_ASSIGN_ERROR(err, NC_EFILE) /* other error */
            else
                err = NC_NOERR;
#else
            err = NC_NOERR;
            TRACE_IO(MPI_File_delete)((char*)path, MPI_INFO_NULL);
            if (mpireturn != MPI_SUCCESS) {
                int errorclass;
                MPI_Error_class(mpireturn, &errorclass);
                if (errorclass != MPI_ERR_NO_SUCH_FILE) /* ignore this error */
                    err = ncmpii_handle_error(mpireturn, "MPI_File_delete");
            }
#endif
        }
        /* all processes must wait here until file deletion is completed */
        TRACE_COMM(MPI_Bcast)(&err, 1, MPI_INT, 0, comm);
        if (err != NC_NOERR) return err;
    }

    /* ignore if NC_NOWRITE set by user */
    fSet(ioflags, NC_WRITE);

    nciop = ncmpiio_new(path, ioflags); /* allocate buffer */
    if (nciop == NULL)
        DEBUG_RETURN_ERROR(NC_ENOMEM)

    nciop->mpiomode  = MPI_MODE_RDWR;
    nciop->mpioflags = 0;

    /* extract MPI-IO hints */
    ncmpiio_extract_hints(nciop, info);

    TRACE_IO(MPI_File_open)(comm, (char *)path, mpiomode, info,
                            &nciop->collective_fh);
    if (mpireturn != MPI_SUCCESS) {
        ncmpiio_free(nciop);
#ifndef HAVE_ACCESS
        if (fIsSet(ioflags, NC_NOCLOBBER)) {
            /* This is the case when NC_NOCLOBBER is used in file creation and
             * function access() is not available. MPI_MODE_EXCL is set in open
             * mode. When MPI_MODE_EXCL is used and the file already exists,
             * MPI-IO should return error class MPI_ERR_FILE_EXISTS. But, some
             * MPI-IO implementations (older ROMIO) do not correctly return
             * this error class. In this case, we can do the followings: check
             * errno to see if it set to EEXIST. Note usually rank 0 makes the
             * file open call and can be the only one having errno set.
             */
            TRACE_COMM(MPI_Bcast)(&errno, 1, MPI_INT, 0, comm);
            if (errno == EEXIST) DEBUG_RETURN_ERROR(NC_EEXIST)
        }
#endif
        return ncmpii_handle_error(mpireturn, "MPI_File_open");
        /* for NC_NOCLOBBER, MPI_MODE_EXCL was added to mpiomode. If the file
         * already exists, MPI-IO should return error class MPI_ERR_FILE_EXISTS
         * which PnetCDF will return error code NC_EEXIST. This is checked
         * inside of ncmpii_handle_error()
         */
    }

    for (i=0; i<MAX_NC_ID; i++)
        if (IDalloc[i] == 0)
            break;

    if (i == MAX_NC_ID) {
        ncmpiio_free(nciop);
        DEBUG_RETURN_ERROR(NC_ENFILE)
    }

    *((int *)&nciop->fd) = i;
    IDalloc[i] = 1;

    /* collective I/O mode is the default mode */
    set_NC_collectiveFh(nciop);

    /* duplicate communicator as user may free it later */
    mpireturn = MPI_Comm_dup(comm, &(nciop->comm));
    if (mpireturn != MPI_SUCCESS)
        return ncmpii_handle_error(mpireturn, "MPI_Comm_dup");

    /* get the file info actually used by MPI-IO (maybe alter user's info) */
    MPI_File_get_info(nciop->collective_fh, &nciop->mpiinfo);

    ncp->nciop = nciop;
    return NC_NOERR;
}

/*----< ncmpiio_open() >-----------------------------------------------------*/
int
ncmpiio_open(MPI_Comm     comm,
             const char  *path,
             int          ioflags,
             MPI_Info     info,
             NC          *ncp)
{
    ncio *nciop;
    int i, mpireturn;
    int mpiomode = fIsSet(ioflags, NC_WRITE) ? MPI_MODE_RDWR : MPI_MODE_RDONLY;

    assert(path != NULL);
    assert(*path != '\0');
    /* checking path consistency is expected done in MPI-IO */

    /* When open an non-existing file for read, we can either call access() to
     * check and return error code NC_ENOENT, or call MPI_File_open and expect
     * error class MPI_ERR_NO_SUCH_FILE. For now, we let MPI-IO to check.
     */
#if 0 && defined(HAVE_ACCESS)
    if (mpiomode == MPI_MODE_RDONLY) { /* file should already exit */
        int rank, file_exist;
        MPI_Comm_rank(comm, &rank);
        if (rank == 0) {
            if (access(path, F_OK) == 0) file_exist = 1;
            else                         file_exist = 0;
        }
        TRACE_COMM(MPI_Bcast)(&file_exist, 1, MPI_INT, 0, comm);
        if (!file_exist) DEBUG_RETURN_ERROR(NC_ENOENT)
    }
#endif

    nciop = ncmpiio_new(path, ioflags);
    if (nciop == NULL)
        DEBUG_RETURN_ERROR(NC_ENOMEM)

    nciop->mpiomode  = mpiomode;
    nciop->mpioflags = 0;

    /* extract MPI-IO hints */
    ncmpiio_extract_hints(nciop, info);

    TRACE_IO(MPI_File_open)(comm, (char *)path, mpiomode,
                            info, &nciop->collective_fh);
    if (mpireturn != MPI_SUCCESS) {
        ncmpiio_free(nciop);
        return ncmpii_handle_error(mpireturn, "MPI_File_open");
    }

    for (i=0; i<MAX_NC_ID && IDalloc[i] != 0; i++);
    if (i == MAX_NC_ID) {
        ncmpiio_free(nciop);
        DEBUG_RETURN_ERROR(NC_ENFILE)
    }
    *((int *)&nciop->fd) = i;
    IDalloc[i] = 1;

    set_NC_collectiveFh(nciop);

    /* duplicate communicator as user may free it later */
    mpireturn = MPI_Comm_dup(comm, &(nciop->comm));
    if (mpireturn != MPI_SUCCESS)
        return ncmpii_handle_error(mpireturn, "MPI_Comm_dup");

    /* get the file info used by MPI-IO */
    MPI_File_get_info(nciop->collective_fh, &nciop->mpiinfo);

    ncp->nciop = nciop;
    return NC_NOERR;
}

/*----< ncmpiio_sync() >-----------------------------------------------------*/
/* This function must be called collectively, no matter if it is in collective
 * or independent data mode.
 */
inline int
ncmpiio_sync(ncio *nciop) {
#ifndef DISABLE_FILE_SYNC
    int mpireturn;

    if (NC_independentFhOpened(nciop)) {
        TRACE_IO(MPI_File_sync)(nciop->independent_fh);
        if (mpireturn != MPI_SUCCESS)
            return ncmpii_handle_error(mpireturn, "MPI_File_sync");
    }
    if (NC_collectiveFhOpened(nciop)) {
        TRACE_IO(MPI_File_sync)(nciop->collective_fh);
        if (mpireturn != MPI_SUCCESS)
            return ncmpii_handle_error(mpireturn, "MPI_File_sync");
    }
    TRACE_COMM(MPI_Barrier)(nciop->comm);
#endif
    return NC_NOERR;
}

/*----< ncmpiio_close() >----------------------------------------------------*/
int
ncmpiio_close(ncio *nciop, int doUnlink) {
    int mpireturn;

    if (nciop == NULL) /* this should never occur */
        DEBUG_RETURN_ERROR(NC_EINVAL)

    if (NC_independentFhOpened(nciop)) {
        TRACE_IO(MPI_File_close)(&(nciop->independent_fh));
        if (mpireturn != MPI_SUCCESS)
            return ncmpii_handle_error(mpireturn, "MPI_File_close");
    }

    if (NC_collectiveFhOpened(nciop)) {
        TRACE_IO(MPI_File_close)(&(nciop->collective_fh));
        if (mpireturn != MPI_SUCCESS)
            return ncmpii_handle_error(mpireturn, "MPI_File_close");
    }
    IDalloc[*((int *)&nciop->fd)] = 0;

    if (doUnlink) {
        TRACE_IO(MPI_File_delete)((char *)nciop->path, nciop->mpiinfo);
        if (mpireturn != MPI_SUCCESS)
            return ncmpii_handle_error(mpireturn, "MPI_File_delete");
    }
    ncmpiio_free(nciop);

    return NC_NOERR;
}

/*----< ncmpiio_move() >-----------------------------------------------------*/
int
ncmpiio_move(ncio *const nciop,
             MPI_Offset  to,
             MPI_Offset  from,
             MPI_Offset  nbytes)
{
    int rank, nprocs, bufcount, mpireturn, err, status=NC_NOERR, min_st;
    void *buf;
    int chunk_size = 1048576; /* move 1 MB per process at a time */
    MPI_Status mpistatus;

    MPI_Comm_size(nciop->comm, &nprocs);
    MPI_Comm_rank(nciop->comm, &rank);

    /* if the file striping unit size is known (obtained from MPI-IO), then
     * we use that instead of 1 MB */
    if (nciop->striping_unit > 0) chunk_size = nciop->striping_unit;

    buf = NCI_Malloc((size_t)chunk_size);
    if (buf == NULL) DEBUG_RETURN_ERROR(NC_ENOMEM)

    /* make fileview entire file visible */
    TRACE_IO(MPI_File_set_view)(nciop->collective_fh, 0, MPI_BYTE, MPI_BYTE,
                                "native", MPI_INFO_NULL);

    /* move the variable starting from its tail toward its beginning */
    while (nbytes > 0) {
        /* calculate how much to move at each time */
        bufcount = chunk_size;
        if (nbytes < (MPI_Offset)nprocs * chunk_size) {
            /* handle the last group of chunks */
            MPI_Offset rem_chunks = nbytes / chunk_size;
            if (rank > rem_chunks) /* these processes do not read/write */
                bufcount = 0;
            else if (rank == rem_chunks) /* this process reads/writes less */
                bufcount = (int)(nbytes % chunk_size);
            nbytes = 0;
        }
        else {
            nbytes -= chunk_size*nprocs;
        }

        /* read the original data @ from+nbytes+rank*chunk_size */
        TRACE_IO(MPI_File_read_at_all)(nciop->collective_fh,
                                       from+nbytes+rank*chunk_size,
                                       buf, bufcount, MPI_BYTE, &mpistatus);
        if (mpireturn != MPI_SUCCESS) {
            err = ncmpii_handle_error(mpireturn, "MPI_File_read_at_all");
            if (err == NC_EFILE) DEBUG_ASSIGN_ERROR(status, NC_EREAD)
        }
        else {
            int get_size;
            MPI_Get_count(&mpistatus, MPI_BYTE, &get_size);
            nciop->get_size += get_size;
        }

        /* MPI_Barrier(nciop->comm); */
        /* important, in case new region overlaps old region */
        TRACE_COMM(MPI_Allreduce)(&status, &min_st, 1, MPI_INT, MPI_MIN, nciop->comm);
        status = min_st;
        if (status != NC_NOERR) break;

        /* write to new location @ to+nbytes+rank*chunk_size */
        TRACE_IO(MPI_File_write_at_all)(nciop->collective_fh,
                                        to+nbytes+rank*chunk_size,
                                        buf, bufcount, MPI_BYTE, &mpistatus);
        if (mpireturn != MPI_SUCCESS) {
            err = ncmpii_handle_error(mpireturn, "MPI_File_write_at_all");
            if (err == NC_EFILE) DEBUG_ASSIGN_ERROR(status, NC_EWRITE)
        }
        else {
            int put_size;
            MPI_Get_count(&mpistatus, MPI_BYTE, &put_size);
            nciop->put_size += put_size;
        }
        TRACE_COMM(MPI_Allreduce)(&status, &min_st, 1, MPI_INT, MPI_MIN, nciop->comm);
        status = min_st;
        if (status != NC_NOERR) break;
    }
    NCI_Free(buf);
    return status;
}

/*----< ncmpiio_move_fixed_vars() >-------------------------------------------*/
/* move one fixed variable at a time, only when the new begin > old begin */
int
ncmpiio_move_fixed_vars(NC *ncp,
                        NC *old)
{
    int i, err, status=NC_NOERR;

    /* move starting from the last fixed variable */
    for (i=old->vars.ndefined-1; i>=0; i--) {
        if (IS_RECVAR(old->vars.value[i])) continue;

        MPI_Offset from = old->vars.value[i]->begin;
        MPI_Offset to   = ncp->vars.value[i]->begin;
        if (to > from) {
            err = ncmpiio_move(ncp->nciop, to, from, ncp->vars.value[i]->len);
            if (status == NC_NOERR) status = err;
        }
    }
    return status;
}

int ncmpiio_get_hint(NC *ncp, char *key, char *value, int *flag)
{
    MPI_Info info;

    /* info hints can come from the file system but can also come from
     * user-specified hints.  the MPI implementation probably should
     * merge the two, but some implementations not only ignore hints
     * they don't understand, but also fail to incorporate those hints
     * into the info struct (this is unfortunate for us, but entirely
     * standards compliant).
     *
     * Our policy will be to use the implementation's info first
     * (perhaps the implementation knows something about the underlying
     * file system), and then consult user-supplied hints should we not
     * find the hint in the info associated with the MPI file descriptor
     */

    /* first check the hint from the MPI library ... */
    MPI_File_get_info(ncp->nciop->collective_fh, &info);
    if (info != MPI_INFO_NULL)
        MPI_Info_get(info, key, MPI_MAX_INFO_VAL-1, value, flag);
    if (*flag == 0)  {
        /* ... then check the hint passed in through ncmpi_create */
        if (ncp->nciop->mpiinfo != MPI_INFO_NULL) {
            MPI_Info_get(ncp->nciop->mpiinfo, key,
                    MPI_MAX_INFO_VAL-1, value, flag);
        }
    }
#ifdef HAVE_MPI_INFO_FREE
    if (info != MPI_INFO_NULL)
        MPI_Info_free(&info);
#endif

    return 0;
}

/*
 * Local variables:
 *  c-indent-level: 4
 *  c-basic-offset: 4
 * End:
 *
 * vim: ts=8 sts=4 sw=4 expandtab
 */

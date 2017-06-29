/*
 *  Copyright (C) 2003, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id: mpinetcdf.c 2322 2016-02-26 22:47:06Z wkliao $ */

#if HAVE_CONFIG_H
# include "ncconfig.h"
#endif

#include <stdio.h>
#include <unistd.h>
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#include <assert.h>
#include <string.h>  /* strtok(), strcpy(), strchr() */
#include <strings.h> /* strcasecmp() */

#include <mpi.h>

#include "nc.h"
#include "ncx.h"
#include "macro.h"
#ifdef ENABLE_SUBFILING
#include "subfile.h"
#endif

/* The const string below is for the RCS ident(1) command to find a string like
 * "\044Id: \100(#) PnetCDF library version 1.4.0 of 16 Nov 2013 $"
 * in the library file (libpnetcdf.a).
 */
static char const pnetcdf_libvers[] =
        "\044Id: \100(#) PnetCDF library version "PNETCDF_VERSION" of "PNETCDF_RELEASE_DATE" $";

/* pnetcdf_libvers is slightly different from the one returned from
 * ncmpi_inq_libvers(). The string pnetcdf_libvers is for command "ident" to
 * use. People can run command ident libpnetcdf.a to obtain the version of a
 * library (or an executable built from that library). In PnetCDF case, the
 * command will print the string of pnetcdf_libvers. Command "ident' looks for
 * a specific keyword pattern and print it. See man page of ident.
 *
 * The API ncmpi_inq_libvers() below on the other hand returns a string to be
 * used by the utility tools like ncmpidump, ncmpigen, etc. Check the last line
 * of output from command "ncmpidump -v".
 */

/*----< ncmpi_inq_libvers() >------------------------------------------------*/
inline const char*
ncmpi_inq_libvers(void) {

    /* match the style used by netCDF API nc_inq_libvers()
     * for example, "4.3.0 of Jun 16 2013 12:11:30 $" */
    /* we need some silly operation so the compiler will emit the otherwise
     * unused pnetcdf_libvers */
    if ((void *)pnetcdf_libvers != (void *)ncmpi_inq_libvers) {
	; /* do nothing */
    }
    return PNETCDF_VERSION " of " PNETCDF_RELEASE_DATE;
}

/* Begin Of Dataset Functions */

/*----< ncmpi_create() >-----------------------------------------------------*/
/**  \ingroup datasets
Create a new netCDF file.

This function creates a new netCDF dataset, returning a netCDF ID that can
subsequently be used to refer to the netCDF dataset in other PnetCDF function
calls. The new netCDF dataset opened for write access and placed in define
mode, ready for you to add dimensions, variables, and attributes.

\param comm The MPI communicator. This API is a collective routine: all
processes must provide the same value for cmode, and all processes must provide
filenames that reference the same file. (Values for info may vary.) comm must
be an MPI intracommunicator.

\param path The file name of the new netCDF dataset.

\param cmode The creation mode flag. The following flags are available:
NC_NOCLOBBER, NC_SHARE, NC_64BIT_OFFSET, and NC_64BIT_DATA.

\param info MPI info object. It is used to provide file access hints,including
existing MPI hints as well as PnetCDF hints.  For MPI hints, users are referred
to MPI user guide for further information. For PnetCDF hints see below.

\param ncidp Pointer to location where returned netCDF ID is to be stored.

<h2>The cmode Flag</h2>

The cmode flag is used to control the type of file created, and some aspects of
how it may be used.

Setting NC_NOCLOBBER means you do not want to clobber (overwrite) an existing
dataset; an error (NC_EEXIST) is returned if the specified dataset already
exists.

The NC_SHARE flag in PnetCDF does not mean sharing the file with the processes
in this MPI program. Instead, it means the file will be concurrently shared
by a different MPI program. Hence, PnetCDF calls MPI_File_sync() right after
every time an MPI_File_write() call is made. This includes writing to metadata
(file header) as well as array data.

Setting NC_64BIT_OFFSET causes PnetCDF to create a 64-bit offset format file
(CDF-2), instead of a netCDF classic format file.  The 64-bit offset format
imposes far fewer restrictions on large (i.e. over 2 GB) data files.  See Large
File Support (The PnetCDF Users Guide).

Setting NC_64BIT_DATA causes PnetCDF to create a 64-bit data format file
(CDF-5). The 64-bit data format allows define variables with more than 4
billion array elements. See Large File Support (The PnetCDF Users Guide).

A zero value (defined for convenience as NC_CLOBBER) specifies the default
behavior: overwrite any existing dataset with the same file name.

<h2>The info object Flag</h2>

Starting from version 1.3.1, the following PnetCDF hints are available:

- nc_header_align_size: This hint allows some extra space between the end of
the header describing the entire file and the first variable. If you have an
application that periodically wishes to add more variables to an already
existing file, expanding the file header size may result in an expensive move
of the entire data file to make room for the definition of the new variables.
Hence, setting this hint to a value that is big enough to accommodate any
additional variables means you may leave your application code as-is and yet
still see tremendous performance improvements.

- nc_var_align_size: If you are writing to a block-based parallel file system,
such as IBM's GPFS or Lustre, then an application write becomes a block write
at the file system layer. If a write straddles two blocks, then locks must be
acquired for both blocks. Aligning the start of a variable to a block boundary,
combined with collective I/O optimization in the MPI-IO library can often
eliminate all unaligned file system accesses.

- nc_record_align_size: This hint aligns the starting file offset of the
record variable section.

- nc_header_read_chunk_size: PnetCDF reads the file headers in chunks. This
hint indicates the chunk size (in bytes). The default is 256 KB.

\returns ::NC_NOERR No error.
\returns ::NC_ENOMEM System out of memory.
\returns ::NC_EEXIST Specified file name exists when using NC_NOCLOBBER.
Can be use to check if the file exists.
\returns ::NC_EMULTIDEFINE_OMODE Bad file create/open mode or modes are
inconsistent across processes
\returns ::NC_EOFILE: Can not open/create file (MPI-IO errors)
\returns ::NC_EFILE: Unknown error in file operation

<h1>Examples</h1>

In this example we create a netCDF dataset named foo.nc; we want the dataset to
be created in the current directory only if a dataset with that name does not
already exist:

@code
     #include <mpi.h>
     #include <pnetcdf.h>
        ...
     int status;
     int ncid;
     MPI_Info info;
        ...
     MPI_Info_create (&info);
     MPI_Info_set (info, "romio_no_indep_rw",    "true");
     MPI_Info_set (info, "nc_header_align_size", "4194304");
     MPI_Info_set (info, "nc_var_align_size",    "1048576");
     MPI_Info_set (info, "nc_record_align_size", "1048576");

     status = ncmpi_create(MPI_COMM_WORLD, "foo.nc", NC_NOCLOBBER, info, &ncid);
     if (status != NC_NOERR) handle_error(status);
     MPI_Info_free(&info);
@endcode

In this example we create a netCDF dataset named foo.nc. It will
be in the CDF-5 format.

@code
     #include <mpi.h>
     #include <pnetcdf.h>
        ...
     int status;
     int ncid;
     int cmode = NC_NOCLOBBER | NC_64BIT_DATA;
     MPI_Info info = MPI_INFO_NULL;
        ...
     status = ncmpi_create(MPI_COMM_WORLD, "foo.nc", cmode, info, &ncid);
     if (status != NC_NOERR) handle_error(status);
@endcode
*/
int
ncmpi_create(MPI_Comm    comm,
             const char *path,
             int         cmode,
             MPI_Info    info,
             int        *ncidp)
{
    int flag, err, status=NC_NOERR, safe_mode=0, mpireturn;
    char *env_str;
    MPI_Info   env_info;
    MPI_Offset chunksize=NC_DEFAULT_CHUNKSIZE;
    NC *ncp;

#ifdef PNC_DEBUG
    safe_mode = 1;
    /* this configure time setting will be overwritten by the run-time
     * environment variable PNETCDF_SAFE_MODE */
#endif
    /* get environment variable PNETCDF_SAFE_MODE
     * if it is set to 1, then we perform a strict parameter consistent test
     */
    if ((env_str = getenv("PNETCDF_SAFE_MODE")) != NULL) {
        if (*env_str == '0') safe_mode = 0;
        else                 safe_mode = 1;
        /* if PNETCDF_SAFE_MODE is set but without a value, *env_str can
         * be '\0' (null character). In this case, safe_mode is enabled */
    }

    if (safe_mode) {
        /* check if cmode is consistent with root's */
        int root_cmode=cmode;

        TRACE_COMM(MPI_Bcast)(&root_cmode, 1, MPI_INT, 0, comm);
        if (mpireturn != MPI_SUCCESS)
            return ncmpii_handle_error(mpireturn, "MPI_Bcast");

        if (root_cmode != cmode) {
            int rank;
            MPI_Comm_rank(comm, &rank);
            /* cmodes are inconsistent, overwrite local cmode with root's */
            printf("rank %d: Warning - inconsistent file create mode, overwrite with root's\n",rank);
            cmode = root_cmode;
            DEBUG_ASSIGN_ERROR(status, NC_EMULTIDEFINE_OMODE)
        }
        TRACE_COMM(MPI_Allreduce)(&status, &err, 1, MPI_INT, MPI_MIN, comm);
        if (err != NC_NOERR) return status;

        /* when safe_mode is disabled, NC_EMULTIDEFINE_OMODE will be reported at
         * the time ncmpi_enddef() returns */
    }

    /* It is illegal to have both NC_64BIT_OFFSET & NC_64BIT_DATA */
    if ((cmode & (NC_64BIT_OFFSET|NC_64BIT_DATA)) ==
                 (NC_64BIT_OFFSET|NC_64BIT_DATA))
        DEBUG_ASSIGN_ERROR(status, NC_EINVAL_CMODE)
    /* In safe_mode, cmodes are sync-ed, so all processes can return the same
     * error code. But, when not in safe mode, if cmode is not consistent
     * among processes, then some processes might not violate this above rule
     * which can cause the program to hang (because MPI_File_open is a
     * collective call). We use MPI_Allreduce below to check the error
     * code, but it is costly.
     */
    if (safe_mode) {
        TRACE_COMM(MPI_Allreduce)(&status, &err, 1, MPI_INT, MPI_MIN, comm);
        status = err;
    }
    if (status != NC_NOERR) return status;

    /* take hints from the environment variable PNETCDF_HINTS
     * a string of hints separated by ";" and each hint is in the
     * form of hint=value. E.g. cb_nodes=16;cb_config_list=*:6
     * If this environment variable is set, it  overrides any values that
     * were set by using calls to MPI_Info_set in the application code.
     */
    env_info = info;
    if ((env_str = getenv("PNETCDF_HINTS")) != NULL) {
        if (env_info == MPI_INFO_NULL)
            MPI_Info_create(&env_info); /* ignore error */

        char *key = strtok(env_str, ";");
        while (key != NULL) {
            char *val;
            val = strchr(key, '=');
            *val = '\0';
            val++;
            /* printf("env hint: key=%s val=%s\n",key,val); */
            MPI_Info_set(env_info, key, val); /* override or add */
            key = strtok(NULL, ";");
        }
    }

    /* get header chunk size from user info */
    if (env_info != MPI_INFO_NULL) {
        char value[MPI_MAX_INFO_VAL];
        MPI_Info_get(env_info, "nc_header_read_chunk_size", MPI_MAX_INFO_VAL-1,
                     value, &flag);
        if (flag) chunksize = atoll(value);
    }

    /* allocate buffer for header object NC */
    if ((ncp = ncmpii_new_NC(&chunksize)) == NULL)
        DEBUG_RETURN_ERROR(NC_ENOMEM)

    ncp->safe_mode = safe_mode;
    ncp->abuf      = NULL;
    ncp->old       = NULL;
#ifdef ENABLE_SUBFILING
    ncp->subfile_mode = 1;
    if (env_info != MPI_INFO_NULL) {
        char value[MPI_MAX_INFO_VAL];
        MPI_Info_get(env_info, "pnetcdf_subfiling", MPI_MAX_INFO_VAL-1,
                     value, &flag);
        if (flag && strcasecmp(value, "disable") == 0)
            ncp->subfile_mode = 0;
    }
    ncp->ncid_sf = -1; /* subfile ncid; init to -1 */
    ncp->nc_num_subfiles = 0; /* num_subfiles; init to 0 */
#endif
    assert(ncp->flags == 0);

    /* set the file format version based on the create mode, cmode */
    if (fIsSet(cmode, NC_64BIT_DATA)) {
#if SIZEOF_MPI_OFFSET <  8
        DEBUG_RETURN_ERROR(NC_ESMALL)
#endif
        fSet(ncp->flags, NC_64BIT_DATA);
    } else if (fIsSet(cmode, NC_64BIT_OFFSET)) {
        /* unlike serial netcdf, we will not bother to support
         * NC_64BIT_OFFSET on systems with off_t smaller than 8 bytes.
         * serial netcdf has proven it's possible if datasets are small, but
         * that's a hassle we don't want to worry about */
        if (SIZEOF_OFF_T < 8) DEBUG_RETURN_ERROR(NC_ESMALL)
        fSet(ncp->flags, NC_64BIT_OFFSET);
    } else {
        /* check default format */
        int default_format;
        ncmpi_inq_default_format(&default_format);
        if (default_format == NC_FORMAT_CDF5) {
#if SIZEOF_MPI_OFFSET <  8
            DEBUG_RETURN_ERROR(NC_ESMALL)
#endif
            fSet(ncp->flags, NC_64BIT_DATA);
        }
        else if (default_format == NC_FORMAT_CDF2) {
            if (SIZEOF_OFF_T < 8) DEBUG_RETURN_ERROR(NC_ESMALL)
            fSet(ncp->flags, NC_64BIT_OFFSET);
        }
        else
            fSet(ncp->flags, NC_32BIT);
    }

    /* find the true header size (not-yet aligned) */
    ncp->xsz = ncmpii_hdr_len_NC(ncp);

    fSet(ncp->flags, NC_NOFILL);

    err = ncmpiio_create(comm, path, cmode, env_info, ncp);
    if (err != NC_NOERR) { /* fatal error */
        ncmpii_free_NC(ncp);
        return err;
    }

    fSet(ncp->flags, NC_CREAT);

    /* the linked list storing the outstanding non-blocking requests */
    ncp->head = NULL;
    ncp->tail = NULL;
    ncp->numGetReqs = 0;
    ncp->numPutReqs = 0;

    /* add to the linked list of opened files */
    ncmpii_add_to_NCList(ncp);
    *ncidp = ncp->nciop->fd;

    if (env_info != info) MPI_Info_free(&env_info);

    return status;
}

/*----< ncmpi_open() >-------------------------------------------------------*/
int
ncmpi_open(MPI_Comm    comm,
           const char *path,
           int         omode,
           MPI_Info    info,
           int        *ncidp)
{
    int i, flag, err, status=NC_NOERR, safe_mode=0, mpireturn;
    char *env_str;
    MPI_Info   env_info;
    MPI_Offset chunksize=NC_DEFAULT_CHUNKSIZE;
    NC *ncp;

#ifdef PNC_DEBUG
    safe_mode = 1;
    /* this configure time setting will be overwritten by the run-time
     * environment variable PNETCDF_SAFE_MODE */
#endif
    /* get environment variable PNETCDF_SAFE_MODE
     * if it is set to 1, then we perform a strict parameter consistent test
     */
    if ((env_str = getenv("PNETCDF_SAFE_MODE")) != NULL) {
        if (*env_str == '0') safe_mode = 0;
        else                 safe_mode = 1;
        /* if PNETCDF_SAFE_MODE is set but without a value, *env_str can
         * be '\0' (null character). In this case, safe_mode is enabled */
    }

    if (safe_mode) {
        /* check if omode is consistent with root's */
        int root_omode=omode;

        /* Note if omode contains NC_NOWRITE, it is equivalent to NC_CLOBBER.
           In pnetcdf.h, they both are defined the same value, 0.
         */

        TRACE_COMM(MPI_Bcast)(&root_omode, 1, MPI_INT, 0, comm);
        if (mpireturn != MPI_SUCCESS)
            return ncmpii_handle_error(mpireturn, "MPI_Bcast");

        if (root_omode != omode) {
            int rank;
            MPI_Comm_rank(comm, &rank);
            /* omodes are inconsistent, overwrite local omode with root's */
            printf("rank %d: Warning - inconsistent file open mode, overwrite with root's\n",rank);
            omode = root_omode;
            DEBUG_ASSIGN_ERROR(status, NC_EMULTIDEFINE_OMODE)
        }
    }

    /* take hints from the environment variable PNETCDF_HINTS
     * a string of hints separated by ";" and each hint is in the
     * form of hint=value. E.g. cb_nodes=16;cb_config_list=*:6
     * If this environment variable is set, it  overrides any values that
     * were set by using calls to MPI_Info_set in the application code.
     */
    env_info = info;
    if ((env_str = getenv("PNETCDF_HINTS")) != NULL) {
        if (env_info == MPI_INFO_NULL)
            MPI_Info_create(&env_info); /* ignore error */

        char *key = strtok(env_str, ";");
        while (key != NULL) {
            char *val;
            val = strchr(key, '=');
            *val = '\0';
            val++;
            /* printf("env hint: key=%s val=%s\n",key,val); */
            MPI_Info_set(env_info, key, val); /* override or add */
            key = strtok(NULL, ";");
        }
    }

    /* get header chunk size from user info, if provided */
    if (env_info != MPI_INFO_NULL) {
        char value[MPI_MAX_INFO_VAL];
        MPI_Info_get(env_info, "nc_header_read_chunk_size", MPI_MAX_INFO_VAL-1,
                     value, &flag);
        if (flag) chunksize = atoll(value);
    }

    ncp = ncmpii_new_NC(&chunksize);
    if (ncp == NULL)
        DEBUG_RETURN_ERROR(NC_ENOMEM)

    ncp->safe_mode = safe_mode;
    ncp->old       = NULL;
#ifdef ENABLE_SUBFILING
    ncp->subfile_mode = 1;
    if (env_info != MPI_INFO_NULL) {
        char value[MPI_MAX_INFO_VAL];
        MPI_Info_get(env_info, "pnetcdf_subfiling", MPI_MAX_INFO_VAL-1,
                     value, &flag);
        if (flag && strcasecmp(value, "disable") == 0)
            ncp->subfile_mode = 0;
    }
    ncp->ncid_sf   = -1;
    ncp->nc_num_subfiles = 0;
#endif

    err = ncmpiio_open(comm, path, omode, env_info, ncp);
    if (err != NC_NOERR) { /* fatal error */
        ncmpii_free_NC(ncp);
        return err;
    }

    assert(ncp->flags == 0);
    fSet(ncp->flags, NC_NOFILL);

    err = ncmpii_hdr_get_NC(ncp); /* read header from file */
    if (err != NC_NOERR) { /* fatal error */
        ncmpiio_close(ncp->nciop, 0);
        ncmpii_free_NC(ncp);
        return err;
    }
    ncp->head = NULL;
    ncp->tail = NULL;
    ncp->numGetReqs = 0;
    ncp->numPutReqs = 0;

    ncmpii_add_to_NCList(ncp);
    *ncidp = ncp->nciop->fd;

#ifdef ENABLE_SUBFILING
    if (ncp->subfile_mode) {
        /* check attr for subfiles */
        err = ncmpi_get_att_int(ncp->nciop->fd, NC_GLOBAL, "num_subfiles",
                                &ncp->nc_num_subfiles);
        if (err == NC_NOERR && ncp->nc_num_subfiles > 1) {
            /* ignore error NC_ENOTATT if this attribute is not defined */
            int nvars;

            err = ncmpi_inq_nvars(ncp->nciop->fd, &nvars);
            if (status == NC_NOERR) status = err;

            for (i=0; i<nvars; i++) {
                err = ncmpi_get_att_int(ncp->nciop->fd, i, "num_subfiles",
                                        &ncp->vars.value[i]->num_subfiles);
                if (err == NC_ENOTATT) continue;
                if (err != NC_NOERR && status == NC_NOERR) { /* other error */
                    status = err;
                    continue;
                }

                if (ncp->vars.value[i]->num_subfiles > 1) {
                    err = ncmpi_get_att_int(ncp->nciop->fd, i, "ndims_org",
                                            &ncp->vars.value[i]->ndims_org);
                    if (status == NC_NOERR) status = err;
                }
            }

            if (ncp->nc_num_subfiles > 1) {
                err = ncmpii_subfile_open(ncp, &ncp->ncid_sf);
                if (status == NC_NOERR) status = err;
            }
        }
    }
    else
        ncp->nc_num_subfiles = 0;
#endif

    if (env_info != info) MPI_Info_free(&env_info);

    /* update the total number of record variables */
    ncp->vars.num_rec_vars = 0;
    for (i=0; i<ncp->vars.ndefined; i++)
        ncp->vars.num_rec_vars += IS_RECVAR(ncp->vars.value[i]);

    return status;
}

/*----< ncmpi_inq_format() >-------------------------------------------------*/
int
ncmpi_inq_format(int  ncid,
                 int *formatp) /* out */
{
    int status;
    NC *ncp;

    status = ncmpii_NC_check_id(ncid, &ncp);
    if (status != NC_NOERR)
        return status;

    if (fIsSet(ncp->flags, NC_64BIT_DATA)) {
        *formatp = NC_FORMAT_CDF5;
    } else if (fIsSet(ncp->flags, NC_64BIT_OFFSET)) {
        *formatp = NC_FORMAT_CDF2;
    } else if (fIsSet(ncp->flags, NC_32BIT)){
        *formatp = NC_FORMAT_CLASSIC;
    } else {
        /* this should not happen, because if ncid is valid, checking for
         * valid CDF format should have already been done already */
        *formatp = NC_FORMAT_UNKNOWN;
    }
    return status;
}

/*----< ncmpi_inq_file_format() >--------------------------------------------*/
int
ncmpi_inq_file_format(char *filename,
                      int  *formatp) /* out */
{
    int ncid, status;
    NC *ncp;

    /* open file for reading its header */
    status = ncmpi_open(MPI_COMM_SELF, filename, NC_NOWRITE, MPI_INFO_NULL,
                        &ncid);
    if (status == NC_ENOTNC)
        DEBUG_ASSIGN_ERROR(*formatp, NC_FORMAT_UNKNOWN)
    if (status != NC_NOERR)
        return status;

    status = ncmpii_NC_check_id(ncid, &ncp);
    if (status != NC_NOERR)
         return status;

    if (fIsSet(ncp->flags, NC_64BIT_DATA)) {
        *formatp = NC_FORMAT_CDF5;
    } else if (fIsSet(ncp->flags, NC_64BIT_OFFSET)) {
        *formatp = NC_FORMAT_CDF2;
    } else if (fIsSet(ncp->flags, NC_32BIT)){
        *formatp = NC_FORMAT_CLASSIC;
    } else {
        /* this should not happen, because if ncid is valid, checking for
         * valid CDF format should have already been done already */
        *formatp = NC_FORMAT_UNKNOWN;
    }
    status = ncmpi_close(ncid);

    return status;
}

/*----< ncmpi_inq_file_info() >-----------------------------------------------*/
int
ncmpi_inq_file_info(int       ncid,
                    MPI_Info *info_used)
{
    int mpireturn, status=NC_NOERR;
    char value[MPI_MAX_INFO_VAL];
    NC *ncp;

    status = ncmpii_NC_check_id(ncid, &ncp);
    if (status != NC_NOERR)
        return status;

#ifdef HAVE_MPI_INFO_DUP
    mpireturn = MPI_Info_dup(ncp->nciop->mpiinfo, info_used);
    if (mpireturn != MPI_SUCCESS)
        return ncmpii_handle_error(mpireturn, "MPI_Info_dup");
#else
    mpireturn = MPI_File_get_info(ncp->nciop->collective_fh, info_used);
    if (mpireturn != MPI_SUCCESS)
        return ncmpii_handle_error(mpireturn, "MPI_File_get_info");
#endif

    sprintf(value, "%lld", ncp->nciop->hints.h_align);
    MPI_Info_set(*info_used, "nc_header_align_size", value);

    sprintf(value, "%lld", ncp->nciop->hints.v_align);
    MPI_Info_set(*info_used, "nc_var_align_size", value);

    sprintf(value, "%lld", ncp->nciop->hints.r_align);
    MPI_Info_set(*info_used, "nc_record_align_size", value);

    sprintf(value, "%lld", ncp->nciop->hints.header_read_chunk_size);
    MPI_Info_set(*info_used, "nc_header_read_chunk_size", value);

#ifdef ENABLE_SUBFILING
    sprintf(value, "%d", ncp->nciop->hints.subfile_mode);
    MPI_Info_set(*info_used, "pnetcdf_subfiling", value);
    sprintf(value, "%d", ncp->nciop->hints.num_subfiles);
    MPI_Info_set(*info_used, "nc_num_subfiles", value);
#endif

    return NC_NOERR;
}

/*----< ncmpi_get_file_info() >-----------------------------------------------*/
int
ncmpi_get_file_info(int       ncid,
                    MPI_Info *info_used)
{
    return ncmpi_inq_file_info(ncid, info_used);
}

/*----< ncmpi_redef() >------------------------------------------------------*/
int
ncmpi_redef(int ncid) {
    int status;
    NC *ncp;

    status = ncmpii_NC_check_id(ncid, &ncp);
    if (status != NC_NOERR)
        return status;

    if (NC_readonly(ncp)) DEBUG_RETURN_ERROR(NC_EPERM) /* read-only */
    /* if open mode is inconsistent, then this return might cause parallel
     * program to hang */

    /* cannot be in define mode */
    if (NC_indef(ncp)) DEBUG_RETURN_ERROR(NC_EINDEFINE)

    /* sync all metadata, including numrecs, if changed in independent mode.
     * also ensure exiting define mode always entering collective data mode
     */
    if (NC_indep(ncp))
        ncmpii_end_indep_data(ncp);

    if (NC_doFsync(ncp)) { /* re-read the header from file */
        status = ncmpii_read_NC(ncp);
        if (status != NC_NOERR) return status;
    }

    /* duplicate a header to be uses in enddef() for checking if header grows */
    ncp->old = ncmpii_dup_NC(ncp);
    if (ncp->old == NULL) DEBUG_RETURN_ERROR(NC_ENOMEM)

    /* we are now entering define mode */
    fSet(ncp->flags, NC_INDEF);

    return NC_NOERR;
}

/*----< ncmpi_begin_indep_data() >-------------------------------------------*/
int
ncmpi_begin_indep_data(int ncid)
{
    int status=NC_NOERR;
    NC *ncp;

    status = ncmpii_NC_check_id(ncid, &ncp);
    if (status != NC_NOERR) return status;

    if (NC_indef(ncp))  /* must not be in define mode */
        DEBUG_RETURN_ERROR(NC_EINDEFINE)

    if (NC_indep(ncp))  /* already in indep data mode */
        return NC_NOERR;

    /* we need no MPI_File_sync() here. If users want a stronger data
     * consistency, they can either use NC_SHARE or call ncmpi_sync()
     */
#if 0 && !defined(DISABLE_FILE_SYNC)
    if (!NC_readonly(ncp) && NC_collectiveFhOpened(ncp->nciop)) {
        /* calling file sync for those already open the file */
        int err, mpireturn;
        /* MPI_File_sync() is collective */
        TRACE_IO(MPI_File_sync)(ncp->nciop->collective_fh);
        if (mpireturn != MPI_SUCCESS) {
            err = ncmpii_handle_error(mpireturn, "MPI_File_sync");
            if (status == NC_NOERR) status = err;
        }
        TRACE_COMM(MPI_Barrier)(ncp->nciop->comm);
    }
#endif

    fSet(ncp->flags, NC_INDEP);

    status = ncmpii_check_mpifh(ncp, 0);

    return status;
}

/*----< ncmpi_end_indep_data() >---------------------------------------------*/
int
ncmpi_end_indep_data(int ncid) {
    int status;
    NC *ncp;

    status = ncmpii_NC_check_id(ncid, &ncp);
    if (status != NC_NOERR) return status;

    if (!NC_indep(ncp)) /* must be in independent data mode */
        DEBUG_RETURN_ERROR(NC_ENOTINDEP)

    return ncmpii_end_indep_data(ncp);
}

/*----< ncmpii_end_indep_data() >--------------------------------------------*/
/* this function is called when:
 * 1. ncmpi_end_indep_data()
 * 2. ncmpi_redef() from independent data mode entering to define more
 * 3. ncmpii_close() when closing the file
 * This function is collective.
 */
int
ncmpii_end_indep_data(NC *ncp)
{
    int status=NC_NOERR;

    if (!NC_readonly(ncp)) {
        if (ncp->vars.num_rec_vars > 0) {
            /* numrecs dirty bit may not be the same across all processes.
             * force sync in memory no matter if dirty or not.
             */
            set_NC_ndirty(ncp);
            status = ncmpii_sync_numrecs(ncp, ncp->numrecs);
            /* the only possible dirty part of the header is numrecs */
        }

#ifndef DISABLE_FILE_SYNC
        /* calling file sync for those already open the file */
        if (NC_doFsync(ncp) && NC_independentFhOpened(ncp->nciop)) {
            int mpireturn;
            /* MPI_File_sync() is collective */
            TRACE_IO(MPI_File_sync)(ncp->nciop->independent_fh);
            if (mpireturn != MPI_SUCCESS) {
                int err = ncmpii_handle_error(mpireturn, "MPI_File_sync");
                if (status == NC_NOERR) status = err;
            }
            TRACE_COMM(MPI_Barrier)(ncp->nciop->comm);
        }
#endif
    }

    fClr(ncp->flags, NC_INDEP);

    return status;
}

/*----< ncmpi_enddef() >-----------------------------------------------------*/
int
ncmpi_enddef(int ncid) {
    int status;
    NC *ncp;

    /* check if file ID ncid is valid */
    status = ncmpii_NC_check_id(ncid, &ncp);
    if (status != NC_NOERR) return status;

    if (!NC_indef(ncp)) /* must currently in define mode */
        DEBUG_RETURN_ERROR(NC_ENOTINDEFINE)

    return ncmpii_enddef(ncp);
}

/*----< ncmpi__enddef() >-----------------------------------------------------*/
int
ncmpi__enddef(int        ncid,
              MPI_Offset h_minfree,
              MPI_Offset v_align,
              MPI_Offset v_minfree,
              MPI_Offset r_align)
{
    int status;
    NC *ncp;

    /* check if file ID ncid is valid */
    status = ncmpii_NC_check_id(ncid, &ncp);
    if (status != NC_NOERR) return status;

    if (!NC_indef(ncp)) /* must currently in define mode */
        DEBUG_RETURN_ERROR(NC_ENOTINDEFINE)

    return ncmpii__enddef(ncp, h_minfree, v_align, v_minfree, r_align);
}

/*----< ncmpi_sync_numrecs() >------------------------------------------------*/
/* this API is collective, but can be called in independent data mode.
 * Note numrecs is always sync-ed in memory and update in file in collective
 * data mode.
 */
int
ncmpi_sync_numrecs(int ncid) {
    int status = NC_NOERR;
    NC *ncp;

    status = ncmpii_NC_check_id(ncid, &ncp);
    if (status != NC_NOERR) return status;

    /* cannot be in define mode */
    if (NC_indef(ncp)) DEBUG_RETURN_ERROR(NC_EINDEFINE)

    /* check if we have defined record variables */
    if (ncp->vars.num_rec_vars == 0) return NC_NOERR;

    if (!NC_indep(ncp)) /* in collective data mode, numrecs is always sync-ed */
        return NC_NOERR;
    else /* if called in independent mode, we force sync in memory */
        set_NC_ndirty(ncp);

    /* sync numrecs in memory and file */
    status = ncmpii_sync_numrecs(ncp, ncp->numrecs);

#ifndef DISABLE_FILE_SYNC
    if (NC_doFsync(ncp)) { /* NC_SHARE is set */
        int err, mpireturn;
        if (NC_indep(ncp)) {
            TRACE_IO(MPI_File_sync)(ncp->nciop->independent_fh);
        }
        else {
            TRACE_IO(MPI_File_sync)(ncp->nciop->collective_fh);
        }
        if (mpireturn != MPI_SUCCESS) {
            err = ncmpii_handle_error(mpireturn, "MPI_File_sync");
            if (status == NC_NOERR) status = err;
        }
        TRACE_COMM(MPI_Barrier)(ncp->nciop->comm);
    }
#endif
    return status;
}

/*----< ncmpi_sync() >--------------------------------------------------------*/
/* This API must be called collectively, no matter if it is in collective
 * or independent data mode.
 */
int
ncmpi_sync(int ncid) {
    int status = NC_NOERR;
    NC *ncp;

    status = ncmpii_NC_check_id(ncid, &ncp);
    if (status != NC_NOERR) return status;

    if (NC_indef(ncp)) DEBUG_RETURN_ERROR(NC_EINDEFINE)

    if (NC_readonly(ncp))
        /* calling sync for file opened for read only means re-read header */
        return ncmpii_read_NC(ncp);

    /* the only part of header that can be dirty is numrecs (caused only by
     * independent APIs) */
    if (ncp->vars.num_rec_vars > 0 && NC_indep(ncp)) {
        /* sync numrecs in memory among processes and in file */
        set_NC_ndirty(ncp);
        status = ncmpii_sync_numrecs(ncp, ncp->numrecs);
        if (status != NC_NOERR) return status;
    }

    /* calling MPI_File_sync() on both collective and independent handlers */
    return ncmpiio_sync(ncp->nciop);
}

/*----< ncmpi_abort() >------------------------------------------------------*/
int
ncmpi_abort(int ncid) {
   /*
    * In data mode, same as ncmpiio_close.
    * In define mode, descard new definition.
    * In create, remove the file.
    */
    int status, err, doUnlink = 0;
    NC *ncp;

    status = ncmpii_NC_check_id(ncid, &ncp);
    if (status != NC_NOERR) return status;

    /* delete the file if it is newly created by ncmpi_create() */
    doUnlink = NC_IsNew(ncp);

    if (ncp->old != NULL) {
        /* a plain redef, not a create */
        assert(!NC_IsNew(ncp));
        assert(fIsSet(ncp->flags, NC_INDEF));
        ncmpii_free_NC(ncp->old);
        ncp->old = NULL;
        fClr(ncp->flags, NC_INDEF);
    }

    if (!doUnlink) {
        if (!NC_readonly(ncp) &&  /* file is open for write */
             NC_indep(ncp)) {     /* in independent data mode */
            status = ncmpii_end_indep_data(ncp); /* sync header */
        }

        if (NC_doFsync(ncp)) {
            err = ncmpiio_sync(ncp->nciop); /* calling MPI_File_sync() */
            if (status == NC_NOERR ) status = err;
        }
    }

    /* close the file */
    err = ncmpiio_close(ncp->nciop, doUnlink);
    if (status == NC_NOERR ) status = err;

    ncp->nciop = NULL;

    /* remove this file from the list of opened files */
    ncmpii_del_from_NCList(ncp);

    /* free up space occupied by the header metadata */
    ncmpii_free_NC(ncp);

    return status;
}

/*----< ncmpi_close() >------------------------------------------------------*/
int
ncmpi_close(int ncid) {
    int status = NC_NOERR;
    NC *ncp;

    status = ncmpii_NC_check_id(ncid, &ncp);
    if (status != NC_NOERR)
        return status;

    /* calling the implementation of ncmpi_close() */
    return ncmpii_close(ncp);
}

/*----< ncmpi_delete() >-----------------------------------------------------*/
/* ncmpi_delete:
 * doesn't do anything to release resources, so call ncmpi_close before calling
 * this function.
 *
 * filename: the name of the
 * file we will remove.  info: mpi info, in case underlying file system needs
 * hints.
 */
int
ncmpi_delete(char     *filename,
             MPI_Info  info)
{
    int err=NC_NOERR, mpireturn;

    TRACE_IO(MPI_File_delete)(filename, info);
    if (mpireturn != MPI_SUCCESS)
        err = ncmpii_handle_error(mpireturn, "MPI_File_delete");
    return err;
}

/* End Of Dataset Functions */

/*----< ncmpii_check_mpifh() >-----------------------------------------------*/
int
ncmpii_check_mpifh(NC  *ncp,
                   int  collective)
{
    int mpireturn;

    if (collective && NC_indep(ncp)) /* collective handle but in indep mode */
        DEBUG_RETURN_ERROR(NC_EINDEP)

    if (!collective && !NC_indep(ncp)) /* indep handle but in collective mode */
        DEBUG_RETURN_ERROR(NC_ENOTINDEP)

    if (collective && !NC_collectiveFhOpened(ncp->nciop)) {
        TRACE_IO(MPI_File_open)(ncp->nciop->comm, (char*)ncp->nciop->path,
                                ncp->nciop->mpiomode, ncp->nciop->mpiinfo,
                                &ncp->nciop->collective_fh);
        if (mpireturn != MPI_SUCCESS)
            return ncmpii_handle_error(mpireturn, "MPI_File_open");

        set_NC_collectiveFh(ncp->nciop);
    }
    else if (!collective && !NC_independentFhOpened(ncp->nciop)) {
        TRACE_IO(MPI_File_open)(MPI_COMM_SELF, (char*)ncp->nciop->path,
                                ncp->nciop->mpiomode, ncp->nciop->mpiinfo,
                                &ncp->nciop->independent_fh);
        if (mpireturn != MPI_SUCCESS)
            return ncmpii_handle_error(mpireturn, "MPI_File_open");

        set_NC_independentFh(ncp->nciop);
    }

    return NC_NOERR;
}

/*----< ncmpi_inq_put_size() >------------------------------------------------*/
/* returns the amount of writes, in bytes, committed to file system so far */
int
ncmpi_inq_put_size(int         ncid,
                   MPI_Offset *size)
{
    int status;
    NC *ncp;

    status = ncmpii_NC_check_id(ncid, &ncp);
    if (status != NC_NOERR)
        return status;

    *size = ncp->nciop->put_size;
    return NC_NOERR;
}

/*----< ncmpi_inq_get_size() >------------------------------------------------*/
/* returns the amount of reads, in bytes, obtained from file system so far */
int
ncmpi_inq_get_size(int         ncid,
                   MPI_Offset *size)
{
    int status;
    NC *ncp;

    status = ncmpii_NC_check_id(ncid, &ncp);
    if (status != NC_NOERR)
        return status;

    *size = ncp->nciop->get_size;
    return NC_NOERR;
}

/*----< ncmpi_inq_striping() >------------------------------------------------*/
/* return file (system) striping settings, striping size and count, if they are
 * available from MPI-IO hint. Otherwise, 0s are returned.
 */
int
ncmpi_inq_striping(int  ncid,
                   int *striping_size,
                   int *striping_count)
{
    int flag, status=NC_NOERR;
    char value[MPI_MAX_INFO_VAL];
    NC *ncp;

    status = ncmpii_NC_check_id(ncid, &ncp);
    if (status != NC_NOERR)
        return status;

    if (striping_size != NULL) {
        MPI_Info_get(ncp->nciop->mpiinfo, "striping_unit", MPI_MAX_INFO_VAL-1,
                     value, &flag);
        *striping_size = 0;
        if (flag) *striping_size = atoi(value);
    }

    if (striping_count != NULL) {
        MPI_Info_get(ncp->nciop->mpiinfo, "striping_factor", MPI_MAX_INFO_VAL-1,
                     value, &flag);
        *striping_count = 0;
        if (flag) *striping_count = atoi(value);
    }
    return NC_NOERR;
}

/*----< ncmpi_inq_malloc_size() >--------------------------------------------*/
/* report the current aggregate size allocated by malloc, yet to be freed */
int ncmpi_inq_malloc_size(MPI_Offset *size)
{
#ifdef PNC_MALLOC_TRACE
    ncmpii_inq_malloc_size(size);
    return NC_NOERR;
#else
    DEBUG_RETURN_ERROR(NC_ENOTENABLED)
#endif
}

/*----< ncmpi_inq_malloc_max_size() >----------------------------------------*/
/* get the max watermark ever researched by malloc (aggregated amount) */
int ncmpi_inq_malloc_max_size(MPI_Offset *size)
{
#ifdef PNC_MALLOC_TRACE
    ncmpii_inq_malloc_max_size(size);
    return NC_NOERR;
#else
    DEBUG_RETURN_ERROR(NC_ENOTENABLED)
#endif
}

/*----< ncmpi_inq_malloc_list() >--------------------------------------------*/
/* walk the malloc tree and print yet-to-be-freed malloc residues */
int ncmpi_inq_malloc_list(void)
{
#ifdef PNC_MALLOC_TRACE
    ncmpii_inq_malloc_list();
    return NC_NOERR;
#else
    DEBUG_RETURN_ERROR(NC_ENOTENABLED)
#endif
}

/*----< ncmpi_inq_files_opened() >-------------------------------------------*/
int
ncmpi_inq_files_opened(int *num, int *ncids)
{
    return ncmpii_inq_files_opened(num, ncids);
}

/*----< ncmpi_inq_recsize() >------------------------------------------------*/
int
ncmpi_inq_recsize(int         ncid,
                  MPI_Offset *recsize)
{
    int status;
    NC *ncp;

    status = ncmpii_NC_check_id(ncid, &ncp);
    if (status != NC_NOERR)
        return status;

    *recsize = ncp->recsize;
    return NC_NOERR;
}

/*----< ncmpi_inq_header_extent() >-------------------------------------------*/
int
ncmpi_inq_header_extent(int         ncid,
                        MPI_Offset *extent)
{
    int err;
    NC *ncp;

    err = ncmpii_NC_check_id(ncid, &ncp);
    if (err != NC_NOERR) return err;

    *extent = ncp->begin_var;

    return NC_NOERR;
}

/*----< ncmpi_inq_header_size() >---------------------------------------------*/
int
ncmpi_inq_header_size(int         ncid,
                      MPI_Offset *size)
{
    int err;
    NC *ncp;

    err = ncmpii_NC_check_id(ncid, &ncp);
    if (err != NC_NOERR) return err;

    *size = ncp->xsz;

    return NC_NOERR;
}


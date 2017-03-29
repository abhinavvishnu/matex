/*
 *  Copyright (C) 2003, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id: nc.c 2326 2016-03-01 17:21:03Z wkliao $ */

#if HAVE_CONFIG_H
# include "ncconfig.h"
#endif

#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include <mpi.h>

#include "nc.h"
#include "rnd.h"
#include "ncx.h"
#include "macro.h"
#ifdef ENABLE_SUBFILING
#include "subfile.h"
#endif

/* list of open netcdf's */
static NC *NClist = NULL;

/* This is the default create format for ncmpi_create and nc__create. */
static int default_create_format = NC_FORMAT_CLASSIC;

/* These have to do with version numbers. */
#define MAGIC_NUM_LEN 4
#define VER_CLASSIC 1
#define VER_64BIT_OFFSET 2
#define VER_HDF5 3
#define VER_64BIT_DATA 5

/* Prototypes for functions used only in this file */
#if 0
static int move_data_r(NC *ncp, NC *old);
static int move_vars_r(NC *ncp, NC *old);
static int NC_check_def(MPI_Comm comm, void *buf, MPI_Offset nn);
static int nc_set_fill(int ncid, int fillmode, int *old_mode_ptr);
#endif

/*----< ncmpii_add_to_NCList() >---------------------------------------------*/
void
ncmpii_add_to_NCList(NC *ncp)
{
    assert(ncp != NULL);

    /* add the newly created NC object to the head of linked list */
    ncp->prev = NULL;
    if (NClist != NULL)
        NClist->prev = ncp;
    ncp->next = NClist;
    NClist = ncp;
}

/*----< ncmpii_del_from_NCList() >-------------------------------------------*/
void
ncmpii_del_from_NCList(NC *ncp)
{
    assert(ncp != NULL);

    if (NClist == ncp) {
        assert(ncp->prev == NULL);
        NClist = ncp->next;
    }
    else {
        assert(ncp->prev != NULL);
        ncp->prev->next = ncp->next;
    }

    if (ncp->next != NULL)
        ncp->next->prev = ncp->prev;

    ncp->next = NULL;
    ncp->prev = NULL;
}

/*----< NC_check_header() >--------------------------------------------------*/
/*
 * Check the consistency of defined header metadata across all processes and
 * overwrite the local header objects with root's if inconsistency is found.
 * This function is collective.
 */
static int
NC_check_header(NC         *ncp,
                void       *buf,
                MPI_Offset  local_xsz) /* size of buf */
{
    void *cmpbuf;
    int rank, g_status, status=NC_NOERR, mpireturn;
    bufferinfo gbp;

    MPI_Comm_rank(ncp->nciop->comm, &rank);

    /* root's header size has been broadcasted in NC_begin() and saved in
     * ncp->xsz.
     */
    if (rank == 0)
        cmpbuf = buf;
    else
        cmpbuf = (void*) NCI_Malloc((size_t)ncp->xsz);

    /* process 0 broadcasts its header
     * TODO: currently the header size cannot be larger than 2^31 bytes,
     * due to the 2nd argument, count, of MPI_Bcast being of type int.
     * Possible solution is to broadcast in chunks of 2^31 bytes.
     */
    if (ncp->xsz != (int)ncp->xsz) {
        if (rank > 0) NCI_Free(cmpbuf);
        DEBUG_RETURN_ERROR(NC_EINTOVERFLOW)
    }

    TRACE_COMM(MPI_Bcast)(cmpbuf, (int)ncp->xsz, MPI_BYTE, 0, ncp->nciop->comm);

    if (rank > 0 && (ncp->xsz != local_xsz || memcmp(buf, cmpbuf, (size_t)ncp->xsz))) {
        /* now part of this process's header is not consistent with root's
         * check and report the inconsistent part
         */

        gbp.nciop  = ncp->nciop; /* will not be used in ncmpii_hdr_check_NC() */
        gbp.offset = 0;          /* will not be used in ncmpii_hdr_check_NC() */
        gbp.size   = ncp->xsz;   /* entire header is in the buffer, cmpbuf */
        gbp.index  = 0;
        gbp.pos    = gbp.base = cmpbuf;

        /* find the inconsistent part of the header, report the difference, and
         * overwrite the local header object with root's. ncmpii_hdr_check_NC()
         * should not have any MPI communication calls.
         */
        status = ncmpii_hdr_check_NC(&gbp, ncp);

        /* header consistency is only checked on non-root processes. The
         * returned status can be a fatal error or header inconsistency error,
         * (fatal errors are due to object allocation), but never NC_NOERR.
         */
    }

    if (ncp->safe_mode) {
        TRACE_COMM(MPI_Allreduce)(&status, &g_status, 1, MPI_INT, MPI_MIN,
                                  ncp->nciop->comm);
        if (mpireturn != MPI_SUCCESS) {
            if (rank > 0) NCI_Free(cmpbuf);
            return ncmpii_handle_error(mpireturn, "MPI_Allreduce"); 
        }

        if (g_status != NC_NOERR) { /* some headers are inconsistent */
            if (status == NC_NOERR) DEBUG_ASSIGN_ERROR(status, NC_EMULTIDEFINE)
        }
    }

    if (rank > 0) NCI_Free(cmpbuf);

    return status;
}


#if 0
/* 'defined but not used': seems like a useful function though. why did we
 * write it?  should we be using it? */

static int
NC_check_def(MPI_Comm comm, void *buf, MPI_Offset nn) {
  int rank;
  int errcheck;
  MPI_Offset compare = 0;
  void *cmpbuf;
  MPI_Offset max_size;

  MPI_Comm_rank(comm, &rank);

  if (rank == 0)
    max_size = nn;
  MPI_Bcast(&max_size, 1, MPI_OFFSET, 0, comm);

  compare = max_size - nn;

  MPI_Allreduce(&compare, &errcheck, 1, MPI_OFFSET, MPI_LOR, comm);

  if (errcheck)
    DEBUG_RETURN_ERROR(NC_EMULTIDEFINE)

  if (rank == 0)
    cmpbuf = buf;
  else
    cmpbuf = (void *)NCI_Malloc(nn);

  MPI_Bcast(cmpbuf, nn, MPI_BYTE, 0, comm);

  if (rank != 0) {
    compare = memcmp(buf, cmpbuf, nn);
    NCI_Free(cmpbuf);
  }

  MPI_Allreduce(&compare, &errcheck, 1, MPI_OFFSET, MPI_LOR, comm);

  if (errcheck){
    DEBUG_RETURN_ERROR(NC_EMULTIDEFINE)
  }else{
    return NC_NOERR;
  }
}
#endif

/*----< ncmpii_NC_check_id() >-----------------------------------------------*/
int
ncmpii_NC_check_id(int   ncid,
                   NC  **ncpp)
{
    NC *ncp;

    if (ncid >= 0) {
        for (ncp = NClist; ncp != NULL; ncp = ncp->next) {
            if (ncp->nciop->fd == ncid) {
                *ncpp = ncp;
                return NC_NOERR; /* normal return */
            }
        }
    }

    /* else, not found */
    DEBUG_RETURN_ERROR(NC_EBADID)
}


/*----< ncmpii_inq_files_opened() >------------------------------------------*/
int
ncmpii_inq_files_opened(int *num, int *ncids)
{
    NC *ncp;

/*
    for (ncp=NClist; ncp!=NULL; ncp=ncp->next)
        printf("still open %s\n",ncp->nciop->path);
*/
    if (num == NULL) DEBUG_RETURN_ERROR(NC_EINVAL)

    *num = 0;
    for (ncp=NClist; ncp!=NULL; ncp=ncp->next)
        (*num)++;

    if (*num > 0 && ncids != NULL) {
        /* when ncids is NULL, we skip getting the values */
        int i=0;
        for (ncp=NClist; ncp!=NULL; ncp=ncp->next)
            ncids[i] = ncp->nciop->fd;
    }
    return NC_NOERR;
}


/*----< ncmpii_free_NC() >----------------------------------------------------*/
inline void
ncmpii_free_NC(NC *ncp)
{
    if (ncp == NULL) return;
    ncmpii_free_NC_dimarray(&ncp->dims);
    ncmpii_free_NC_attrarray(&ncp->attrs);
    ncmpii_free_NC_vararray(&ncp->vars);
    NCI_Free(ncp);
}


/*----< ncmpii_new_NC() >----------------------------------------------------*/
inline NC *
ncmpii_new_NC(const MPI_Offset *chunkp)
{
    NC *ncp = (NC *) NCI_Calloc(1, sizeof(NC));

    if (ncp == NULL) return NULL;

    ncp->chunk = (chunkp != NULL) ? *chunkp : NC_SIZEHINT_DEFAULT;

    return ncp;
}

/*----< ncmpi_set_default_format() >-----------------------------------------*/
/* This function sets a default create file format.
 * Valid formats are NC_FORMAT_CLASSIC, NC_FORMAT_CDF2, and NC_FORMAT_CDF5
 * This API is collective.
 */
int
ncmpi_set_default_format(int format, int *old_formatp)
{
    int safe_mode=0, mpireturn, status=NC_NOERR;
    char *env_str;

    /* Return existing format if desired. */
    if (old_formatp)
        *old_formatp = default_create_format;

    env_str = getenv("PNETCDF_SAFE_MODE");
    if (env_str != NULL) {
        if (*env_str == '0') safe_mode = 0;
        else                 safe_mode = 1;
    }

    if (safe_mode) {
        /* check if format is consistent with root's */
        int root_format=format;

        TRACE_COMM(MPI_Bcast)(&root_format, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (mpireturn != MPI_SUCCESS)
            return ncmpii_handle_error(mpireturn, "MPI_Bcast");

        if (root_format != format) {
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            /* formats are inconsistent, overwrite local format with root's */
            printf("rank %d: Warning - inconsistent file format, overwrite with root's\n",rank);
            format = root_format;
            DEBUG_ASSIGN_ERROR(status, NC_EMULTIDEFINE_OMODE)
        }
    }


    /* Make sure only valid format is set. */
    if (format != NC_FORMAT_CLASSIC &&
        format != NC_FORMAT_CDF2 &&
        format != NC_FORMAT_CDF5) {
        DEBUG_RETURN_ERROR(NC_EINVAL)
    }
    default_create_format = format;

    return status;
}

/* returns a value suitable for a create flag.  Will return one or more of the
 * following values OR-ed together:
 * NC_64BIT_OFFSET, NC_CLOBBER, NC_LOCK, NC_SHARE */
int
ncmpi_inq_default_format(int *formatp)
{
    if (formatp == NULL) DEBUG_RETURN_ERROR(NC_EINVAL)

    *formatp = default_create_format;
    return NC_NOERR;
}

/*----< ncmpii_dup_NC() >----------------------------------------------------*/
NC *
ncmpii_dup_NC(const NC *ref)
{
    NC *ncp;

    ncp = (NC *) NCI_Malloc(sizeof(NC));
    if (ncp == NULL) return NULL;

    memset(ncp, 0, sizeof(NC));

    if (ncmpii_dup_NC_dimarray(&ncp->dims,   &ref->dims)  != NC_NOERR ||
        ncmpii_dup_NC_attrarray(&ncp->attrs, &ref->attrs) != NC_NOERR ||
        ncmpii_dup_NC_vararray(&ncp->vars,   &ref->vars)  != NC_NOERR) {
        ncmpii_free_NC(ncp);
        return NULL;
    }

    ncp->xsz       = ref->xsz;
    ncp->begin_var = ref->begin_var;
    ncp->begin_rec = ref->begin_rec;
    ncp->recsize   = ref->recsize;

    NC_set_numrecs(ncp, NC_get_numrecs(ref));
    return ncp;
}


/*
 *  Verify that this is a user nc_type
 * Formerly
NCcktype()
 * Sense of the return is changed.
 */
inline int
ncmpii_cktype(int     cdf_ver,
              nc_type type)
{
    /* the max data type supported by CDF-5 is NC_UINT64 */
    if (type <= 0 || type > NC_UINT64)
        DEBUG_RETURN_ERROR(NC_EBADTYPE)

    /* For CDF-1 and CDF-2 files, only classic types are allowed. */
    if (cdf_ver < 5 && type > NC_DOUBLE)
        DEBUG_RETURN_ERROR(NC_ESTRICTCDF2)

    return NC_NOERR;
}


/*
 * How many objects of 'type'
 * will fit into xbufsize?
 */
inline MPI_Offset
ncmpix_howmany(nc_type type, MPI_Offset xbufsize)
{
    switch(type){
        case NC_BYTE:
        case NC_UBYTE:
        case NC_CHAR:   return xbufsize;
        case NC_SHORT:  return xbufsize/X_SIZEOF_SHORT;
        case NC_USHORT: return xbufsize/X_SIZEOF_USHORT;
        case NC_INT:    return xbufsize/X_SIZEOF_INT;
        case NC_UINT:   return xbufsize/X_SIZEOF_UINT;
        case NC_FLOAT:  return xbufsize/X_SIZEOF_FLOAT;
        case NC_DOUBLE: return xbufsize/X_SIZEOF_DOUBLE;
        case NC_INT64:  return xbufsize/X_SIZEOF_INT64;
        case NC_UINT64: return xbufsize/X_SIZEOF_UINT64;
        default:
                assert("ncmpix_howmany: Bad type" == 0);
                return(0);
    }
}

#define D_RNDUP(x, align) _RNDUP(x, (off_t)(align))

/*----< NC_begins() >--------------------------------------------------------*/
/*
 * This function is only called at enddef().
 * It computes each variable's 'begin' offset, and sets/updates the followings:
 *    ncp->xsz                   ---- header size
 *    ncp->vars.value[*]->begin  ---- each variable's 'begin' offset
 *    ncp->begin_var             ---- offset of first non-record variable
 *    ncp->begin_rec             ---- offset of first     record variable
 *    ncp->recsize               ---- sum of single records
 *    ncp->numrecs               ---- number of records (set only if new file)
 */
static int
NC_begins(NC         *ncp,
          MPI_Offset  h_align,  /* header alignment */
          MPI_Offset  h_minfree,/* free space for header */
          MPI_Offset  v_align,  /* alignment for each fixed variable */
          MPI_Offset  v_minfree,/* free space for fixed variable section */
          MPI_Offset  r_align)  /* alignment for record variable section */
{
    int i, j, rank, cdf_format, mpireturn;
    MPI_Offset end_var=0;
    NC_var *last = NULL;
    NC_var *first_var = NULL;       /* first "non-record" var */

    /* cdf_format determines the size of variable's "begin" in the header */
    if (fIsSet(ncp->flags, NC_64BIT_DATA))
        cdf_format = 5;  /* CDF-5 */
    else if (fIsSet(ncp->flags, NC_64BIT_OFFSET))
        cdf_format = 2;  /* CDF-2 */
    else
        cdf_format = 1;  /* CDF-1 */

    /* get the true header size (un-aligned one) */
    MPI_Comm_rank(ncp->nciop->comm, &rank);
    if (rank ==0) ncp->xsz = ncmpii_hdr_len_NC(ncp);

    /* only root's header size matters */
    TRACE_COMM(MPI_Bcast)(&ncp->xsz, 1, MPI_OFFSET, 0, ncp->nciop->comm);
    if (mpireturn != MPI_SUCCESS)
        return ncmpii_handle_error(mpireturn, "MPI_Bcast"); 

    /* This function is called in ncmpi_enddef(), which can happen either when
     * creating a new file or opening an existing file with metadata modified.
     * For the former case, ncp->begin_var == 0 here.
     * For the latter case, we set begin_var a new value only if the new header
     * grows out of its extent or the start of non-record variables is not
     * aligned as requested by h_align.
     * Note ncp->xsz is header size and ncp->begin_var is header extent.
     * Add the minimum header free space requested by user.
     */
    if (h_minfree < 0) h_minfree = 0;
    ncp->begin_var = D_RNDUP(ncp->xsz + h_minfree, h_align);

    if (ncp->old != NULL) {
        /* If this define mode was entered from a redef(), we check whether
         * the new begin_var against the old begin_var. We do not shrink
         * the header extent.
         */
        if (ncp->begin_var < ncp->old->begin_var)
            ncp->begin_var = ncp->old->begin_var;
    }

    /* ncp->begin_var is the aligned starting file offset of the first
       variable, also the extent of file header */

    /* Now calculate the starting file offsets for all variables.
       loop thru vars, first pass is for the 'non-record' vars */
    end_var = ncp->begin_var;
    for (j=0, i=0; i<ncp->vars.ndefined; i++) {
        if (IS_RECVAR(ncp->vars.value[i]))
            /* skip record variables on this pass */
            continue;
        if (first_var == NULL) first_var = ncp->vars.value[i];

        /* for CDF-1 check if over the file size limit 32-bit integer */
        if (cdf_format == 1 && end_var > X_OFF_MAX)
            DEBUG_RETURN_ERROR(NC_EVARSIZE)

        /* this will pad out non-record variables with zero to the
         * requested alignment.  record variables are a bit trickier.
         * we don't do anything special with them */
        ncp->vars.value[i]->begin = D_RNDUP(end_var, v_align);

        if (ncp->old != NULL) {
            /* move to the next fixed variable */
            for (; j<ncp->old->vars.ndefined; j++)
                if (!IS_RECVAR(ncp->old->vars.value[j]))
                    break;
            if (j < ncp->old->vars.ndefined) {
                if (ncp->vars.value[i]->begin < ncp->old->vars.value[j]->begin)
                    /* the first ncp->vars.ndefined non-record variables should
                       be the same. If the new begin is smaller, reuse the old
                       begin */
                    ncp->vars.value[i]->begin = ncp->old->vars.value[j]->begin;
                j++;
            }
        }
        /* end_var is the end offset of variable i */
        end_var = ncp->vars.value[i]->begin + ncp->vars.value[i]->len;
    }

    /* end_var now is pointing to the end of last non-record variable */

    /* only (re)calculate begin_rec if there is not sufficient
     * space at end of non-record variables or if start of record
     * variables is not aligned as requested by r_align.
     * If the existing begin_rec is already >= index, then leave the
     * begin_rec as is (in case some non-record variables are deleted)
     */
    if (ncp->begin_rec < end_var ||
        ncp->begin_rec != D_RNDUP(ncp->begin_rec, v_align))
        ncp->begin_rec = D_RNDUP(end_var, v_align);

    /* expand free space for fixed variable section */
    if (ncp->begin_rec < end_var + v_minfree)
        ncp->begin_rec = D_RNDUP(end_var + v_minfree, v_align);

    /* align the starting offset for record variable section */
    if (r_align > 1)
        ncp->begin_rec = D_RNDUP(ncp->begin_rec, r_align);

    if (ncp->old != NULL) {
        /* check whether the new begin_rec is smaller */
        if (ncp->begin_rec < ncp->old->begin_rec)
            ncp->begin_rec = ncp->old->begin_rec;
    }

    if (first_var != NULL)
        ncp->begin_var = first_var->begin;
    else
        ncp->begin_var = ncp->begin_rec;

    end_var = ncp->begin_rec;
    /* end_var now is pointing to the beginning of record variables
     * note that this can be larger than the end of last non-record variable
     */

    ncp->recsize = 0;

    /* TODO: alignment for record variables (maybe using a new hint) */

    /* loop thru vars, second pass is for the 'record' vars,
     * re-calculate the starting offset for each record variable */
    for (j=0, i=0; i<ncp->vars.ndefined; i++) {
        if (!IS_RECVAR(ncp->vars.value[i]))
            /* skip non-record variables on this pass */
            continue;

        /* X_OFF_MAX is the max of 32-bit integer */
        if (cdf_format == 1 && end_var > X_OFF_MAX)
            DEBUG_RETURN_ERROR(NC_EVARSIZE)

        /* A few attempts at aligning record variables have failed
         * (either with range error or 'value read not that expected',
         * or with an error in ncmpi_redef )).  Not sufficient to align
         * 'begin', but haven't figured out what else to adjust */
        ncp->vars.value[i]->begin = end_var;

        if (ncp->old != NULL) {
            /* move to the next record variable */
            for (; j<ncp->old->vars.ndefined; j++)
                if (IS_RECVAR(ncp->old->vars.value[j]))
                    break;
            if (j < ncp->old->vars.ndefined) {
                if (ncp->vars.value[i]->begin < ncp->old->vars.value[j]->begin)
                    /* if the new begin is smaller, use the old begin */
                    ncp->vars.value[i]->begin = ncp->old->vars.value[j]->begin;
                j++;
            }
        }
        end_var += ncp->vars.value[i]->len;
        /* end_var is the end offset of record variable i */

        /* check if record size must fit in 32-bits */
#if SIZEOF_OFF_T == SIZEOF_SIZE_T && SIZEOF_SIZE_T == 4
        if (ncp->recsize > X_UINT_MAX - ncp->vars.value[i]->len)
            DEBUG_RETURN_ERROR(NC_EVARSIZE)
#endif
        ncp->recsize += ncp->vars.value[i]->len;
        last = ncp->vars.value[i];
    }

    /*
     * for special case (Check CDF-1 and CDF-2 file format specifications.)
     * "A special case: Where there is exactly one record variable, we drop the
     * requirement that each record be four-byte aligned, so in this case there
     * is no record padding."
     */
    if (last != NULL) {
        if (ncp->recsize == last->len) {
            /* exactly one record variable, pack value */
            ncp->recsize = *last->dsizes * last->xsz;
        }
#if 0
        else if (last->len == UINT32_MAX) { /* huge last record variable */
            ncp->recsize += *last->dsizes * last->xsz;
        }
#endif
    }

/* below is only needed if alignment is performed on record variables */
#if 0
    /*
     * for special case of exactly one record variable, pack value
     */
    /* if there is exactly one record variable, then there is no need to
     * pad for alignment -- there's nothing after it */
    if (last != NULL && ncp->recsize == last->len)
        ncp->recsize = *last->dsizes * last->xsz;
#endif

    if (NC_IsNew(ncp))
        NC_set_numrecs(ncp, 0);

    return NC_NOERR;
}

#define NC_NUMRECS_OFFSET 4

/*----< ncmpii_sync_numrecs() >-----------------------------------------------*/
/* Synchronize the number of records in memory and write numrecs to file.
 * This function is called by:
 * 1. ncmpi_sync_numrecs(): by the user
 * 2. ncmpi_sync(): by the user
 * 3. ncmpii_end_indep_data(): exit from independent data mode
 * 4. all blocking collective put APIs (getput.m4) when writing record variable
 * 5. collective nonblocking wait API (ncmpii_wait_getput)
 * 6. ncmpii_close(): file close and currently in independent data mode
 *
 * This function is collective.
 */
int
ncmpii_sync_numrecs(NC         *ncp,
                    MPI_Offset  new_numrecs)
{
    int rank, status=NC_NOERR, mpireturn, err;
    MPI_File fh;
    MPI_Offset max_numrecs;

    assert(!NC_readonly(ncp));
    assert(!NC_indef(ncp)); /* can only be called by APIs in data mode */

    /* find the max new_numrecs among all processes
     * Note new_numrecs may be smaller than ncp->numrecs
     */
    TRACE_COMM(MPI_Allreduce)(&new_numrecs, &max_numrecs, 1, MPI_OFFSET,
                              MPI_MAX, ncp->nciop->comm);
    if (mpireturn != MPI_SUCCESS)
        return ncmpii_handle_error(mpireturn, "MPI_Allreduce");

    fh = ncp->nciop->collective_fh;
    if (NC_indep(ncp))
        fh = ncp->nciop->independent_fh;

    /* root process writes numrecs in file */
    MPI_Comm_rank(ncp->nciop->comm, &rank);
    if (rank == 0 && /* Only root process writes to file header */
        (max_numrecs > ncp->numrecs || NC_ndirty(ncp))) {
         /* For collective data mode, we check max_numrecs against root's
          * ncp->numrecs because root's numrecs has not been updated.
          * For independent data mode, we check NC_ndirty bit, because root's
          * numrecs may have been updated and in this case NC_ndirty bit has
          * been set to dirty. */
        int len;
        char pos[8], *buf=pos;
        MPI_Status mpistatus;

        if (ncp->flags & NC_64BIT_DATA) {
            len = X_SIZEOF_INT64;
            err = ncmpix_put_uint64((void**)&buf, max_numrecs);
            if (status == NC_NOERR) status = err;
        }
        else {
            if (max_numrecs != (int)max_numrecs) DEBUG_ASSIGN_ERROR(status, NC_EINTOVERFLOW)
            len = X_SIZEOF_SIZE_T;
            err = ncmpix_put_uint32((void**)&buf, (uint)max_numrecs);
            if (status == NC_NOERR) status = err;
        }
        /* ncmpix_put_xxx advances the 1st argument with size len */

        /* root's file view always includes the entire file header */

        TRACE_IO(MPI_File_write_at)(fh, NC_NUMRECS_OFFSET, (void*)pos, len,
                                    MPI_BYTE, &mpistatus);
        if (mpireturn != MPI_SUCCESS) {
            err = ncmpii_handle_error(mpireturn, "MPI_File_write_at");
            if (status == NC_NOERR && err == NC_EFILE) DEBUG_ASSIGN_ERROR(status, NC_EWRITE)
        }
        else {
            int put_size;
            MPI_Get_count(&mpistatus, MPI_BYTE, &put_size);
            ncp->nciop->put_size += put_size;
        }
    }
    /* update numrecs in all processes's memory only if the new one is larger.
     * Note new_numrecs may be smaller than ncp->numrecs
     */
    if (max_numrecs > ncp->numrecs) ncp->numrecs = max_numrecs;

    if (ncp->safe_mode == 1) {
        /* broadcast root's status, because only root writes to the file */
        int root_status = status;
        TRACE_COMM(MPI_Bcast)(&root_status, 1, MPI_INT, 0, ncp->nciop->comm);
        /* root's write has failed, which is serious */
        if (root_status == NC_EWRITE) DEBUG_ASSIGN_ERROR(status, NC_EWRITE)
    }

    /* clear numrecs dirty bit */
    fClr(ncp->flags, NC_NDIRTY);

    return status;
}

/*
 * Read in the header
 * It is expensive.
 */

inline int
ncmpii_read_NC(NC *ncp) {
  int status = NC_NOERR;

  ncmpii_free_NC_dimarray(&ncp->dims);
  ncmpii_free_NC_attrarray(&ncp->attrs);
  ncmpii_free_NC_vararray(&ncp->vars);

  status = ncmpii_hdr_get_NC(ncp);

  if (status == NC_NOERR)
      fClr(ncp->flags, NC_NDIRTY);

  return status;
}

/*----< write_NC() >---------------------------------------------------------*/
/*
 * This function is collective and only called by enddef().
 * Write out the header
 * 1. Call ncmpii_hdr_put_NC() to copy the header object, ncp, to a buffer.
 * 2. Call NC_check_header() to check if header is consistent across all
 *    processes.
 * 3. Process rank 0 writes the header to file.
 * This is a collective call.
 */
static int
write_NC(NC *ncp)
{
    void *buf;
    int status, mpireturn, err, max_err, rank;
    MPI_Offset local_xsz;

    assert(!NC_readonly(ncp));

    /* ncp->xsz is root's header size, we need to calculate local's */
    local_xsz = ncmpii_hdr_len_NC(ncp);

    buf = NCI_Malloc((size_t)local_xsz); /* buffer for local header object */

    /* copy the entire local header object to buffer */
    status = ncmpii_hdr_put_NC(ncp, buf);
    if (status != NC_NOERR) { /* a fatal error */
        NCI_Free(buf);
        return status;
    }

    /* check the header consistency across all processes and sync header.
     * When safe_mode is on:
     *   The returned status on root can be either NC_NOERR (all headers are
     *   consistent) or NC_EMULTIDEFINE (some headers are inconsistent).
     *   The returned status on non-root processes can be NC_NOERR, fatal
     *   error (>-250), or inconsistency error (-250 to -269).
     * When safe_mode is off:
     *   The returned status on root is always NC_NOERR
     *   The returned status on non-root processes can be NC_NOERR, fatal
     *   error (>-250), or inconsistency error (-250 to -269).
     * For fatal error, we should stop. For others, we can continue.
     */
    status = NC_check_header(ncp, buf, local_xsz);

    /* check for fatal error */
    err =  (status != NC_NOERR && !ErrIsHeaderDiff(status)) ? 1 : 0;
    max_err = err;

    if (ncp->safe_mode == 1) {
        TRACE_COMM(MPI_Allreduce)(&err, &max_err, 1, MPI_INT, MPI_MAX,
                                  ncp->nciop->comm);
        if (mpireturn != MPI_SUCCESS) {
            ncmpii_handle_error(mpireturn,"MPI_Allreduce");
            DEBUG_RETURN_ERROR(NC_EMPI)
        }
    }

    if (max_err == 1) { /* some processes encounter a fatal error */
        NCI_Free(buf);
        return status;
    }
    /* For non-fatal error, we continue to write header to the file, as now the
     * header object in memory has been sync-ed across all processes. */

    /* only rank 0's header gets written to the file */
    MPI_Comm_rank(ncp->nciop->comm, &rank);
    if (rank == 0) {
        /* rank 0's fileview already includes the file header */
        MPI_Status mpistatus;
        if (ncp->xsz != (int)ncp->xsz) DEBUG_ASSIGN_ERROR(status, NC_EINTOVERFLOW)
        TRACE_IO(MPI_File_write_at)(ncp->nciop->collective_fh, 0, buf,
                                    (int)ncp->xsz, MPI_BYTE, &mpistatus);
        if (mpireturn != MPI_SUCCESS) {
            err = ncmpii_handle_error(mpireturn, "MPI_File_write_at");
            /* write has failed, which is more serious than inconsistency */
            if (err == NC_EFILE) DEBUG_ASSIGN_ERROR(status, NC_EWRITE)
        }
        else {
            int put_size;
            MPI_Get_count(&mpistatus, MPI_BYTE, &put_size);
            ncp->nciop->put_size += put_size;
        }
    }

    if (ncp->safe_mode == 1) {
        /* broadcast root's status, because only root writes to the file */
        int root_status = status;
        TRACE_COMM(MPI_Bcast)(&root_status, 1, MPI_INT, 0, ncp->nciop->comm);
        /* root's write has failed, which is more serious than inconsistency */
        if (root_status == NC_EWRITE) DEBUG_ASSIGN_ERROR(status, NC_EWRITE)
    }

    fClr(ncp->flags, NC_NDIRTY);
    NCI_Free(buf);

    return status;
}

inline int
ncmpii_dset_has_recvars(NC *ncp)
{
    /* possible further optimization: set a flag on the header data
     * structure when record variable created so we can skip this loop*/
    int i;
    NC_var **vpp;

    vpp = ncp->vars.value;
    for (i=0; i< ncp->vars.ndefined; i++, vpp++) {
        if (IS_RECVAR(*vpp)) return 1;
    }
    return 0;
}


#if 0
/*
 * header size increases, shift all record and non-record variables down
 */
static int
move_data_r(NC *ncp, NC *old) {
    /* no new record or non-record variable inserted, header size increases,
     * must shift (move) the whole contiguous data part down
     * Note ncp->numrecs may be > old->numrecs
     */
    return ncmpiio_move(ncp->nciop, ncp->begin_var, old->begin_var,
                        old->begin_rec - old->begin_var +
                        ncp->recsize * ncp->numrecs);
}
#endif

/*
 * Move the record variables down,
 * re-arrange records as needed
 * Fill as needed.
 */
static int
move_recs_r(NC *ncp, NC *old) {
    int status;
    MPI_Offset recno;
    const MPI_Offset nrecs = ncp->numrecs;
    const MPI_Offset ncp_recsize = ncp->recsize;
    const MPI_Offset old_recsize = old->recsize;
    const off_t ncp_off = ncp->begin_rec;
    const off_t old_off = old->begin_rec;

    assert(ncp_recsize >= old_recsize);

    if (ncp_recsize == old_recsize) {
        if (ncp_recsize == 0) /* no record variable defined yet */
            return NC_NOERR;

        /* No new record variable inserted, move all record variables as a whole */
        status = ncmpiio_move(ncp->nciop, ncp_off, old_off, ncp_recsize * nrecs);
        if (status != NC_NOERR)
            return status;
    } else {
        /* new record variables inserted, move one whole record at a time */
        for (recno = nrecs-1; recno >= 0; recno--) {
            status = ncmpiio_move(ncp->nciop,
                                  ncp_off+recno*ncp_recsize,
                                  old_off+recno*old_recsize,
                                  old_recsize);
            if (status != NC_NOERR)
                return status;
        }
    }

    return NC_NOERR;
}


#if 0
/*
 * Move the "non record" variables "out".
 * Fill as needed.
 */

static int
move_vars_r(NC *ncp, NC *old) {
  return ncmpiio_move(ncp->nciop, ncp->begin_var, old->begin_var,
                   old->begin_rec - old->begin_var);
}
#endif

/*
 * Given a valid ncp, return NC_EVARSIZE if any variable has a bad len
 * (product of non-rec dim sizes too large), else return NC_NOERR.
 */
static int
ncmpii_NC_check_vlens(NC *ncp)
{
    NC_var **vpp;
    /* maximum permitted variable size (or size of one record's worth
       of a record variable) in bytes.  This is different for format 1
       and format 2. */
    MPI_Offset vlen_max;
    MPI_Offset ii;
    MPI_Offset large_vars_count;
    MPI_Offset rec_vars_count;
    int last = 0;

    if(ncp->vars.ndefined == 0)
       return NC_NOERR;

    if ((ncp->flags & NC_64BIT_DATA) && SIZEOF_OFF_T > 4)
       return NC_NOERR;

    if ((ncp->flags & NC_64BIT_OFFSET) && SIZEOF_OFF_T > 4) {
       /* CDF2 format and LFS */
       vlen_max = X_UINT_MAX - 3; /* "- 3" handles rounded-up size */
    } else {
       /* CDF1 format */
       vlen_max = X_INT_MAX - 3;
    }
    /* Loop through vars, first pass is for non-record variables.   */
    large_vars_count = 0;
    rec_vars_count = 0;
    vpp = ncp->vars.value;
    for (ii = 0; ii < ncp->vars.ndefined; ii++, vpp++) {
       if( !IS_RECVAR(*vpp) ) {
           last = 0;
           if( ncmpii_NC_check_vlen(*vpp, vlen_max) == 0 ) {
               large_vars_count++;
               last = 1;
           }
       } else {
         rec_vars_count++;
       }
    }
    /* OK if last non-record variable size too large, since not used to
       compute an offset */
    if( large_vars_count > 1) { /* only one "too-large" variable allowed */
      DEBUG_RETURN_ERROR(NC_EVARSIZE)
    }
    /* and it has to be the last one */
    if( large_vars_count == 1 && last == 0) {
      DEBUG_RETURN_ERROR(NC_EVARSIZE)
    }
    if( rec_vars_count > 0 ) {
       /* and if it's the last one, there can't be any record variables */
       if( large_vars_count == 1 && last == 1) {
           DEBUG_RETURN_ERROR(NC_EVARSIZE)
       }
       /* Loop through vars, second pass is for record variables.   */
       large_vars_count = 0;
       vpp = ncp->vars.value;
       for (ii = 0; ii < ncp->vars.ndefined; ii++, vpp++) {
           if( IS_RECVAR(*vpp) ) {
               last = 0;
               if( ncmpii_NC_check_vlen(*vpp, vlen_max) == 0 ) {
                   large_vars_count++;
                   last = 1;
               }
           }
       }
       /* OK if last record variable size too large, since not used to
          compute an offset */
       if( large_vars_count > 1) { /* only one "too-large" variable allowed */
           DEBUG_RETURN_ERROR(NC_EVARSIZE)
       }
       /* and it has to be the last one */
       if( large_vars_count == 1 && last == 0) {
           DEBUG_RETURN_ERROR(NC_EVARSIZE)
       }
    }
    return NC_NOERR;
}

#define DEFAULT_ALIGNMENT 512
#define HEADER_ALIGNMENT_LB 4

/* Many subroutines called in ncmpii_NC_enddef() are collective. We check the
 * error codes of all processes only in safe mode, so the program can stop
 * collectively, if any one process got an error. However, when safe mode is
 * off, we simply return the error and program may hang if some processes
 * do not get error and proceed to the next subroutine call.
 */ 
#define CHECK_ERROR(status) {                                                \
    if (ncp->safe_mode == 1) {                                               \
        int g_status;                                                        \
        TRACE_COMM(MPI_Allreduce)(&status, &g_status, 1, MPI_INT, MPI_MIN,   \
                                  ncp->nciop->comm);                         \
        if (mpireturn != MPI_SUCCESS)                                        \
            return ncmpii_handle_error(mpireturn, "MPI_Allreduce");          \
        if (g_status != NC_NOERR) return status;                             \
    }                                                                        \
    else if (status != NC_NOERR)                                             \
        return status;                                                       \
}

/*----< ncmpii_NC_enddef() >-------------------------------------------------*/
static int
ncmpii_NC_enddef(NC         *ncp,
                 MPI_Offset  h_align,
                 MPI_Offset  h_minfree,
                 MPI_Offset  v_align,
                 MPI_Offset  v_minfree,
                 MPI_Offset  r_align)
{
    int i, err, status=NC_NOERR, mpireturn;
    char value[MPI_MAX_INFO_VAL];
#ifdef ENABLE_SUBFILING
    NC *ncp_sf=NULL;
#endif

    assert(h_align > 0);  /* alignment size cannot be zero */
    assert(v_align > 0);
    assert(r_align > 0);

    /* all CDF formats require 4-bytes alignment */
    h_align = D_RNDUP(h_align, 4);
    v_align = D_RNDUP(v_align, 4);
    r_align = D_RNDUP(r_align, 4);

    /* reflect the hint changes to the MPI info object, so the user can
     * query exactly what hint values are being used
     */
    ncp->nciop->hints.h_align = h_align;
    ncp->nciop->hints.v_align = v_align;
    ncp->nciop->hints.r_align = r_align;

    sprintf(value, "%lld", h_align);
    MPI_Info_set(ncp->nciop->mpiinfo, "nc_header_align_size", value);
    sprintf(value, "%lld", v_align);
    MPI_Info_set(ncp->nciop->mpiinfo, "nc_var_align_size", value);
    sprintf(value, "%lld", r_align);
    MPI_Info_set(ncp->nciop->mpiinfo, "nc_record_align_size", value);

#ifdef ENABLE_SUBFILING
    /* num of subfiles has been determined already */
    ncp->subfile_mode    = ncp->nciop->hints.subfile_mode;
    ncp->nc_num_subfiles = ncp->nciop->hints.num_subfiles;

    if (ncp->nc_num_subfiles > 1) {
        /* TODO: should return subfile-related msg when there's an error */
        status = ncmpii_subfile_partition(ncp, &ncp->ncid_sf);
        if (status != NC_NOERR)
            printf("Error in file %s line %d (%s)\n",__FILE__,__LINE__,
                   ncmpi_strerror(status));

        CHECK_ERROR(status)
    }
#endif

    /* check on dimension lengths */
    status = ncmpii_NC_check_vlens(ncp);
    CHECK_ERROR(status)

    /* When ncp->old == NULL, this enddef is called the first time after file
     * create call. In this case, we compute each variable's 'begin', starting
     * file offset as well as the offsets of record variables.
     * When ncp->old != NULL, this enddef is called after a redef. In this
     * case, we re-used all variable offsets as many as possible.
     */
    status = NC_begins(ncp, h_align, h_minfree, v_align, v_minfree, r_align);
    CHECK_ERROR(status)

#ifdef ENABLE_SUBFILING
    if (ncp->nc_num_subfiles > 1) {
        /* get ncp info for the subfile */
        status = ncmpii_NC_check_id(ncp->ncid_sf, &ncp_sf);
        CHECK_ERROR(status)

        status = NC_begins(ncp_sf, h_align, h_minfree, v_align, v_minfree,
                           r_align);
        CHECK_ERROR(status)
    }
#endif

    if (ncp->old != NULL) {
        /* The current define mode was entered from ncmpi_redef, not from
         * ncmpi_create. We must check if header has been expanded.
         */

        assert(!NC_IsNew(ncp));
        assert(fIsSet(ncp->flags, NC_INDEF));
        assert(ncp->begin_rec >= ncp->old->begin_rec);
        assert(ncp->begin_var >= ncp->old->begin_var);
        assert(ncp->vars.ndefined >= ncp->old->vars.ndefined);
        /* ncp->numrecs has already sync-ed in ncmpi_redef */

        if (ncp->vars.ndefined > 0) { /* no. record and non-record variables */
            if (ncp->begin_var > ncp->old->begin_var) {
                /* header size increases, shift the entire data part down */
                /* shift record variables first */
                status = move_recs_r(ncp, ncp->old);
                CHECK_ERROR(status)

                /* shift non-record variables */
                /* status = move_vars_r(ncp, ncp->old); */
                status = ncmpiio_move_fixed_vars(ncp, ncp->old);
                CHECK_ERROR(status)
            }
            else if (ncp->begin_rec > ncp->old->begin_rec ||
                     ncp->recsize   > ncp->old->recsize) {
                /* number of non-record variables increases, or
                   number of records of record variables increases,
                   shift and move all record variables down */
                status = move_recs_r(ncp, ncp->old);
                CHECK_ERROR(status)
            }
        }
    } /* ... ncp->old != NULL */

    /* first sync header objects in memory across all processes, and then root
     * writes the header to file. Note safe_mode error check is already done
     * in write_NC() */
    status = write_NC(ncp);

    /* we should continue to exit define mode, even if header is inconsistent
     * among processes, so the program can proceed, say to close file properly.
     * However, if ErrIsHeaderDiff(status) is true, this error should
     * be considered fatal, as inconsistency is about the data structure,
     * rather then contents (such as attribute values) */

#ifdef ENABLE_SUBFILING
    /* write header to subfile */
    if (ncp->nc_num_subfiles > 1) {
        err = write_NC(ncp_sf);
        if (status == NC_NOERR) status = err;
    }
#endif

    /* update the total number of record variables */
    ncp->vars.num_rec_vars = 0;
    for (i=0; i<ncp->vars.ndefined; i++)
        ncp->vars.num_rec_vars += IS_RECVAR(ncp->vars.value[i]);

    /* fill variables according to their fill mode settings */
    err = ncmpii_fill_vars(ncp);
    if (status == NC_NOERR) status = err;

    if (ncp->old != NULL) {
        ncmpii_free_NC(ncp->old);
        ncp->old = NULL;
    }
    fClr(ncp->flags, NC_CREAT | NC_INDEF);

#ifdef ENABLE_SUBFILING
    if (ncp->nc_num_subfiles > 1)
        fClr(ncp_sf->flags, NC_CREAT | NC_INDEF);
#endif

    /* If the user sets NC_SHARE, we enforce a stronger data consistency */
    if (NC_doFsync(ncp))
        ncmpiio_sync(ncp->nciop); /* calling MPI_File_sync() */

    return status;
}

/*----< ncmpii_inq_env_align_hints() >---------------------------------------*/
/* check if environment variable PNETCDF_HINTS sets any of the following hints.
 * nc_header_align_size      if yes, set its value to h_align
 * nc_var_align_size         if yes, set its value to v_align
 * nc_header_read_chunk_size if yes, set its value to h_chunk
 * nc_record_align_size      if yes, set its value to r_align
 */
static int
ncmpii_inq_env_align_hints(MPI_Offset *h_align,
                           MPI_Offset *v_align,
                           MPI_Offset *h_chunk,
                           MPI_Offset *r_align)
{
    char *env_str;

    *h_align = 0;
    *v_align = 0;
    *h_chunk = 0;
    *r_align = 0;

    if ((env_str = getenv("PNETCDF_HINTS")) != NULL) {
        char *key = strtok(env_str, ";");
        while (key != NULL) {
            char *val = strchr(key, '=');
            *val = '\0';
            val++;
            if (strcasecmp(key, "nc_header_align_size") == 0)
                *h_align = atoll(val);
            else if (strcasecmp(key, "nc_var_align_size") == 0)
                *v_align = atoll(val);
            else if (strcasecmp(key, "nc_header_read_chunk_size") == 0)
                *h_chunk = atoll(val);
            else if (strcasecmp(key, "nc_record_align_size") == 0)
                *r_align = atoll(val);
            key = strtok(NULL, ";");
        }
    }
    return 1;
}

/*----< ncmpii_enddef() >----------------------------------------------------*/
int
ncmpii_enddef(NC *ncp)
{
    int i, flag, striping_unit;
    char value[MPI_MAX_INFO_VAL];
    MPI_Offset h_align, v_align, r_align, all_var_size;
    MPI_Offset env_h_align, env_v_align, env_h_chunk, env_r_align;

    assert(!NC_readonly(ncp));
    assert(NC_indef(ncp));

    /* calculate a good align size for PnetCDF level hints:
     * header_align_size and var_align_size based on the MPI-IO hint
     * striping_unit. This hint can be either supplied by the user or obtained
     * from MPI-IO (for example, ROMIO's Lustre driver makes a system call to
     * get the striping parameters of a file).
     */
    MPI_Info_get(ncp->nciop->mpiinfo, "striping_unit", MPI_MAX_INFO_VAL-1,
                 value, &flag);
    striping_unit = 0;
    if (flag) striping_unit = atoi(value);
    ncp->nciop->striping_unit = striping_unit;

    all_var_size = 0;  /* sum of all defined variables */
    for (i=0; i<ncp->vars.ndefined; i++)
        all_var_size += ncp->vars.value[i]->len;

    /* check if any hints have been set in the environment variable
     * PNETCDF_HINTS, as they will overwrite the ones set in the MPI_info
     * provided in ncmpi_creat() or ncmpi_open() */
    ncmpii_inq_env_align_hints(&env_h_align, &env_v_align, &env_h_chunk,
                               &env_r_align);

    /* align file offsets for file header space and fixed variables.
       These alignment hints have been extracted from the MPI_Info object when
       ncmpi_create() is called.
     */
    h_align = (env_h_align == 0) ? ncp->nciop->hints.h_align : env_h_align;
    v_align = (env_v_align == 0) ? ncp->nciop->hints.v_align : env_v_align;
    r_align = (env_r_align == 0) ? ncp->nciop->hints.r_align : env_r_align;

    if (h_align == 0) { /* user info does not set hint nc_header_align_size */
        if (striping_unit &&
            all_var_size > HEADER_ALIGNMENT_LB * striping_unit)
            /* if striping_unit is available and file size sufficiently large */
            h_align = striping_unit;
        else
            h_align = DEFAULT_ALIGNMENT;
    }
    /* else respect user hint */

    if (v_align == 0) { /* user info does not set hint nc_var_align_size */
        if (striping_unit &&
            all_var_size > HEADER_ALIGNMENT_LB * striping_unit)
            /* if striping_unit is available and file size sufficiently large */
            v_align = striping_unit;
        else
            v_align = DEFAULT_ALIGNMENT;
    }
    /* else respect user hint */

    if (r_align == 0) { /* user info does not set hint nc_record_align_size */
        if (striping_unit)
            r_align = striping_unit;
        else
            r_align = DEFAULT_ALIGNMENT;
    }
    /* else respect user hint */

    return ncmpii_NC_enddef(ncp, h_align, 0, v_align, 0, r_align);
}

/*----< ncmpii__enddef() >---------------------------------------------------*/
int
ncmpii__enddef(NC         *ncp,
               MPI_Offset  h_minfree,
               MPI_Offset  v_align,
               MPI_Offset  v_minfree,
               MPI_Offset  r_align)
{
    int i, flag, striping_unit;
    char value[MPI_MAX_INFO_VAL];
    MPI_Offset h_align, all_var_size;
    MPI_Offset env_h_align, env_v_align, env_h_chunk, env_r_align;

    assert(!NC_readonly(ncp));
    assert(NC_indef(ncp));

    /* calculate a good align size for PnetCDF level hints:
     * header_align_size and var_align_size based on the MPI-IO hint
     * striping_unit. This hint can be either supplied by the user or obtained
     * from MPI-IO (for example, ROMIO's Lustre driver makes a system call to
     * get the striping parameters of a file).
     */
    MPI_Info_get(ncp->nciop->mpiinfo, "striping_unit", MPI_MAX_INFO_VAL-1,
                 value, &flag);
    striping_unit = 0;
    if (flag) striping_unit = atoi(value);
    ncp->nciop->striping_unit = striping_unit;

    all_var_size = 0;  /* sum of all defined variables */
    for (i=0; i<ncp->vars.ndefined; i++)
        all_var_size += ncp->vars.value[i]->len;

    /* check if any hints have been set in the environment variable
     * PNETCDF_HINTS, as they will overwrite the ones passed in this function
     * and those passed-in alignment values will overwrite the ones set in the
     * MPI_info provided in ncmpi_creat() or ncmpi_open() */
    ncmpii_inq_env_align_hints(&env_h_align, &env_v_align, &env_h_chunk,
                               &env_r_align);

    /* align file offsets for file header space and fixed variables */
    h_align = (env_h_align == 0) ? ncp->nciop->hints.h_align : env_h_align;
    v_align = (env_v_align == 0) ? v_align                   : env_v_align;
    r_align = (env_r_align == 0) ? r_align                   : env_r_align;

    if (v_align == 0) /* 0 means let PnetCDF decide, 1 means no align */
        v_align = ncp->nciop->hints.v_align;

    if (r_align == 0) /* 0 means let PnetCDF decide, 1 means no align */
        r_align = ncp->nciop->hints.r_align;

    if (h_align == 0) { /* user does not set hint nc_header_align_size */
        if (striping_unit &&
            all_var_size > HEADER_ALIGNMENT_LB * striping_unit)
            /* if striping_unit is available and file size sufficiently large */
            h_align = striping_unit;
        else
            h_align = DEFAULT_ALIGNMENT;
    }
    /* else respect user hint */

    if (v_align == 0) { /* user does not set hint nc_var_align_size */
        if (striping_unit &&
            all_var_size > HEADER_ALIGNMENT_LB * striping_unit)
            /* if striping_unit is available and file size sufficiently large */
            v_align = striping_unit;
        else
            v_align = DEFAULT_ALIGNMENT;
    }
    /* else respect user hint */

    if (r_align == 0) { /* user does not set hint nc_record_align_size */
        if (striping_unit)
            r_align = striping_unit;
        else
            r_align = DEFAULT_ALIGNMENT;
    }
    /* else respect user hint */

    return ncmpii_NC_enddef(ncp, h_align, h_minfree, v_align, v_minfree,
                            r_align);
}

/*----< ncmpii_close() >------------------------------------------------------*/
/* This function is collective */
int
ncmpii_close(NC *ncp)
{
    int err, status=NC_NOERR;

    if (NC_indef(ncp)) { /* currently in define mode */
        status = ncmpii_enddef(ncp); /* TODO: defaults */
        if (status != NC_NOERR ) {
            /* To do: Abort new definition, if any */
            if (ncp->old != NULL) {
                ncmpii_free_NC(ncp->old);
                ncp->old = NULL;
                fClr(ncp->flags, NC_INDEF);
            }
        }
    }

    if (!NC_readonly(ncp) &&  /* file is open for write */
         NC_indep(ncp)) {     /* exit independent data mode will sync header */
        err = ncmpii_end_indep_data(ncp);
        if (status == NC_NOERR ) status = err;
    }

    /* if entering this function in  collective data mode, we do not have to
     * update header in file, as file header is always up-to-date */

#ifdef ENABLE_SUBFILING
    /* ncmpii_enddef() will update nc_num_subfiles */
    /* TODO: should check ncid_sf? */
    /* if the file has subfiles, close them first */
    if (ncp->nc_num_subfiles > 1)
        ncmpii_subfile_close(ncp);
#endif

    /* We can cancel or complete all outstanding nonblocking I/O.
     * For now, cancelling makes more sense. */
    if (ncp->head != NULL) {
#ifdef COMPLETE_NONBLOCKING_IO
        ncmpii_wait(ncp, INDEP_IO, NC_REQ_ALL, NULL, NULL);
#else
        ncmpii_cancel(ncp, NC_REQ_ALL, NULL, NULL);
#endif
    }

    /* If the user wants a stronger data consistency by setting NC_SHARE */
    if (fIsSet(ncp->nciop->ioflags, NC_SHARE))
        ncmpiio_sync(ncp->nciop); /* calling MPI_File_sync() */

    /* calling MPI_File_close() */
    ncmpiio_close(ncp->nciop, 0);
    ncp->nciop = NULL;

    /* remove this file from the list of opened files */
    ncmpii_del_from_NCList(ncp);

    /* free up space occupied by the header metadata */
    ncmpii_free_NC(ncp);

    return status;
}

/* Public */

int
ncmpi_inq(int  ncid,
          int *ndimsp,
          int *nvarsp,
          int *nattsp,
          int *xtendimp)
{
    int status;
    NC *ncp;

    status = ncmpii_NC_check_id(ncid, &ncp);
    if (status != NC_NOERR)
        return status;

    if (ndimsp != NULL)
        *ndimsp = (int) ncp->dims.ndefined;
    if (nvarsp != NULL)
        *nvarsp = (int) ncp->vars.ndefined;
    if (nattsp != NULL)
        *nattsp = (int) ncp->attrs.ndefined;
    if (xtendimp != NULL)
        *xtendimp = ncmpii_find_NC_Udim(&ncp->dims, NULL);

    return NC_NOERR;
}

/*----< ncmpi_inq_version() >-----------------------------------------------*/
int
ncmpi_inq_version(int ncid, int *nc_mode)
{
    int status;
    NC *ncp;

    status = ncmpii_NC_check_id(ncid, &ncp);
    if (status != NC_NOERR)
        return status;

    if (fIsSet(ncp->flags, NC_64BIT_DATA))
        *nc_mode = NC_64BIT_DATA;
    else if (fIsSet(ncp->flags, NC_64BIT_OFFSET))
        *nc_mode = NC_64BIT_OFFSET;
    else
        *nc_mode = NC_CLASSIC_MODEL;

    return NC_NOERR;
}


int
ncmpi_inq_ndims(int ncid, int *ndimsp)
{
    int status;
    NC *ncp;

    status = ncmpii_NC_check_id(ncid, &ncp);
    if (status != NC_NOERR) return status;

    if (ndimsp != NULL)
        *ndimsp = (int) ncp->dims.ndefined;

    return NC_NOERR;
}

int
ncmpi_inq_nvars(int ncid, int *nvarsp)
{
    int status;
    NC *ncp;

    status = ncmpii_NC_check_id(ncid, &ncp);
    if (status != NC_NOERR) return status;

    if (nvarsp != NULL)
        *nvarsp = (int) ncp->vars.ndefined;

    return NC_NOERR;
}

int
ncmpi_inq_natts(int ncid, int *nattsp)
{
    int status;
    NC *ncp;

    status = ncmpii_NC_check_id(ncid, &ncp);
    if (status != NC_NOERR) return status;

    if (nattsp != NULL)
        *nattsp = (int) ncp->attrs.ndefined;

    return NC_NOERR;
}

int
ncmpi_inq_unlimdim(int ncid, int *xtendimp)
{
    int status;
    NC *ncp;

    status = ncmpii_NC_check_id(ncid, &ncp);
    if (status != NC_NOERR) return status;

    if (xtendimp != NULL)
        *xtendimp = ncmpii_find_NC_Udim(&ncp->dims, NULL);

    return NC_NOERR;
}

/*----< ncmpi_inq_num_rec_vars() >-------------------------------------------*/
int
ncmpi_inq_num_rec_vars(int ncid, int *nvarsp)
{
    int i, status;
    NC *ncp;

    /* get ncp object */
    status = ncmpii_NC_check_id(ncid, &ncp);
    if (status != NC_NOERR)
        return status;

    if (nvarsp != NULL) {
        if (NC_indef(ncp)) {
            /* if in define mode, recalculate the number of record variables */
            *nvarsp = 0;
            for (i=0; i<ncp->vars.ndefined; i++)
                *nvarsp += IS_RECVAR(ncp->vars.value[i]);
        }
        else
            *nvarsp = ncp->vars.num_rec_vars;
    }

    return NC_NOERR;
}

/*----< ncmpi_inq_num_fix_vars() >-------------------------------------------*/
int
ncmpi_inq_num_fix_vars(int ncid, int *nvarsp)
{
    int i, status;
    NC *ncp;

    /* get ncp object */
    status = ncmpii_NC_check_id(ncid, &ncp);
    if (status != NC_NOERR)
        return status;

    if (nvarsp != NULL) {
        if (NC_indef(ncp)) {
            /* if in define mode, recalculate the number of record variables */
            *nvarsp = 0;
            for (i=0; i<ncp->vars.ndefined; i++)
                *nvarsp += IS_RECVAR(ncp->vars.value[i]);
        }
        else
            *nvarsp = ncp->vars.num_rec_vars;

        /* no. fixed-size == ndefined - no. record variables */
        *nvarsp = ncp->vars.ndefined- *nvarsp;
    }

    return NC_NOERR;
}


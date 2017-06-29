/*
 *  Copyright (C) 2015, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id: fill.c 2320 2016-02-24 13:18:32Z wkliao $ */

#if HAVE_CONFIG_H
# include "ncconfig.h"
#endif

#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#include <stdio.h>
#include <string.h>

#include <mpi.h>

#include "nc.h"
#include "macro.h"

#define CHECK_ERROR(status) {                                                \
    if (ncp->safe_mode == 1) {                                               \
        int g_status;                                                        \
        TRACE_COMM(MPI_Allreduce)(&status, &g_status, 1, MPI_INT, MPI_MIN,   \
                                  ncp->nciop->comm);                         \
        if (g_status != NC_NOERR) return status;                             \
    }                                                                        \
    else if (status != NC_NOERR)                                             \
        return status;                                                       \
}

/* The default fill values defined in pnetcdf.h.inc must be the same as the
 * ones defined in netCDF-4 and match the hexadecimal values set below
 *
#define NC_FILL_BYTE    ((signed char)-127)
#define NC_FILL_CHAR    ((char)0)
#define NC_FILL_SHORT   ((short)-32767)
#define NC_FILL_INT     (-2147483647L)
#define NC_FILL_FLOAT   (9.9692099683868690e+36f)
#define NC_FILL_DOUBLE  (9.9692099683868690e+36)
#define NC_FILL_UBYTE   (255)
#define NC_FILL_USHORT  (65535)
#define NC_FILL_UINT    (4294967295U)
#define NC_FILL_INT64   ((long long)-9223372036854775806LL)
#define NC_FILL_UINT64  ((unsigned long long)18446744073709551614ULL)
*/
static unsigned char FILL_CHAR[1]   = {0x00};
static unsigned char FILL_BYTE[1]   = {0x81};
static unsigned char FILL_SHORT[2]  = {0x80, 0x01};
static unsigned char FILL_INT[4]    = {0x80, 0x00, 0x00, 0x01};
static unsigned char FILL_FLOAT[4]  = {0x7C, 0xF0, 0x00, 0x00};
static unsigned char FILL_DOUBLE[8] = {0x47, 0x9E, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
static unsigned char FILL_UBYTE[1]  = {0xFF};
static unsigned char FILL_USHORT[2] = {0xFF, 0xFF};
static unsigned char FILL_UINT[4]   = {0xFF, 0xFF, 0xFF, 0xFF};
static unsigned char FILL_INT64[8]  = {0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02};
static unsigned char FILL_UINT64[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE};

/*----< inq_default_fill_value() >-------------------------------------------*/
/* copy the default fill value to the memory space pointed by fill_value */
static int
inq_default_fill_value(int type, void *fill_value)
{
    if (fill_value == NULL) return NC_NOERR;

    switch(type) {
        case NC_CHAR   :               *(char*)fill_value = NC_FILL_CHAR;   break;
        case NC_BYTE   :        *(signed char*)fill_value = NC_FILL_BYTE;   break;
        case NC_SHORT  :              *(short*)fill_value = NC_FILL_SHORT;  break;
        case NC_INT    :                *(int*)fill_value = NC_FILL_INT;    break;
        case NC_FLOAT  :              *(float*)fill_value = NC_FILL_FLOAT;  break;
        case NC_DOUBLE :             *(double*)fill_value = NC_FILL_DOUBLE; break;
        case NC_UBYTE  :      *(unsigned char*)fill_value = NC_FILL_UBYTE;  break;
        case NC_USHORT :     *(unsigned short*)fill_value = NC_FILL_USHORT; break;
        case NC_UINT   :       *(unsigned int*)fill_value = NC_FILL_UINT;   break;
        case NC_INT64  :          *(long long*)fill_value = NC_FILL_INT64;  break;
        case NC_UINT64 : *(unsigned long long*)fill_value = NC_FILL_UINT64; break;
        default : DEBUG_RETURN_ERROR(NC_EBADTYPE)
    }
    return NC_NOERR;
}

/*----< ncmpii_fill_var_buf() >----------------------------------------------*/
/* fill the buffer, buf, with either user-defined fill values or default
 * values */
static int
ncmpii_fill_var_buf(const NC_var *varp,
                    MPI_Offset    bnelems, /* number of elements in buf */
                    void         *buf)
{
    int i, indx;

    indx = ncmpii_NC_findattr(&varp->attrs, _FillValue);
    if (indx >= 0) {
        /* User defined fill value */
        NC_attr *attrp = varp->attrs.value[indx];
        if (attrp->type != varp->type || attrp->nelems != 1)
            DEBUG_RETURN_ERROR(NC_EBADTYPE)

        /* Use the user defined value */
        char *bufp = buf;
        for (i=0; i<bnelems; i++) {
            memcpy(bufp, attrp->xvalue, (size_t)varp->xsz);
            bufp += varp->xsz;
        }
    }
    else { /* use the default */
        void *xvalue;
        switch(varp->type) {
            case NC_CHAR   : xvalue = &FILL_CHAR[0];   break;
            case NC_BYTE   : xvalue = &FILL_BYTE[0];   break;
            case NC_SHORT  : xvalue = &FILL_SHORT[0];  break;
            case NC_INT    : xvalue = &FILL_INT[0];    break;
            case NC_FLOAT  : xvalue = &FILL_FLOAT[0];  break;
            case NC_DOUBLE : xvalue = &FILL_DOUBLE[0]; break;
            case NC_UBYTE  : xvalue = &FILL_UBYTE[0];  break;
            case NC_USHORT : xvalue = &FILL_USHORT[0]; break;
            case NC_UINT   : xvalue = &FILL_UINT[0];   break;
            case NC_INT64  : xvalue = &FILL_INT64[0];  break;
            case NC_UINT64 : xvalue = &FILL_UINT64[0]; break;
            default : DEBUG_RETURN_ERROR(NC_EBADTYPE)
        }

        char *bufp = buf;
        for (i=0; i<bnelems; i++) {
            memcpy(bufp, xvalue, (size_t)varp->xsz);
            bufp += varp->xsz;
        }
    }
    return NC_NOERR;
}

/*----< ncmpii_fill_var() >--------------------------------------------------*/
/* For fixed-size variables, write the entire variable with fill values and
 * ignore argument recno. For record variables, write one record of that
 * variable with pre-defined/supplied fill value.
 */
static int
ncmpii_fill_var_rec(NC         *ncp,
                    NC_var     *varp,
                    MPI_Offset  recno) /* record number */
{
    int err, mpireturn, rank, nprocs;
    void *buf;
    MPI_Offset var_len, start, count, offset;
    MPI_File fh;
    MPI_Status mpistatus;

    MPI_Comm_rank(ncp->nciop->comm, &rank);
    MPI_Comm_size(ncp->nciop->comm, &nprocs);

    if (varp->ndims == 0) /* scalar variable */
        var_len = 1;
    else if (varp->ndims == 1 && IS_RECVAR(varp))
        var_len = 1;
    else if (IS_RECVAR(varp))
        var_len = varp->dsizes[1];
    else
        var_len = varp->dsizes[0];

    /* divide total number of elements of this variable among all processes */
    count = var_len / nprocs;
    start = count * rank;
    if (rank < var_len % nprocs) {
        start += rank;
        count++;
    }
    else {
        start += var_len % nprocs;
    }

    /* allocate buffer space */
    buf = NCI_Malloc((size_t)(count * varp->xsz));

    /* fill buffer with fill values */
    err = ncmpii_fill_var_buf(varp, count, buf);
    if (err != NC_NOERR) {
        NCI_Free(buf);
        return err;
    }

    /* calculate the starting file offset for each process */
    offset = varp->begin;
    if (IS_RECVAR(varp))
        offset += ncp->recsize * recno;
    offset += start * varp->xsz;

    fh = ncp->nciop->collective_fh;

    /* make the entire file visible */
    TRACE_IO(MPI_File_set_view)(fh, 0, MPI_BYTE, MPI_BYTE, "native",
                                MPI_INFO_NULL);

    count *= varp->xsz;
    if (count != (int)count) DEBUG_ASSIGN_ERROR(err, NC_EINTOVERFLOW)
    if (err != NC_NOERR)
        count = 0; /* participate collective write with 0-length request */

    /* write to variable collectively */
    TRACE_IO(MPI_File_write_at_all)(fh, offset, buf, (int)count,
                                    MPI_BYTE, &mpistatus);
    NCI_Free(buf);

    if (err != NC_NOERR) return err;

    if (mpireturn != MPI_SUCCESS)
        return ncmpii_handle_error(mpireturn, "MPI_File_write_at_all");

    if (IS_RECVAR(varp)) { /* update header's number of records in memory */
        err = ncmpii_sync_numrecs(ncp, recno+1);
        if (err == NC_NOERR) return err;
    }

    return NC_NOERR;
}

/*----< ncmpi_fill_var() >---------------------------------------------------*/
/* fill an entire record of a record variable
 * this API is collective, must be called in data mode */
int
ncmpi_fill_var_rec(int        ncid,
                   int        varid,
                   MPI_Offset recno) /* record number, ignored if non-record var */
{
    int     indx, err;
    NC     *ncp;
    NC_var *varp;

    /* check if ncid is valid */
    err = ncmpii_NC_check_id(ncid, &ncp);
    if (err != NC_NOERR) return err;

    if (NC_readonly(ncp)) DEBUG_RETURN_ERROR(NC_EPERM) /* read-only */

    /* This must be called in data mode */
    if (NC_indef(ncp)) DEBUG_RETURN_ERROR(NC_EINDEFINE)                                

    /* must be called in collective data mode */
    if (NC_indep(ncp)) DEBUG_RETURN_ERROR(NC_EINDEP)

    err = ncmpii_NC_lookupvar(ncp, varid, &varp);
    if (err != NC_NOERR) return err;

    /* error if this is not a record variable */
    if (!IS_RECVAR(varp)) DEBUG_RETURN_ERROR(NC_ENOTRECVAR)

    /* check if _FillValue attribute is defined */
    indx = ncmpii_NC_findattr(&varp->attrs, _FillValue);

    /* error if the fill mode of this variable is not on */
    if (varp->no_fill && indx == -1) DEBUG_RETURN_ERROR(NC_ENOTFILL)

    return ncmpii_fill_var_rec(ncp, varp, recno);
}

/*----< ncmpi_set_fill() >---------------------------------------------------*/
/* this API is collective, must be called in define mode, contrary to netCDF
 * where nc_set_fill() can also be called in data mode. The reason of PnetCDF
 * enforcing this requirement is because PnetCDF only fills fix-sized variables
 * at ncmpi_enddef() and record variables in ncmpi_fill_var_rec().
 */
int
ncmpi_set_fill(int  ncid,
               int  fill_mode,
               int *old_fill_mode)
{
    int i, status=NC_NOERR, mpireturn, oldmode;
    NC *ncp;

    status = ncmpii_NC_check_id(ncid, &ncp);
    if (status != NC_NOERR) return status;

    if (NC_readonly(ncp)) DEBUG_RETURN_ERROR(NC_EPERM) /* read-only */

    /* check if called in define mode */
    if (!NC_indef(ncp)) DEBUG_RETURN_ERROR(NC_ENOTINDEFINE)

    if (ncp->safe_mode) {
        int root_fill_mode=fill_mode;
        TRACE_COMM(MPI_Bcast)(&root_fill_mode, 1, MPI_INT, 0, ncp->nciop->comm);
        if (mpireturn != MPI_SUCCESS)
            return  ncmpii_handle_error(mpireturn, "MPI_Bcast"); 
        if (fill_mode != root_fill_mode) {
            /* dataset's fill mode is inconsistent with root's */
            printf("Warning: fill mode set in %s() is inconsistent\n", __func__);
            DEBUG_ASSIGN_ERROR(status, NC_EMULTIDEFINE_FILL_MODE)
        }
    }

    oldmode = fIsSet(ncp->flags, NC_NOFILL) ? NC_NOFILL : NC_FILL;

    if (fill_mode == NC_NOFILL)
        fSet(ncp->flags, NC_NOFILL);
    else if (fill_mode == NC_FILL)
        fClr(ncp->flags, NC_NOFILL);
    else
        DEBUG_RETURN_ERROR(NC_EINVAL) /* Invalid fill_mode */

    if (old_fill_mode != NULL) *old_fill_mode = oldmode;

    /* loop thru all variables defined so far to set/overwrite its fill mode */
    for (i=0; i<ncp->vars.ndefined; i++)
        ncp->vars.value[i]->no_fill = (fill_mode == NC_NOFILL) ? 1 : 0;

    /* once the file's fill mode is set, any new variables defined after this
     * call will check NC_dofill(ncp) and set their no_fill accordingly. See
     * ncmpi_def_var() */

    return status;
}

/*----< ncmpi_def_var_fill() >------------------------------------------------*/
/* this API is collective, and must be called in define mode */
int
ncmpi_def_var_fill(int   ncid,
                   int   varid,
                   int   no_fill,    /* 1: no fill, 0: fill */
                   void *fill_value) /* when NULL, use default fill value */
{
    int err, status=NC_NOERR, mpireturn, free_fill_value=0;
    NC *ncp;
    NC_var *varp;

    err = ncmpii_NC_check_id(ncid, &ncp);
    if (err != NC_NOERR) return err;

    if (NC_readonly(ncp)) DEBUG_RETURN_ERROR(NC_EPERM) /* read-only */

    /* must be called in define mode */
    if (!NC_indef(ncp)) DEBUG_RETURN_ERROR(NC_ENOTINDEFINE)

    /* find the pointer to this variable's object */
    err = ncmpii_NC_lookupvar(ncp, varid, &varp);
    if (err != NC_NOERR) return err;

    if (ncp->safe_mode) {
        int root_no_fill=no_fill;
        TRACE_COMM(MPI_Bcast)(&root_no_fill, 1, MPI_INT, 0, ncp->nciop->comm);
        if (mpireturn != MPI_SUCCESS)
            return  ncmpii_handle_error(mpireturn, "MPI_Bcast"); 

        if (no_fill != root_no_fill) {
            /* variable's fill mode is inconsistent with root's */
            printf("Warning: variable (%s) fill mode (%d) set in %s() is inconsistent\n",
                   varp->name->cp, no_fill, __func__);
            if (status == NC_NOERR) DEBUG_ASSIGN_ERROR(status, NC_EMULTIDEFINE_VAR_FILL_MODE)
        }

        /* check if fill_value is consistent among processes */
        void *root_fill_value = NCI_Malloc((size_t)varp->xsz);
        if (fill_value == NULL) {
            /* user intends to use default fill value */
            fill_value = NCI_Malloc((size_t)varp->xsz);
            err = inq_default_fill_value(varp->type, fill_value);
            if (err != NC_NOERR) {
                NCI_Free(fill_value);
                NCI_Free(root_fill_value);
                return err;
            }
            free_fill_value=1;
        }
        memcpy(root_fill_value, fill_value, (size_t)varp->xsz);
            
        TRACE_COMM(MPI_Bcast)(root_fill_value, varp->xsz, MPI_BYTE, 0, ncp->nciop->comm);
        if (memcmp(fill_value, root_fill_value, (size_t)varp->xsz)) {
            /* variable's fill value is inconsistent with root's */
            printf("Warning: variable (%s) fill value set in %s() is inconsistent\n",
                   varp->name->cp, __func__);
            if (status == NC_NOERR) DEBUG_ASSIGN_ERROR(status, NC_EMULTIDEFINE_VAR_FILL_VALUE)
        }
        if (free_fill_value) {
            NCI_Free(fill_value);
            fill_value = NULL;
        }
        NCI_Free(root_fill_value);
    }

    if (no_fill)
        varp->no_fill = 1;
    else
        varp->no_fill = 0;

    /* Are we setting a fill value? */
    if (fill_value != NULL && !varp->no_fill) {

        /* If there's a _FillValue attribute, delete it. */
        err = ncmpi_del_att(ncid, varid, _FillValue);
        if (err != NC_NOERR && err != NC_ENOTATT)
            return err;

        /* Create a _FillValue attribute. */
        err = ncmpi_put_att(ncid, varid, _FillValue, varp->type, 1, fill_value);
        if (err != NC_NOERR) return err;
    }

    return status;
}

/*----< ncmpi_inq_var_fill() >-----------------------------------------------*/
/* this API can be called independently and in both data and define mode */
int
ncmpi_inq_var_fill(int   ncid,
                   int   varid,
                   int  *no_fill,    /* OUT: 1 not fill mode, 0 fill mode */
                   void *fill_value) /* OUT: user-defined or default fill value */
{
    int err;
    NC *ncp;
    NC_var *varp;

    err = ncmpii_NC_check_id(ncid, &ncp);
    if (err != NC_NOERR) return err;

    /* find the pointer to this variable's object */
    err = ncmpii_NC_lookupvar(ncp, varid, &varp);
    if (err != NC_NOERR) return err;

    *no_fill = varp->no_fill;

    if (fill_value != NULL) {
        err = ncmpi_get_att(ncid, varid, _FillValue, fill_value);
        if (err != NC_NOERR && err != NC_ENOTATT)
            return err;
        if (err == NC_ENOTATT) {
            err = inq_default_fill_value(varp->type, fill_value);
            if (err != NC_NOERR) return err;
        }
    }

    return NC_NOERR;
}

#ifdef FILL_ONE_VAR_AT_A_TIME
/*----< fillerup() >---------------------------------------------------------*/
/* fill the newly created fixed-size variables */
static int
fillerup(NC *ncp)
{
    int i, indx, err=NC_NOERR;

    /* loop thru all variables */
    for (i=0; i<ncp->vars.ndefined; i++) {
        if (IS_RECVAR(ncp->vars.value[i]))
            /* skip record variables. In PnetCDF record variables mus be
             * explicitly filled by calling ncmpi_fill_var_rec() */
            continue;

        /* check if _FillValue attribute is defined */
        indx = ncmpii_NC_findattr(&ncp->vars.value[i]->attrs, _FillValue);

        /* only if filling this variable is requested. Fill mode can be
         * enabled by 2 ways: explictly call to ncmpi_def_var_fill() or put
         * the attribute named _FillValue */
        if (ncp->vars.value[i]->no_fill && indx == -1) continue;

        /* collectively fill the entire variable */
        err = ncmpii_fill_var_rec(ncp, ncp->vars.value[i], 0);
        if (err != NC_NOERR) break;
    }
    return err;
}

/*----< fill_added() >-------------------------------------------------------*/
/* fill the newly added variables */
static int
fill_added(NC *ncp, NC *old_ncp)
{
    int indx, err=NC_NOERR, varid;

    /* loop thru all new variables */
    varid = old_ncp->vars.ndefined;
    for (; varid<ncp->vars.ndefined; varid++) {
        if (IS_RECVAR(ncp->vars.value[varid]))
            /* skip record variables */
            continue;

        /* check if _FillValue attribute is defined */
        indx = ncmpii_NC_findattr(&ncp->vars.value[varid]->attrs, _FillValue);

        /* only if filling this variable is requested */
        if (ncp->vars.value[varid]->no_fill && indx == -1) continue;

        /* collectively fill the entire variable */
        err = ncmpii_fill_var_rec(ncp, ncp->vars.value[varid], 0);
        if (err != NC_NOERR) break;
    }
    return err;
}

/*----< fill_added_recs() >--------------------------------------------------*/
/* for each newly added record variable, we fill the records one at a time */
static int
fill_added_recs(NC *ncp, NC *old_ncp)
{
    MPI_Offset old_nrecs = NC_get_numrecs(old_ncp);
    int indx, err, recno, varid;

    /* loop thru all old records */
    for (recno=0; recno<old_nrecs; recno++) {
        /* check newly added variables only */
        for (varid=old_ncp->vars.ndefined; varid<ncp->vars.ndefined; varid++) {
            if (!IS_RECVAR(ncp->vars.value[varid]))
                /* skip non-record variables */
                continue;

            /* check if _FillValue attribute is defined */
            indx = ncmpii_NC_findattr(&ncp->vars.value[varid]->attrs, _FillValue);

            /* only if filling this variable is requested */
            if (ncp->vars.value[varid]->no_fill && indx == -1) continue;

            /* collectively fill the record */
            err = ncmpii_fill_var_rec(ncp, ncp->vars.value[varid], recno);
            if (err != NC_NOERR) return err;
        }
    }
    return NC_NOERR;
}
#endif

/*----< fillerup_aggregate() >-----------------------------------------------*/
/* fill the newly created/added variables
 * This version aggregates all writes into a single one
 */
static int
fillerup_aggregate(NC *ncp, NC *old_ncp)
{
    int i, j, k, rank, nprocs, start_vid, nsegs, recno;
    int nVarsFill, *blocklengths;
    int mpireturn, err, status=NC_NOERR;
    char *buf_ptr, *noFill;
    void *buf;
    size_t buf_len;
    MPI_Offset var_len, nrecs, start, *count;
    MPI_Aint *offset;
    MPI_Datatype filetype;
    MPI_File fh;
    MPI_Status mpistatus;
    NC_var *varp;

    MPI_Comm_rank(ncp->nciop->comm, &rank);
    MPI_Comm_size(ncp->nciop->comm, &nprocs);

    /* find the starting vid for newly added variables */
    start_vid = 0;
    nrecs = 0;  /* the current number of records */
    if (old_ncp != NULL) {
        start_vid = old_ncp->vars.ndefined;
        nrecs = NC_get_numrecs(old_ncp);
    }

    /* Because fill mode is not part of file header, we must broadcast root's
     * variables' fill modes and overwrite local's if an inconsistency is found
     * Note ncp->vars.ndefined is already made consistent by this point.
     */
    noFill = (char*) NCI_Malloc(ncp->vars.ndefined - start_vid);
    for (i=start_vid; i<ncp->vars.ndefined; i++)
        noFill[i-start_vid] = ncp->vars.value[i]->no_fill;

    TRACE_COMM(MPI_Bcast)(noFill, (ncp->vars.ndefined - start_vid), MPI_BYTE, 0,
                          ncp->nciop->comm);
    nVarsFill = 0;
    for (i=start_vid; i<ncp->vars.ndefined; i++) { /* overwrite local's mode */
        ncp->vars.value[i]->no_fill = noFill[i-start_vid];
        if (!noFill[i-start_vid]) nVarsFill++;
    }
    if (nVarsFill == 0) { /* no variables in fill mode */
        NCI_Free(noFill);
        return NC_NOERR;
    }

    /* find the number of write segments (upper bound) */
    nsegs = ncp->vars.ndefined + ncp->vars.num_rec_vars * nrecs;
    count  = (MPI_Offset*) NCI_Malloc(nsegs * sizeof(MPI_Offset));
    offset = (MPI_Aint*)   NCI_Malloc(nsegs * sizeof(MPI_Aint));

    /* calculate each segment's offset and count */
    buf_len = 0; /* total write amount, used to allocate buffer */
    j = 0;
    for (i=start_vid; i<ncp->vars.ndefined; i++) {
        if (noFill[i-start_vid]) continue;

        varp = ncp->vars.value[i];

        if (IS_RECVAR(varp)) continue; /* first, fixed-size variables only */

        if (varp->ndims == 0) var_len = 1; /* scalar */
        else                  var_len = varp->dsizes[0];

        /* divide evenly total number of variable's elements among processes */
        count[j] = var_len / nprocs;
        start = count[j] * rank;
        if (rank < var_len % nprocs) {
            start += rank;
            count[j]++;
        }
        else
            start += var_len % nprocs;

        /* calculate the starting file offset */
        start *= varp->xsz;
        start += varp->begin;
        offset[j] = (MPI_Aint)start;
        if (start != offset[j]) {
            DEBUG_ASSIGN_ERROR(err, NC_EINTOVERFLOW)
            if (status == NC_NOERR) status = err;
            noFill[i-start_vid] = 1; /* skip this variable */
            continue;
        }
        /* add up the buffer size */
        buf_len += count[j] * varp->xsz;

        j++;
    }

    /* loop thru all record variables to find the aggregated write amount */
    for (recno=0; recno<nrecs; recno++) {
        for (i=start_vid; i<ncp->vars.ndefined; i++) {
            if (noFill[i-start_vid]) continue;

            varp = ncp->vars.value[i];
            if (!IS_RECVAR(varp)) continue; /* record variables only */

            if (varp->ndims <= 1) var_len = 1;
            else                  var_len = varp->dsizes[1];

            /* divide total number of variable's elements among all processes */
            count[j] = var_len / nprocs;
            start = count[j] * rank;
            if (rank < var_len % nprocs) {
                start += rank;
                count[j]++;
            }
            else
                start += var_len % nprocs;

            /* calculate the starting file offset */
            start *= varp->xsz;
            start += varp->begin + ncp->recsize * recno;
            offset[j] = (MPI_Aint)start;
            if (start != offset[j]) {
                DEBUG_ASSIGN_ERROR(err, NC_EINTOVERFLOW)
                if (status == NC_NOERR) status = err;
                noFill[i-start_vid] = 1; /* skip this variable */
                continue;
            }
            /* add up the buffer size */
            buf_len += count[j] * varp->xsz;

            j++;
        }
    }
    /* j is now the number of valid write segments */

    /* allocate one contiguous buffer space for all writes */
    blocklengths = (int*) NCI_Malloc(j * sizeof(int));
    buf = NCI_Malloc(buf_len);
    buf_ptr = (char*)buf;

    /* fill write buffers for fixed-size variables first */
    j = k = 0;
    for (i=start_vid; i<ncp->vars.ndefined; i++) {
        if (noFill[i-start_vid]) continue;

        varp = ncp->vars.value[i];
        if (IS_RECVAR(varp)) continue;

        if (k < j) {  /* coalesce count[] and offset[] */
            count[k]  = count[j];
            offset[k] = offset[j];
        }
        j++;
        err = ncmpii_fill_var_buf(varp, count[k], buf_ptr);
        if (err != NC_NOERR) {
            if (status == NC_NOERR) status = err;
            continue; /* skip this request */
        }

        count[k] *= varp->xsz;
        if (count[k] != (int)count[k]) {
            DEBUG_ASSIGN_ERROR(err, NC_EINTOVERFLOW)
            if (status == NC_NOERR) status = err;
            continue; /* skip this request */
        }
        buf_ptr += count[k];
        blocklengths[k] = (int)count[k];
        k++;
    }
    /* k is the number of valid write requests thus far */

    /* loop thru all record variables to fill write buffers */
    for (recno=0; recno<nrecs; recno++) {
        for (i=start_vid; i<ncp->vars.ndefined; i++) {
            if (noFill[i-start_vid]) continue;

            varp = ncp->vars.value[i];
            if (!IS_RECVAR(varp)) continue;

            if (k < j) {  /* coalesce count[] and offset[] */
                count[k]  = count[j];
                offset[k] = offset[j];
            }
            j++;
            err = ncmpii_fill_var_buf(varp, count[k], buf_ptr);
            if (err != NC_NOERR) {
                if (status == NC_NOERR) status = err;
                continue; /* skip this request */
            }

            count[k] *= varp->xsz;
            if (count[k] != (int)count[k]) {
                DEBUG_ASSIGN_ERROR(err, NC_EINTOVERFLOW)
                if (status == NC_NOERR) status = err;
                continue; /* skip this request */
            }
            buf_ptr += count[k];
            blocklengths[k] = (int)count[k];
            k++;
        }
    }
    /* k is the number of valid write requests */
    NCI_Free(noFill);

    /* create the fileview: a list of contiguous segment for each variable */
#ifdef HAVE_MPI_TYPE_CREATE_HINDEXED
    mpireturn = MPI_Type_create_hindexed(k, blocklengths, offset, MPI_BYTE,
                                         &filetype);
#else
    mpireturn = MPI_Type_hindexed(k, blocklengths, offset, MPI_BYTE, &filetype);
#endif
    if (mpireturn != MPI_SUCCESS) {
        int err = ncmpii_handle_error(mpireturn, "MPI_Type_hindexed");
        /* return the first encountered error if there is any */
        if (status == NC_NOERR) status = err;
    }
    else
        MPI_Type_commit(&filetype);

    NCI_Free(blocklengths);
    NCI_Free(count);
    NCI_Free(offset);

    fh = ncp->nciop->collective_fh;

    TRACE_IO(MPI_File_set_view)(fh, 0, MPI_BYTE, filetype, "native",
                                MPI_INFO_NULL);
    MPI_Type_free(&filetype);

    /* write to variable collectively */
    TRACE_IO(MPI_File_write_at_all)(fh, 0, buf, buf_len, MPI_BYTE, &mpistatus);

    NCI_Free(buf);

    TRACE_IO(MPI_File_set_view)(fh, 0, MPI_BYTE, MPI_BYTE, "native",
                                MPI_INFO_NULL);

    if (status != NC_NOERR) return status;

    if (mpireturn != MPI_SUCCESS)
        return ncmpii_handle_error(mpireturn, "MPI_File_write_at_all");

    return NC_NOERR;
}

/*----< ncmpii_fill_vars() >-------------------------------------------------*/
int
ncmpii_fill_vars(NC *ncp)
{
    int status=NC_NOERR;

#ifdef FILL_ONE_VAR_AT_A_TIME
    int err;

    /* fill variables according to their fill mode settings */
    if (NC_IsNew(ncp))
        /* file is just created */
        status = fillerup(ncp);
    else if (ncp->vars.ndefined > ncp->old->vars.ndefined) {
        /* old file, but new variables have been added */
        status = fill_added(ncp, ncp->old);

        err = fill_added_recs(ncp, ncp->old);
        if (status == NC_NOERR) status = err;
    }
#else
    /* fill variables according to their fill mode settings */
    if (NC_IsNew(ncp))
        /* file is just created and never exits define mode */
        status = fillerup_aggregate(ncp, NULL);
    else
        /* old file, but new variables have been added */
        status = fillerup_aggregate(ncp, ncp->old);
#endif
    return status;
}


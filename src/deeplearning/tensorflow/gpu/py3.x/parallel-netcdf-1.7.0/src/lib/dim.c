/*
 *  Copyright (C) 2003, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id: dim.c 2302 2016-01-10 20:33:45Z wkliao $ */

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
#include "ncx.h"
#include "fbits.h"
#include "macro.h"
#include "utf8proc.h"

/*
 * Free dim
 * Formerly
NC_free_dim(dim)
 */
inline void
ncmpii_free_NC_dim(NC_dim *dimp)
{
    if (dimp == NULL) return;
    ncmpii_free_NC_string(dimp->name);
    NCI_Free(dimp);
}


/* allocate and return a new NC_dim object */
inline NC_dim *
ncmpii_new_x_NC_dim(NC_string *name)
{
    NC_dim *dimp;

    dimp = (NC_dim *) NCI_Malloc(sizeof(NC_dim));
    if (dimp == NULL) return NULL;

    dimp->name = name;
    dimp->size = 0;

    return(dimp);
}

/*----< ncmpii_new_NC_dim() >------------------------------------------------*/
/*
 * Formerly, NC_new_dim(const char *name, long size)
 */
static NC_dim *
ncmpii_new_NC_dim(const char *uname,  /* dimension name */
                  MPI_Offset  size)
{
    NC_string *strp;
    NC_dim *dimp;

    char *name = (char *)utf8proc_NFC((const unsigned char *)uname);
    if (name == NULL) return NULL;

    strp = ncmpii_new_NC_string(strlen(name), name);
    free(name);
    if (strp == NULL) return NULL;

    dimp = ncmpii_new_x_NC_dim(strp);
    if (dimp == NULL) {
        ncmpii_free_NC_string(strp);
        return NULL;
    }

    dimp->size = size;

    return(dimp);
}


NC_dim *
dup_NC_dim(const NC_dim *dimp)
{
    return ncmpii_new_NC_dim(dimp->name->cp, dimp->size);
}

/*----< ncmpii_find_NC_Udim() >----------------------------------------------*/
/*
 * Step thru NC_DIMENSION array, seeking the UNLIMITED dimension.
 * Return dimid or -1 on not found.
 * *dimpp is set to the appropriate NC_dim.
 */
int
ncmpii_find_NC_Udim(const NC_dimarray  *ncap,
                    NC_dim            **dimpp)
{
    int dimid;

    assert(ncap != NULL);

    if (ncap->ndefined == 0) return -1;

    /* note that the number of dimensions allowed is < 2^32 */
    for (dimid=0; dimid<ncap->ndefined; dimid++)
        if (ncap->value[dimid]->size == NC_UNLIMITED) {
            /* found the matched name */
            if (dimpp != NULL)
                *dimpp = ncap->value[dimid];
            return dimid;
        }

    /* not found */
    return -1;
}

/*----< NC_finddim() >-------------------------------------------------------*/
/*
 * Step thru NC_DIMENSION array, seeking match on name.
 * Return dimid or -1 on not found.
 * *dimpp is set to the appropriate NC_dim.
 */
static int
NC_finddim(const NC_dimarray  *ncap,
           const char         *uname,
           NC_dim            **dimpp)
{
    int dimid;
    size_t nchars;

    assert(ncap != NULL);

    if (ncap->ndefined == 0) return -1;

    char *name = (char *)utf8proc_NFC((const unsigned char *)uname);
    nchars = strlen(name);

    /* note that the number of dimensions allowed is < 2^32 */
    for (dimid=0; dimid<ncap->ndefined; dimid++) {
        if (ncap->value[dimid]->name->nchars == nchars &&
            strncmp(ncap->value[dimid]->name->cp, name, nchars) == 0) {
            /* found the matched name */
            if (dimpp != NULL)
                *dimpp = ncap->value[dimid];
            free(name);
            return dimid;
        }
    }
    free(name);

    /* the name is not found */
    return -1;
}


/* dimarray */


/*----< ncmpii_free_NC_dimarray() >------------------------------------------*/
/*
 * Free NC_dimarray values.
 * formerly
NC_free_array()
 */
inline void
ncmpii_free_NC_dimarray(NC_dimarray *ncap)
{
    int i;

    assert(ncap != NULL);
    if (ncap->nalloc == 0) return;

    assert(ncap->value != NULL);
    for (i=0; i<ncap->ndefined; i++)
        ncmpii_free_NC_dim(ncap->value[i]);

    NCI_Free(ncap->value);
    ncap->value    = NULL;
    ncap->nalloc   = 0;
    ncap->ndefined = 0;
}


/*----< ncmpii_dup_NC_dimarray() >-------------------------------------------*/
int
ncmpii_dup_NC_dimarray(NC_dimarray *ncap, const NC_dimarray *ref)
{
    int i, status=NC_NOERR;

    assert(ref != NULL);
    assert(ncap != NULL);

    if (ref->nalloc == 0) {
        ncap->nalloc   = 0;
        ncap->ndefined = 0;
        ncap->value    = NULL;
        return NC_NOERR;
    }

    if (ref->nalloc > 0) {
        ncap->value = (NC_dim **) NCI_Calloc((size_t)ref->nalloc, sizeof(NC_dim *));
        if (ncap->value == NULL) DEBUG_RETURN_ERROR(NC_ENOMEM)
        ncap->nalloc = ref->nalloc;
    }

    ncap->ndefined = 0;
    for (i=0; i<ref->ndefined; i++) {
        ncap->value[i] = dup_NC_dim(ref->value[i]);
        if (ncap->value[i] == NULL) {
            DEBUG_ASSIGN_ERROR(status, NC_ENOMEM)
            break;
        }
    }

    if (status != NC_NOERR) {
        ncmpii_free_NC_dimarray(ncap);
        return status;
    }

    ncap->ndefined = ref->ndefined;

    return NC_NOERR;
}


/*----< incr_NC_dimarray() >---------------------------------------------- --*/
/*
 * Add a new handle to the end of an array of handles
 * Formerly, NC_incr_array(array, tail)
 */
int
incr_NC_dimarray(NC_dimarray *ncap,
                 NC_dim      *newdimp)
{
    NC_dim **vp;

    assert(ncap != NULL);

    if (ncap->nalloc == 0) {
        assert(ncap->ndefined == 0);
        vp = (NC_dim **) NCI_Malloc(NC_ARRAY_GROWBY * sizeof(NC_dim *));
        if (vp == NULL) DEBUG_RETURN_ERROR(NC_ENOMEM)

        ncap->value = vp;
        ncap->nalloc = NC_ARRAY_GROWBY;
    }
    else if (ncap->ndefined + 1 > ncap->nalloc) {
        vp = (NC_dim **) NCI_Realloc(ncap->value,
             (size_t)(ncap->nalloc + NC_ARRAY_GROWBY) * sizeof(NC_dim *));
        if (vp == NULL) DEBUG_RETURN_ERROR(NC_ENOMEM)

        ncap->value = vp;
        ncap->nalloc += NC_ARRAY_GROWBY;
    }
    /* else here means some space still available */

    if (newdimp != NULL) {
        ncap->value[ncap->ndefined] = newdimp;
        ncap->ndefined++;
    }

    return NC_NOERR;
}


/*----< ncmpii_elem_NC_dimarray() >------------------------------------------*/
inline NC_dim *
ncmpii_elem_NC_dimarray(const NC_dimarray *ncap,
                        int                dimid)
{
    /* returns the dimension ID defined earlier */
    assert(ncap != NULL);

    if (dimid < 0 || ncap->ndefined == 0 || dimid >= ncap->ndefined)
        return NULL;

    assert(ncap->value != NULL);

    return ncap->value[dimid];
}


/* Public */

/*----< ncmpi_def_dim() >----------------------------------------------------*/
int
ncmpi_def_dim(int         ncid,    /* IN:  file ID */
              const char *name,    /* IN:  name of dimension */
              MPI_Offset  size,    /* IN:  dimension size */
              int        *dimidp)  /* OUT: dimension ID */
{
    int dimid, file_ver, status;
    NC *ncp;
    NC_dim *dimp;

    /* check if ncid is valid */
    status = ncmpii_NC_check_id(ncid, &ncp);
    if (status != NC_NOERR) return status;

    /* check if called in define mode */
    if (!NC_indef(ncp)) DEBUG_RETURN_ERROR(NC_ENOTINDEFINE)

    /* check if the name string is legal for netcdf format */
    file_ver = 1;
    if (fIsSet(ncp->flags, NC_64BIT_OFFSET))
        file_ver = 2;
    else if (fIsSet(ncp->flags, NC_64BIT_DATA))
        file_ver = 5;

    status = ncmpii_NC_check_name(name, file_ver);
    if (status != NC_NOERR) return status;

    /* MPI_Offset is usually a signed value, but serial netcdf uses
     * size_t -- normally unsigned
     * In 1999 ISO C standard, size_t is a unsigned integer type of at least
     * 16 bit. */
    if ((ncp->flags & NC_64BIT_OFFSET) && SIZEOF_OFF_T > 4) {
        /* CDF2 format and LFS, max is 2^32-4 */
        if (size > X_UINT_MAX - 3 || (size < 0))
            /* "-3" handles rounded-up size */
            DEBUG_RETURN_ERROR(NC_EDIMSIZE)
    } else if ((ncp->flags & NC_64BIT_DATA)) {
        /* CDF5 format*/
        if (size < 0)
            DEBUG_RETURN_ERROR(NC_EDIMSIZE)
    } else {
        /* CDF1 format, max is 2^31-4 */
        if (size > X_INT_MAX - 3 || (size < 0))
            /* "-3" handles rounded-up size */
            DEBUG_RETURN_ERROR(NC_EDIMSIZE)
    }

    if (size == NC_UNLIMITED) {
        /* check for any existing unlimited dimension, netcdf allows
         * one per file
         */
        dimid = ncmpii_find_NC_Udim(&ncp->dims, &dimp);
        if (dimid != -1) DEBUG_RETURN_ERROR(NC_EUNLIMIT) /* found an existing one */
    }

    /* check if exceeds the upperbound has reached */
    if (ncp->dims.ndefined >= NC_MAX_DIMS) DEBUG_RETURN_ERROR(NC_EMAXDIMS)

    /* check if the name string is previously used */
    dimid = NC_finddim(&ncp->dims, name, &dimp);
    if (dimid != -1) DEBUG_RETURN_ERROR(NC_ENAMEINUSE)

    /* create a new dimension object */
    dimp = ncmpii_new_NC_dim(name, size);
    if (dimp == NULL) DEBUG_RETURN_ERROR(NC_ENOMEM)

    /* Add a new handle to the end of an array of handles */
    status = incr_NC_dimarray(&ncp->dims, dimp);
    if (status != NC_NOERR) {
        ncmpii_free_NC_dim(dimp);
        return status;
    }

    if (dimidp != NULL)
        *dimidp = (int)ncp->dims.ndefined -1;
        /* ncp->dims.ndefined has been increased in incr_NC_dimarray() */

    return NC_NOERR;
}


/*----< ncmpi_inq_dimid() >--------------------------------------------------*/
int
ncmpi_inq_dimid(int         ncid,
                const char *name,
                int        *dimid_ptr)
{
    int status;
    NC *ncp;
    int dimid;

    status = ncmpii_NC_check_id(ncid, &ncp);
    if (status != NC_NOERR) return status;

    dimid = NC_finddim(&ncp->dims, name, NULL);
    if (dimid == -1) DEBUG_RETURN_ERROR(NC_EBADDIM)

    *dimid_ptr = dimid;
    return NC_NOERR;
}


/*----< ncmpi_inq_dim() >----------------------------------------------------*/
int
ncmpi_inq_dim(int         ncid,
              int         dimid,
              char       *name,
              MPI_Offset *sizep)
{
    int status;
    NC *ncp;
    NC_dim *dimp;

    status = ncmpii_NC_check_id(ncid, &ncp);
    if (status != NC_NOERR) return status;

    dimp = ncmpii_elem_NC_dimarray(&ncp->dims, dimid);
    if (dimp == NULL) DEBUG_RETURN_ERROR(NC_EBADDIM)

    if (name != NULL)
        /* in PnetCDF, name->cp is always NULL character terminated */
        strcpy(name, dimp->name->cp);

    if (sizep != NULL) {
        if (dimp->size == NC_UNLIMITED)
            *sizep = NC_get_numrecs(ncp);
        else
            *sizep = dimp->size;
    }
    return NC_NOERR;
}


/*----< ncmpi_inq_dimname() >------------------------------------------------*/
int
ncmpi_inq_dimname(int   ncid,
                  int   dimid,
                  char *name)
{
    return ncmpi_inq_dim(ncid, dimid, name, NULL);
}


/*----< ncmpi_inq_dimlen() >-------------------------------------------------*/
int
ncmpi_inq_dimlen(int         ncid,
                 int         dimid,
                 MPI_Offset *lenp)
{
    return ncmpi_inq_dim(ncid, dimid, NULL, lenp);
}


/*----< ncmpi_rename_dim() >--------------------------------------------------*/
/* This API is collective if called in data mode */
int
ncmpi_rename_dim(int         ncid,
                 int         dimid,
                 const char *newname)
{
    int file_ver, status, existid, err, mpireturn;
    NC *ncp;
    NC_dim *dimp;

    status = ncmpii_NC_check_id(ncid, &ncp);
    if (status != NC_NOERR) return status;

    if (NC_readonly(ncp)) DEBUG_RETURN_ERROR(NC_EPERM)

    file_ver = 1;
    if (fIsSet(ncp->flags, NC_64BIT_OFFSET))
        file_ver = 2;
    else if (fIsSet(ncp->flags, NC_64BIT_DATA))
        file_ver = 5;

    status = ncmpii_NC_check_name(newname, file_ver);
    if (status != NC_NOERR) return status;

    existid = NC_finddim(&ncp->dims, newname, &dimp);
    if (existid != -1) DEBUG_RETURN_ERROR(NC_ENAMEINUSE)

    dimp = ncmpii_elem_NC_dimarray(&ncp->dims, dimid);
    if (dimp == NULL) DEBUG_RETURN_ERROR(NC_EBADDIM)

    if (NC_indef(ncp)) {
        NC_string *newStr = ncmpii_new_NC_string(strlen(newname), newname);
        if (newStr == NULL) DEBUG_RETURN_ERROR(NC_ENOMEM)

        ncmpii_free_NC_string(dimp->name);
        dimp->name = newStr;
        return NC_NOERR;
    }
    /* else, not in define mode.
     * if called in data mode (collective or independent), this function must
     * be called collectively, i.e. all processes must participate.
     */

    if (ncp->safe_mode) {
        int nchars = (int)strlen(newname);
        TRACE_COMM(MPI_Bcast)(&nchars, 1, MPI_INT, 0, ncp->nciop->comm);
        if (mpireturn != MPI_SUCCESS)
            return ncmpii_handle_error(mpireturn, "MPI_Bcast"); 

        if (nchars != strlen(newname)) {
            /* newname's length is inconsistent with root's */
            printf("Warning: dimension name(%s) used in %s() is inconsistent\n",
                   newname, __func__);
            if (status == NC_NOERR) DEBUG_ASSIGN_ERROR(status, NC_EMULTIDEFINE_DIM_NAME)
        }
    }

    /* ncmpii_set_NC_string() will check for strlen(newname) > nchars error */
    err = ncmpii_set_NC_string(dimp->name, newname);
    if (status == NC_NOERR) status = err;

    /* PnetCDF expects all processes use the same name, However, when names
     * are not the same, only root's value is significant. Broadcast the
     * new name at root to overwrite new names at other processes.
     * (This API is collective if called in data mode)
     */
    TRACE_COMM(MPI_Bcast)(dimp->name->cp, (int)dimp->name->nchars, MPI_CHAR, 0,
                          ncp->nciop->comm);
    if (mpireturn != MPI_SUCCESS) {
        err = ncmpii_handle_error(mpireturn, "MPI_Bcast"); 
        if (status == NC_NOERR) status = err;
    }

    /* Let root write the entire header to the file. Note that we cannot just
     * update the variable name in its space occupied in the file header,
     * because if the file space occupied by the name shrinks, all the metadata
     * following it must be moved ahead.
     */
    err = ncmpii_write_header(ncp);
    if (status == NC_NOERR) status = err;

    return status;
}

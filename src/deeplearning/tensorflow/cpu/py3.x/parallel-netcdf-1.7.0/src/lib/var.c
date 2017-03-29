/*
 *  Copyright (C) 2003, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id: var.c 2322 2016-02-26 22:47:06Z wkliao $ */

#if HAVE_CONFIG_H
# include "ncconfig.h"
#endif

#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#include <stdio.h>
#include <string.h> /* memset() */
#include <assert.h>

#include <mpi.h>

#include "nc.h"
#include "ncx.h"
#include "rnd.h"
#include "macro.h"
#include "utf8proc.h"

/*----< ncmpii_free_NC_var() >------------------------------------------------*/
/*
 * Free var
 * Formerly
NC_free_var(var)
 */
inline void
ncmpii_free_NC_var(NC_var *varp)
{
    if (varp == NULL) return;
    ncmpii_free_NC_attrarray(&varp->attrs);
    ncmpii_free_NC_string(varp->name);
#ifdef ENABLE_SUBFILING
    if (varp->num_subfiles > 1) /* deallocate it */
        NCI_Free(varp->dimids_org);
#endif
    NCI_Free(varp);
}


/*----< ncmpii_new_x_NC_var() >-----------------------------------------------*/
/*
 * Common code for ncmpii_new_NC_var()
 * and ncx_get_NC_var()
 */
NC_var *
ncmpii_new_x_NC_var(NC_string *strp,
                    int        ndims)
{
    NC_var *varp;
    int shape_space   = M_RNDUP(ndims * SIZEOF_MPI_OFFSET);
    int dsizes_space  = M_RNDUP(ndims * SIZEOF_MPI_OFFSET);
    int dimids_space  = M_RNDUP(ndims * SIZEOF_INT);
    size_t sizeof_NC_var = M_RNDUP(sizeof(NC_var));
    size_t sz = sizeof_NC_var + (size_t)(shape_space + dsizes_space + dimids_space);

    /* this function allocates a contiguous memory space to put all
     * members of NC_var structure together:
     * dimid_space is for dimids[],
     * shape_space is for shape[],
     * dsizes_space is for dsizes[]
     * (I don't know why M_RNDUP is needed here and why they should be kept
     * in a contiguous memory space.)
     */
    varp = (NC_var *) NCI_Malloc(sz);
    if (varp == NULL ) return NULL;

    memset(varp, 0, sz);

    varp->name = strp;
    varp->ndims = ndims;

    if (ndims != 0) {
        /*
         * NOTE: lint may complain about the next 3 lines:
         * "pointer cast may result in improper alignment".
         * We use the M_RNDUP() macro to get the proper alignment.
         * roundup to a double
         */
        varp->shape  = (MPI_Offset *)((char *)varp + sizeof_NC_var);
        varp->dsizes = (MPI_Offset *)((char *)varp->shape  + shape_space);
        varp->dimids = (int *)       ((char *)varp->dsizes + dsizes_space);
    }

    varp->xsz = 0;
    varp->len = 0;
    varp->begin = 0;

    return varp;
}


/*----< ncmpii_new_NC_var() >------------------------------------------------*/
/*
 * Formerly, NC_new_var()
 */
static NC_var *
ncmpii_new_NC_var(const char *uname,  /* variable name (NULL terminated) */
                  nc_type     type,
                  int         ndims,
                  const int  *dimids)
{
    NC_string *strp;
    NC_var *varp;

    char *name = (char *)utf8proc_NFC((const unsigned char *)uname);
    if (name == NULL) return NULL;

    strp = ncmpii_new_NC_string(strlen(name), name);
    free(name);
    if (strp == NULL) return NULL;

    varp = ncmpii_new_x_NC_var(strp, ndims);
    if (varp == NULL ) {
        ncmpii_free_NC_string(strp);
        return NULL;
    }

    varp->type = type;

    if (ndims != 0 && dimids != NULL)
        memcpy(varp->dimids, dimids, (size_t)ndims * SIZEOF_INT);

    return(varp);
}

/*----< dup_NC_var() >--------------------------------------------------------*/
NC_var *
dup_NC_var(const NC_var *rvarp)
{
    NC_var *varp;

    varp = ncmpii_new_NC_var(rvarp->name->cp, rvarp->type, rvarp->ndims,
                             rvarp->dimids);
    if (varp == NULL) return NULL;

    if (ncmpii_dup_NC_attrarray(&varp->attrs, &rvarp->attrs) != NC_NOERR) {
        ncmpii_free_NC_var(varp);
        return NULL;
    }

    /* copy the contents of shape may not be necessary, as one must call
     * ncmpii_NC_computeshapes() to recompute it after a new variable is
     * created */
    memcpy(varp->shape,  rvarp->shape,  (size_t)rvarp->ndims * SIZEOF_MPI_OFFSET);
    memcpy(varp->dsizes, rvarp->dsizes, (size_t)rvarp->ndims * SIZEOF_MPI_OFFSET);
    varp->xsz = rvarp->xsz;
    varp->len = rvarp->len;
    varp->begin = rvarp->begin;

    return varp;
}

/* vararray */

/*----< ncmpii_free_NC_vararray() >-------------------------------------------*/
/*
 * Free NC_vararray values.
 * formerly
NC_free_array()
 */
inline void
ncmpii_free_NC_vararray(NC_vararray *ncap)
{
    int i;

    assert(ncap != NULL);
    if (ncap->nalloc == 0) return;

    assert(ncap->value != NULL);
    for (i=0; i<ncap->ndefined; i++) {
        if (ncap->value[i] != NULL)
            ncmpii_free_NC_var(ncap->value[i]);
    }

    NCI_Free(ncap->value);
    ncap->value    = NULL;
    ncap->nalloc   = 0;
    ncap->ndefined = 0;
}


/*----< ncmpii_dup_NC_vararray() >--------------------------------------------*/
int
ncmpii_dup_NC_vararray(NC_vararray       *ncap,
                       const NC_vararray *ref)
{
    int i, status=NC_NOERR;

    assert(ref != NULL);
    assert(ncap != NULL);

    if (ref->nalloc == 0) {
        ncap->nalloc = 0;
        ncap->ndefined = 0;
        ncap->value = NULL;
        return NC_NOERR;
    }

    if (ref->nalloc > 0) {
        ncap->value = (NC_var **) NCI_Calloc((size_t)ref->nalloc, sizeof(NC_var*));
        if (ncap->value == NULL) DEBUG_RETURN_ERROR(NC_ENOMEM)
        ncap->nalloc = ref->nalloc;
    }

    ncap->ndefined = 0;
    for (i=0; i<ref->ndefined; i++) {
        ncap->value[i] = dup_NC_var(ref->value[i]);
        if (ncap->value[i] == NULL) {
            DEBUG_ASSIGN_ERROR(status, NC_ENOMEM)
            break;
        }
    }

    if (status != NC_NOERR) {
        ncmpii_free_NC_vararray(ncap);
        return status;
    }

    ncap->ndefined = ref->ndefined;

    return NC_NOERR;
}


/*
 * Add a new handle on the end of an array of handles
 * Formerly
NC_incr_array(array, tail)
 */
int
incr_NC_vararray(NC_vararray *ncap,
                 NC_var      *newvarp)
{
    NC_var **vp;

    assert(ncap != NULL);
    assert(newvarp != NULL);

    if (ncap->nalloc == 0) { /* no variable has been allocated yet */
        assert(ncap->ndefined == 0);
        vp = (NC_var **) NCI_Malloc(NC_ARRAY_GROWBY * sizeof(NC_var *));
        if (vp == NULL) DEBUG_RETURN_ERROR(NC_ENOMEM)

        ncap->value = vp;
        ncap->nalloc = NC_ARRAY_GROWBY;
    }
    else if (ncap->ndefined + 1 > ncap->nalloc) {
        vp = (NC_var **) NCI_Realloc(ncap->value,
                         (size_t)(ncap->nalloc + NC_ARRAY_GROWBY) * sizeof(NC_var *));
        if (vp == NULL) DEBUG_RETURN_ERROR(NC_ENOMEM)

        ncap->value = vp;
        ncap->nalloc += NC_ARRAY_GROWBY;
    }

    if (newvarp != NULL) {
        ncap->value[ncap->ndefined] = newvarp;
        ncap->ndefined++;
    }

    return NC_NOERR;
}


inline static NC_var *
elem_NC_vararray(const NC_vararray *ncap,
                 int                varid)
{
    assert(ncap != NULL);
    /* cast needed for braindead systems with signed MPI_Offset */
    if ((varid < 0) ||  ncap->ndefined == 0 || varid >= ncap->ndefined)
        return NULL;

    assert(ncap->value != NULL);

    return ncap->value[varid];
}


/* End vararray per se */


/*
 * Step thru NC_VARIABLE array, seeking match on name.
 * Return varid or -1 on not found.
 * *varpp is set to the appropriate NC_var.
 * Formerly (sort of)
NC_hvarid
 */
static int
ncmpii_NC_findvar(const NC_vararray  *ncap,
                  const char         *uname,
                  NC_var            **varpp)
{
    int varid;
    size_t nchars;
    char *name;
    NC_var **loc;

    assert (ncap != NULL);

    if (ncap->ndefined == 0) return -1;

    loc = (NC_var **) ncap->value;

    /* normalized version of uname */
    name = (char *)utf8proc_NFC((const unsigned char *)uname);
    if (name == NULL) DEBUG_RETURN_ERROR(NC_ENOMEM)
    nchars = strlen(name);

    for (varid=0; varid<ncap->ndefined; varid++, loc++) {
        if ((*loc)->name->nchars == nchars &&
            strncmp((*loc)->name->cp, name, nchars) == 0) {
            if (varpp != NULL)
                *varpp = *loc;
            free(name);
            return (varid); /* Normal return */
        }
    }
    free(name);
    return (-1); /* not found */
}

/*
 * For a netcdf type
 *  return the size of one element in the external representation.
 * Note that arrays get rounded up to X_ALIGN boundaries.
 * Formerly
NC_xtypelen
 * See also ncx_len()
 */
static int
ncx_szof(nc_type type)
{
    switch(type){
        case NC_BYTE:
        case NC_CHAR:   return (1);
        case NC_SHORT:  return (2);
        case NC_INT:    return X_SIZEOF_INT;
        case NC_FLOAT:  return X_SIZEOF_FLOAT;
        case NC_DOUBLE: return X_SIZEOF_DOUBLE;
        case NC_UBYTE:  return (1);
        case NC_USHORT: return (2);
        case NC_UINT:   return X_SIZEOF_UINT;
        case NC_INT64:  return X_SIZEOF_INT64;
        case NC_UINT64: return X_SIZEOF_UINT64;
        default:
             fprintf(stderr,"ncx_szof invalid type %d\n", type);
             assert(0);
    }
    /* default */
    return 0;
}

/*----< ncmpii_NC_var_shape64() >--------------------------------------------*/
/*
 * set varp->xsz, varp->shape and varp->len of a variable
 */
int
ncmpii_NC_var_shape64(NC                *ncp,
                      NC_var            *varp,
                      const NC_dimarray *dims)
{
    int i;
    MPI_Offset product = 1;

    /* set the size of 1 element */
    varp->xsz = ncx_szof(varp->type);

    if (varp->ndims == 0) goto out;

    /*
     * use the user supplied dimension indices to determine the shape
     */
    for (i=0; i<varp->ndims; i++) {
        const NC_dim *dimp;

        if (varp->dimids[i] < 0)
            DEBUG_RETURN_ERROR(NC_EBADDIM)

        if (varp->dimids[i] >= ((dims != NULL) ? dims->ndefined : 1))
            DEBUG_RETURN_ERROR(NC_EBADDIM)

        /* get the pointer to the dim object */
        dimp = ncmpii_elem_NC_dimarray(dims, varp->dimids[i]);
        varp->shape[i] = dimp->size;

        /* check for record variable, only the highest dimension can
         * be unlimited */
        if (varp->shape[i] == NC_UNLIMITED && i != 0)
            DEBUG_RETURN_ERROR(NC_EUNLIMPOS)
    }

    /*
     * compute the dsizes, the right to left product of shape
     */
    product = 1;
    if (varp->ndims == 1) {
        if (varp->shape[0] == NC_UNLIMITED)
            varp->dsizes[0] = 1;
        else {
            varp->dsizes[0] = varp->shape[0];
            product = varp->shape[0];
        }
    }
    else { /* varp->ndims > 1 */
        varp->dsizes[varp->ndims-1] = varp->shape[varp->ndims-1];
        product = varp->shape[varp->ndims-1];
        for (i=varp->ndims-2; i>=0; i--) {
            if (varp->shape[i] != NC_UNLIMITED)
                product *= varp->shape[i];
            varp->dsizes[i] = product;
        }
    }

out :
    /*
     * For CDF-1 and CDF-2 formats, the total number of array elements
     * cannot exceed 2^32, unless this variable is the last fixed-size
     * variable, there is no record variable, and the file starting
     * offset of this variable is less than 2GiB.
     * We will check this in ncmpi_enddef() which calls ncmpii_NC_enddef()
     * which calls ncmpii_NC_check_vlens()
     *
    if (! fIsSet(ncp->flags, NC_64BIT_DATA) && product >= X_UINT_MAX)
        DEBUG_RETURN_ERROR(NC_EVARSIZE)
     */

    /*
     * align variable size to 4 byte boundary, required by all netcdf
     * file formats
     */
    varp->len = product * varp->xsz;
    if (varp->len % 4 > 0)
        varp->len += 4 - varp->len % 4; /* round up */

    return NC_NOERR;
}

/*
 * Check whether variable size is less than or equal to vlen_max,
 * without overflowing in arithmetic calculations.  If OK, return 1,
 * else, return 0.  For CDF1 format or for CDF2 format on non-LFS
 * platforms, vlen_max should be 2^31 - 4, but for CDF2 format on
 * systems with LFS it should be 2^32 - 4.
 */
inline int
ncmpii_NC_check_vlen(NC_var     *varp,
                     MPI_Offset  vlen_max)
{
    int ii;
    MPI_Offset prod=varp->xsz;     /* product of xsz and dimensions so far */

    for (ii = IS_RECVAR(varp) ? 1 : 0; ii < varp->ndims; ii++) {
        if (varp->shape[ii] > vlen_max / prod) {
            return 0;           /* size in bytes won't fit in a 32-bit int */
        }
        prod *= varp->shape[ii];
    }
    return 1;                  /* OK */
}

/*----< ncmpii_NC_lookupvar() >----------------------------------------------*/
/*
 * Given valid ncp and varid, return var
 *  else NULL on error
 * Formerly
NC_hlookupvar()
 */
int
ncmpii_NC_lookupvar(NC      *ncp,
                    int      varid,
                    NC_var **varp)
{
    if (varid == NC_GLOBAL) /* Global is error in this context */
        DEBUG_RETURN_ERROR(NC_EGLOBAL)

    *varp = elem_NC_vararray(&ncp->vars, varid);
    if (*varp == NULL) /* could not find variable with varid */
        DEBUG_RETURN_ERROR(NC_ENOTVAR)

    return NC_NOERR;
}


/* Public */

/*----< ncmpi_def_var() >----------------------------------------------------*/
int
ncmpi_def_var(int         ncid,
              const char *name,
              nc_type     type,
              int         ndims,
              const int  *dimids,
              int        *varidp)
{
    int varid, file_ver, status;
    NC *ncp;
    NC_var *varp;

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

    /* check if type is a valid netcdf type */
    status = ncmpii_cktype(file_ver, type);
    if (status != NC_NOERR) return status;

    /* TODO: make ndims of type MPI_Offset so ndims can be > 2^31-1 in CDF-5
    if ((ndims < 0) || ndims > X_INT_MAX) DEBUG_RETURN_ERROR(NC_EINVAL)
    */
    if (ndims < 0) DEBUG_RETURN_ERROR(NC_EINVAL)

    /* there is an upperbound for the number of variables defined in a file */
    if (ncp->vars.ndefined >= NC_MAX_VARS) DEBUG_RETURN_ERROR(NC_EMAXVARS)

    /* check whether the variable name has been used */
    varid = ncmpii_NC_findvar(&ncp->vars, name, &varp);
    if (varid != -1) DEBUG_RETURN_ERROR(NC_ENAMEINUSE)

    /* create a new variable */
    varp = ncmpii_new_NC_var(name, type, ndims, dimids);
    if (varp == NULL) DEBUG_RETURN_ERROR(NC_ENOMEM)

    /* set up array dimensional structures */
    status = ncmpii_NC_var_shape64(ncp, varp, &ncp->dims);
    if (status != NC_NOERR) {
        ncmpii_free_NC_var(varp);
        return status;
    }

    /* Add a new handle to the end of an array of handles */
    status = incr_NC_vararray(&ncp->vars, varp);
    if (status != NC_NOERR) {
        ncmpii_free_NC_var(varp);
        return status;
    }

    if (varidp != NULL)
        *varidp = (int)ncp->vars.ndefined - 1; /* varid */
        /* ncp->vars.ndefined has been increased in incr_NC_vararray() */

    /* default is NOFILL */
    varp->no_fill = 1;

    /* change to FILL only if the entire dataset fill mode is FILL */
    if (NC_dofill(ncp)) varp->no_fill = 0;

    return NC_NOERR;
}


/*----< ncmpi_inq_varid() >--------------------------------------------------*/
int
ncmpi_inq_varid(int         ncid,
                const char *name,
                int        *varid_ptr)
{
    int status;
    NC *ncp;
    NC_var *varp;
    int varid;

    status = ncmpii_NC_check_id(ncid, &ncp);
    if (status != NC_NOERR) return status;

    varid = ncmpii_NC_findvar(&ncp->vars, name, &varp);
    if (varid == -1) DEBUG_RETURN_ERROR(NC_ENOTVAR)

    *varid_ptr = varid;
    return NC_NOERR;
}

/*----< ncmpi_inq_var() >----------------------------------------------------*/
int
ncmpi_inq_var(int      ncid,
              int      varid,
              char    *name,
              nc_type *typep,
              int     *ndimsp,
              int     *dimids,
              int     *nattsp)
{
    int status;
    NC *ncp;
    NC_var *varp;

    status = ncmpii_NC_check_id(ncid, &ncp);
    if (status != NC_NOERR) return status;

    /* using NC_GLOBAL in varid is illegal for this API. See
     * http://www.unidata.ucar.edu/mailing_lists/archives/netcdfgroup/2015/msg00196.html
     */
    if (varid == NC_GLOBAL) DEBUG_RETURN_ERROR(NC_EGLOBAL)

    varp = elem_NC_vararray(&ncp->vars, varid);
    if (varp == NULL) DEBUG_RETURN_ERROR(NC_ENOTVAR)

    if (name != NULL)
        /* in PnetCDF, name->cp is always NULL character terminated */
        strcpy(name, varp->name->cp);

    if (typep != 0)
        *typep = varp->type;

    if (ndimsp != 0) {
#ifdef ENABLE_SUBFILING
        /* varp->num_subfiles is already set during open or enddef */
        if (varp->num_subfiles > 1)
            *ndimsp = varp->ndims_org;
        else
#endif
            *ndimsp = varp->ndims;
    }
    if (dimids != 0) {
#ifdef ENABLE_SUBFILING
        /* varp->dimids_org is already set during open or enddef */
        if (varp->num_subfiles > 1)
            memcpy(dimids, varp->dimids_org, (size_t)varp->ndims_org * SIZEOF_INT);
        else
#endif
            memcpy(dimids, varp->dimids, (size_t)varp->ndims * SIZEOF_INT);
    }
    if (nattsp != 0)
        *nattsp = (int) varp->attrs.ndefined;

    return NC_NOERR;
}


/*----< ncmpi_inq_varname() >------------------------------------------------*/
int
ncmpi_inq_varname(int   ncid,
                  int   varid,
                  char *name)
{
    int status;
    NC *ncp;
    NC_var *varp;

    status = ncmpii_NC_check_id(ncid, &ncp);
    if (status != NC_NOERR) return status;

    /* using NC_GLOBAL in varid is illegal for this API. See
     * http://www.unidata.ucar.edu/mailing_lists/archives/netcdfgroup/2015/msg00196.html
     */
    if (varid == NC_GLOBAL) DEBUG_RETURN_ERROR(NC_EGLOBAL)

    varp = elem_NC_vararray(&ncp->vars, varid);
    if (varp == NULL) DEBUG_RETURN_ERROR(NC_ENOTVAR)

    if (name != NULL)
        /* in PnetCDF, name->cp is always NULL character terminated */
        strcpy(name, varp->name->cp);

    return NC_NOERR;
}

/*----< ncmpi_inq_vartype() >------------------------------------------------*/
int
ncmpi_inq_vartype(int      ncid,
                  int      varid,
                  nc_type *typep)
{
    int status;
    NC *ncp;
    NC_var *varp;

    status = ncmpii_NC_check_id(ncid, &ncp);
    if (status != NC_NOERR) return status;

    /* using NC_GLOBAL in varid is illegal for this API. See
     * http://www.unidata.ucar.edu/mailing_lists/archives/netcdfgroup/2015/msg00196.html
     */
    if (varid == NC_GLOBAL) DEBUG_RETURN_ERROR(NC_EGLOBAL)

    varp = elem_NC_vararray(&ncp->vars, varid);
    if (varp == NULL) DEBUG_RETURN_ERROR(NC_ENOTVAR)

    if (typep != NULL) *typep = varp->type;

    return NC_NOERR;
}

/*----< ncmpi_inq_varndims() >-----------------------------------------------*/
int
ncmpi_inq_varndims(int ncid, int varid, int *ndimsp)
{
    int status;
    NC *ncp;
    NC_var *varp;

    status = ncmpii_NC_check_id(ncid, &ncp);
    if (status != NC_NOERR) return status;

    /* using NC_GLOBAL in varid is illegal for this API. See
     * http://www.unidata.ucar.edu/mailing_lists/archives/netcdfgroup/2015/msg00196.html
     */
    if (varid == NC_GLOBAL) DEBUG_RETURN_ERROR(NC_EGLOBAL)

    varp = elem_NC_vararray(&ncp->vars, varid);
    if (varp == NULL) DEBUG_RETURN_ERROR(NC_ENOTVAR)

    if (ndimsp != 0) {
#ifdef ENABLE_SUBFILNIG
        if (varp->num_subfiles > 1)
            *ndimsp = varp->ndims_org;
        else
#endif
            *ndimsp = varp->ndims;
    }

    return NC_NOERR;
}

/*----< ncmpi_inq_vardimid() >-----------------------------------------------*/
int
ncmpi_inq_vardimid(int ncid, int varid, int *dimids)
{
    int status;
    NC *ncp;
    NC_var *varp;

    status = ncmpii_NC_check_id(ncid, &ncp);
    if (status != NC_NOERR) return status;

    /* using NC_GLOBAL in varid is illegal for this API. See
     * http://www.unidata.ucar.edu/mailing_lists/archives/netcdfgroup/2015/msg00196.html
     */
    if (varid == NC_GLOBAL) DEBUG_RETURN_ERROR(NC_EGLOBAL)

    varp = elem_NC_vararray(&ncp->vars, varid);
    if (varp == NULL) DEBUG_RETURN_ERROR(NC_ENOTVAR)

    if (dimids != 0) {
#ifdef ENABLE_SUBFILING
        if (varp->num_subfiles > 1)
            memcpy(dimids, varp->dimids_org, (size_t)varp->ndims_org * SIZEOF_INT);
        else
#endif
            memcpy(dimids, varp->dimids, (size_t)varp->ndims * SIZEOF_INT);
    }

    return NC_NOERR;
}


/*----< ncmpi_inq_varnatts() >------------------------------------------------*/
int
ncmpi_inq_varnatts(int  ncid,
                   int  varid,
                   int *nattsp)
{
    int status;
    NC *ncp;
    NC_var *varp;

    if (varid == NC_GLOBAL)
        return ncmpi_inq_natts(ncid, nattsp);

    status = ncmpii_NC_check_id(ncid, &ncp);
    if (status != NC_NOERR) return status;

    varp = elem_NC_vararray(&ncp->vars, varid);
    if (varp == NULL) DEBUG_RETURN_ERROR(NC_ENOTVAR)

    if (nattsp != NULL)
        *nattsp = (int) varp->attrs.ndefined;

    return NC_NOERR;
}

/*----< ncmpi_rename_var() >--------------------------------------------------*/
/* This API is collective if called in data mode */
int
ncmpi_rename_var(int         ncid,
                 int         varid,
                 const char *newname)
{
    int file_ver, status=NC_NOERR, other, err, mpireturn;
    NC *ncp;
    NC_var *varp;

    status = ncmpii_NC_check_id(ncid, &ncp);
    if (status != NC_NOERR) return status;

    if (NC_readonly(ncp)) DEBUG_RETURN_ERROR(NC_EPERM)

    /* check if variable ID is valid*/
    status = ncmpii_NC_lookupvar(ncp, varid, &varp);
    if (status != NC_NOERR) return status;

    file_ver = 1;
    if (fIsSet(ncp->flags, NC_64BIT_OFFSET))
        file_ver = 2;
    else if (fIsSet(ncp->flags, NC_64BIT_DATA))
        file_ver = 5;

    status = ncmpii_NC_check_name(newname, file_ver);
    if (status != NC_NOERR) return status;

    /* check for name in use */
    other = ncmpii_NC_findvar(&ncp->vars, newname, &varp);
    if (other != -1)
        DEBUG_RETURN_ERROR(NC_ENAMEINUSE)

    /* if called in define mode, just update to the NC object */
    if (NC_indef(ncp)) {
        NC_string *newStr = ncmpii_new_NC_string(strlen(newname), newname);
        if (newStr == NULL) DEBUG_RETURN_ERROR(NC_ENOMEM)

        ncmpii_free_NC_string(varp->name);
        varp->name = newStr;
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

        if (nchars != (int) strlen(newname)) {
            /* newname's length is inconsistent with root's */
            printf("Warning: variable name(%s) used in %s() is inconsistent\n",
                   newname, __func__);
            if (status == NC_NOERR) DEBUG_ASSIGN_ERROR(status, NC_EMULTIDEFINE_VAR_NAME)
        }
    }

    /* ncmpii_set_NC_string() will check for strlen(newname) > nchars error */
    err = ncmpii_set_NC_string(varp->name, newname);
    if (status == NC_NOERR) status = err;

    /* PnetCDF expects all processes use the same name, However, when names
     * are not the same, only root's value is significant. Broadcast the
     * new name at root to overwrite new names at other processes.
     * (This API is collective if called in data mode)
     */
    TRACE_COMM(MPI_Bcast)(varp->name->cp, (int)varp->name->nchars, MPI_CHAR, 0,
                          ncp->nciop->comm);

    /* Let root write the entire header to the file. Note that we cannot just
     * update the variable name in its space occupied in the file header,
     * because if the file space occupied by the name shrinks, all the metadata
     * following it must be moved ahead.
     */
    err = ncmpii_write_header(ncp);
    if (status == NC_NOERR) status = err;

    return status;
}

/* some utility functions for debugging purpose */

/*----< ncmpi_inq_varoffset() >-----------------------------------------------*/
int
ncmpi_inq_varoffset(int         ncid,
                    int         varid,
                    MPI_Offset *offset)
{
    int     status;
    NC     *ncp;
    NC_var *varp;

    status = ncmpii_NC_check_id(ncid, &ncp);
    if (status != NC_NOERR) return status;

    /* using NC_GLOBAL in varid is illegal for this API. See
     * http://www.unidata.ucar.edu/mailing_lists/archives/netcdfgroup/2015/msg00196.html
     */
    if (varid == NC_GLOBAL) DEBUG_RETURN_ERROR(NC_EGLOBAL)

    varp = elem_NC_vararray(&ncp->vars, varid);
    if (varp == NULL) DEBUG_RETURN_ERROR(NC_ENOTVAR)

    if (offset != NULL)
        *offset = varp->begin;

    return NC_NOERR;
}

#ifdef __DEBUG

/*----< ncmpi_print_all_var_offsets() >---------------------------------------*/
int
ncmpi_print_all_var_offsets(int ncid) {
    int i;
    NC_var **vpp;
    NC *ncp;

    ncmpii_NC_check_id(ncid, &ncp);

    if (ncp->begin_var%1048576)
        printf("%s header size (ncp->begin_var)=%lld MB + %lld\n",
        ncp->nciop->path, ncp->begin_var/1048575, ncp->begin_var%1048576);
    else
        printf("%s header size (ncp->begin_var)=%lld MB\n",
        ncp->nciop->path, ncp->begin_var/1048575);

    vpp = ncp->vars.value;
    for (i=0; i<ncp->vars.ndefined; i++, vpp++) {
        char str[1024];
        MPI_Offset off = (*vpp)->begin;
        MPI_Offset rem = off % 1048576;;

        if (IS_RECVAR(*vpp))
            sprintf(str,"    Record variable \"%20s\": ",(*vpp)->name->cp);
        else
            sprintf(str,"non-record variable \"%20s\": ",(*vpp)->name->cp);

        if (rem)
            printf("%s offset=%12lld MB + %7lld len=%lld\n", str, off/1048576, rem,(*vpp)->len);
        else
            printf("%s offset=%12lld MB len=%lld\n", str, off/1048576,(*vpp)->len);
    }
    return NC_NOERR;
}

#endif

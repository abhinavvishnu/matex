/*
 *  Copyright (C) 2003, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id: header.c 2322 2016-02-26 22:47:06Z wkliao $ */

#if HAVE_CONFIG_H
# include "ncconfig.h"
#endif

#include <assert.h>
#include <string.h>  /* memcpy(), memcmp() */
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#include <stdio.h>

#include <mpi.h>

#include "nc.h"
#include "ncx.h"
#include "macro.h"

#ifdef SIZEOF_INT
# if SIZEOF_INT == 4
#  define lld(x) (x)
# elif  SIZEOF_INT == 8
#  define lld(x) (long long)(x)
# endif
#endif

/* Prototypes for functions used only in this file */
static MPI_Offset hdr_len_NC_name(const NC_string *ncstrp, int sizeof_t);
static MPI_Offset hdr_len_NC_dim(const NC_dim *dimp, int sizeof_t);
static MPI_Offset hdr_len_NC_dimarray(const NC_dimarray *ncap, int sizeof_t);
static MPI_Offset hdr_len_NC_attr(const NC_attr *attrp, int sizeof_t);
static MPI_Offset hdr_len_NC_attrarray(const NC_attrarray *ncap, int sizeof_t);
static MPI_Offset hdr_len_NC_var(const NC_var *varp, int sizeof_off_t, int sizeof_t);
static MPI_Offset hdr_len_NC_vararray(const NC_vararray *ncap, int sizeof_t, int sizeof_off_t);
static int hdr_put_NC_name(bufferinfo *pbp, const NC_string *ncstrp);
static int hdr_put_NC_attrV(bufferinfo *pbp, const NC_attr *attrp);
static int hdr_put_NC_dim(bufferinfo *pbp, const NC_dim *dimp);
static int hdr_put_NC_attr(bufferinfo *pbp, const NC_attr *attrp);
static int hdr_put_NC_var(bufferinfo *pbp, const NC_var *varp);
static int hdr_put_NC_dimarray(bufferinfo *pbp, const NC_dimarray *ncap);
static int hdr_put_NC_attrarray(bufferinfo *pbp, const NC_attrarray *ncap);
static int hdr_put_NC_vararray(bufferinfo *pbp, const NC_vararray *ncap);
static int hdr_fetch(bufferinfo *gbp);
static int hdr_check_buffer(bufferinfo *gbp, MPI_Offset nextread);
static int hdr_get_NCtype(bufferinfo *gbp, NCtype *typep);
static int hdr_get_NC_name(bufferinfo *gbp, NC_string **ncstrpp);
static int hdr_get_NC_dim(bufferinfo *gbp, NC_dim **dimpp);
static int hdr_get_NC_dimarray(bufferinfo *gbp, NC_dimarray *ncap);
static int hdr_get_nc_type(bufferinfo *gbp, nc_type *typep);
static int hdr_get_NC_attrV(bufferinfo *gbp, NC_attr *attrp);
static int hdr_get_NC_attr(bufferinfo *gbp, NC_attr **attrpp);
static int hdr_get_NC_attrarray(bufferinfo *gbp, NC_attrarray *ncap);
static int hdr_get_NC_var(bufferinfo *gbp, NC_var **varpp);
static int hdr_get_NC_vararray(bufferinfo *gbp, NC_vararray *ncap);

/*
 * "magic number" at beginning of file: 0x43444601 (big endian)
 */
static const schar ncmagic1[] = {'C', 'D', 'F', 0x01};
static const schar ncmagic2[] = {'C', 'D', 'F', 0x02};
static const schar ncmagic5[] = {'C', 'D', 'F', 0x05};

/*
 * Recompute the shapes of all variables
 * Sets ncp->begin_var to start of first variable.
 * Sets ncp->begin_rec to start of first record variable.
 * Returns -1 on error. The only possible error is an reference
 * to a non existent dimension, which would occur for a corrupt
 * netcdf file.
 */
int
ncmpii_NC_computeshapes(NC *ncp)
{
    NC_var **vpp = (NC_var **)ncp->vars.value;
    NC_var *const *const end = &vpp[ncp->vars.ndefined];
    NC_var *first_var = NULL;       /* first "non-record" var */
    NC_var *first_rec = NULL;       /* first "record" var */
    int status;

    ncp->begin_var = (MPI_Offset) ncp->xsz;
    ncp->begin_rec = (MPI_Offset) ncp->xsz;
    ncp->recsize = 0;

    if (ncp->vars.ndefined == 0) return NC_NOERR;

    for ( /*NADA*/; vpp < end; vpp++) {
        /* (*vpp)->len is recomputed from dimensions in ncmpii_NC_var_shape64() */
        status = ncmpii_NC_var_shape64(ncp, *vpp, &ncp->dims);

        if (status != NC_NOERR) return status ;

        if (IS_RECVAR(*vpp)) {
            if (first_rec == NULL)
                first_rec = *vpp;
            ncp->recsize += (*vpp)->len;
        }
        else {
            if (first_var == NULL)
            first_var = *vpp;
            /*
             * Overwritten each time thru.
             * Usually overwritten in first_rec != NULL clause.
             */
            ncp->begin_rec = (*vpp)->begin + (MPI_Offset)(*vpp)->len;
        }
    }

    if (first_rec != NULL) {
        if (ncp->begin_rec > first_rec->begin)
            DEBUG_RETURN_ERROR(NC_ENOTNC) /* not a netCDF file or corrupted */

        ncp->begin_rec = first_rec->begin;
        /*
         * for special case of exactly one record variable, pack value
         */
        if (ncp->recsize == first_rec->len)
            ncp->recsize = *first_rec->dsizes * first_rec->xsz;
    }

    if (first_var != NULL)
        ncp->begin_var = first_var->begin;
    else
        ncp->begin_var = ncp->begin_rec;

    if (ncp->begin_var <= 0 ||
        ncp->xsz > ncp->begin_var ||
        ncp->begin_rec <= 0 ||
        ncp->begin_var > ncp->begin_rec)
        DEBUG_RETURN_ERROR(NC_ENOTNC) /* not a netCDF file or corrupted */

    return NC_NOERR;
}

/*
 * To compute how much space will the xdr'd header take
 */

#define X_SIZEOF_NC_TYPE X_SIZEOF_INT
#define X_SIZEOF_NCTYPE X_SIZEOF_INT

/*----< hdr_len_NC_name() >--------------------------------------------------*/
inline static MPI_Offset
hdr_len_NC_name(const NC_string *ncstrp,
                int              sizeof_t)     /* NON_NEG */
{
    /* netCDF file format:
     * name       = nelems  namestring
     * nelems     = NON_NEG
     * namestring = ID1 [IDN ...] padding
     * ID1        = alphanumeric | '_'
     * IDN        = alphanumeric | special1 | special2
     * padding    = <0, 1, 2, or 3 bytes to next 4-byte boundary>
     * NON_NEG    = <non-negative INT> |  // CDF-1 and CDF-2
     *              <non-negative INT64>  // CDF-5
     */
    MPI_Offset sz = sizeof_t; /* nelems */

    assert(ncstrp != NULL);

    if (ncstrp->nchars != 0)  /* namestring */
        sz += _RNDUP(ncstrp->nchars, X_ALIGN);

    return sz;
}

/*----< hdr_len_NC_dim() >---------------------------------------------------*/
inline static MPI_Offset
hdr_len_NC_dim(const NC_dim *dimp,
               int           sizeof_t)     /* NON_NEG */
{
    /* netCDF file format:
     *  ...
     * dim        = name  dim_length
     * dim_length = NON_NEG
     * NON_NEG    = <non-negative INT> |  // CDF-1 and CDF-2
     *              <non-negative INT64>  // CDF-5
     */
    MPI_Offset sz;

    assert(dimp != NULL);

    sz = hdr_len_NC_name(dimp->name, sizeof_t); /* name */
    sz += sizeof_t;                             /* dim_length */

    return sz;
}

/*----< hdr_len_NC_dimarray() >----------------------------------------------*/
inline static MPI_Offset
hdr_len_NC_dimarray(const NC_dimarray *ncap,
                    int                sizeof_t)     /* NON_NEG */
{
    /* netCDF file format:
     *  ...
     * dim_list     = ABSENT | NC_DIMENSION  nelems  [dim ...]
     * ABSENT       = ZERO  ZERO |  // list is not present for CDF-1 and 2
     *                ZERO  ZERO64  // for CDF-5
     * ZERO         = \x00 \x00 \x00 \x00                      // 32-bit zero
     * ZERO64       = \x00 \x00 \x00 \x00 \x00 \x00 \x00 \x00  // 64-bit zero
     * NC_DIMENSION = \x00 \x00 \x00 \x0A         // tag for list of dimensions
     * nelems       = NON_NEG       // number of elements in following sequence
     * NON_NEG      = <non-negative INT> |        // CDF-1 and CDF-2
     *                <non-negative INT64>        // CDF-5
     */
    int i;
    MPI_Offset xlen;

    xlen = X_SIZEOF_NCTYPE;           /* NC_DIMENSION */
    xlen += sizeof_t;                 /* nelems */

    if (ncap == NULL) /* ABSENT: no dimension is defined */
        return xlen;

    /* [dim ...] */
    for (i=0; i<ncap->ndefined; i++)
        xlen += hdr_len_NC_dim(ncap->value[i], sizeof_t);

    return xlen;
}

/*----< hdr_len_NC_attr() >--------------------------------------------------*/
inline static MPI_Offset
hdr_len_NC_attr(const NC_attr *attrp,
                int            sizeof_t)     /* NON_NEG */
{
    /* netCDF file format:
     *  ...
     * attr    = name  nc_type  nelems  [values ...]
     * nc_type = NC_BYTE | NC_CHAR | NC_SHORT | ...
     * nelems  = NON_NEG       // number of elements in following sequence
     * values  = bytes | chars | shorts | ints | floats | doubles
     * bytes   = [BYTE ...]  padding
     * chars   = [CHAR ...]  padding
     * shorts  = [SHORT ...]  padding
     * ints    = [INT ...]
     * floats  = [FLOAT ...]
     * doubles = [DOUBLE ...]
     * padding = <0, 1, 2, or 3 bytes to next 4-byte boundary>
     * NON_NEG = <non-negative INT> |  // CDF-1 and CDF-2
     *           <non-negative INT64>  // CDF-5
     */
    MPI_Offset sz;

    assert(attrp != NULL);

    sz  = hdr_len_NC_name(attrp->name, sizeof_t); /* name */
    sz += X_SIZEOF_NC_TYPE;                       /* nc_type */
    sz += sizeof_t;                               /* nelems */
    sz += attrp->xsz;                             /* [values ...] */

    return sz;
}

/*----< hdr_len_NC_attrarray() >---------------------------------------------*/
inline static MPI_Offset
hdr_len_NC_attrarray(const NC_attrarray *ncap,
                     int                 sizeof_t)     /* NON_NEG */
{
    /* netCDF file format:
     *  ...
     * att_list     = ABSENT | NC_ATTRIBUTE  nelems  [attr ...]
     * ABSENT       = ZERO  ZERO |  // list is not present for CDF-1 and 2
     *                ZERO  ZERO64  // for CDF-5
     * ZERO         = \x00 \x00 \x00 \x00                      // 32-bit zero
     * ZERO64       = \x00 \x00 \x00 \x00 \x00 \x00 \x00 \x00  // 64-bit zero
     * NC_ATTRIBUTE = \x00 \x00 \x00 \x0C         // tag for list of attributes
     * nelems       = NON_NEG       // number of elements in following sequence
     * NON_NEG      = <non-negative INT> |        // CDF-1 and CDF-2
     *                <non-negative INT64>        // CDF-5
     */
    int i;
    MPI_Offset xlen;

    xlen = X_SIZEOF_NCTYPE;        /* NC_ATTRIBUTE */
    xlen += sizeof_t;              /* nelems */

    if (ncap == NULL) /* ABSENT: no attribute is defined */
        return xlen;

    for (i=0; i<ncap->ndefined; i++) /* [attr ...] */
        xlen += hdr_len_NC_attr(ncap->value[i], sizeof_t);

    return xlen;
}

/*----< hdr_len_NC_var() >---------------------------------------------------*/
inline static MPI_Offset
hdr_len_NC_var(const NC_var *varp,
               int           sizeof_off_t, /* OFFSET */
               int           sizeof_t)     /* NON_NEG */
{
    /* netCDF file format:
     * netcdf_file = header data
     * header      = magic numrecs dim_list gatt_list var_list
     *  ...
     * var         = name nelems [dimid ...] vatt_list nc_type vsize begin
     * nelems      = NON_NEG
     * dimid       = NON_NEG
     * vatt_list   = att_list
     * nc_type     = NC_BYTE | NC_CHAR | NC_SHORT | ...
     * vsize       = NON_NEG
     * begin       = OFFSET        // Variable start location.
     * OFFSET      = <non-negative INT> |  // CDF-1
     *               <non-negative INT64>  // CDF-2 and CDF-5
     * NON_NEG     = <non-negative INT> |  // CDF-1 and CDF-2
     *               <non-negative INT64>  // CDF-5
     */
    MPI_Offset sz;

    assert(varp != NULL);

    /* for CDF-1, sizeof_off_t == 4 && sizeof_t == 4
     * for CDF-2, sizeof_off_t == 8 && sizeof_t == 4
     * for CDF-5, sizeof_off_t == 8 && sizeof_t == 8
     */
    sz = hdr_len_NC_name(varp->name, sizeof_t);         /* name */
    sz += sizeof_t;                                     /* nelems */
    sz += sizeof_t * varp->ndims;                       /* [dimid ...] */
    sz += hdr_len_NC_attrarray(&varp->attrs, sizeof_t); /* vatt_list */
    sz += X_SIZEOF_NC_TYPE;                             /* nc_type */
    sz += sizeof_t;                                     /* vsize */
    sz += sizeof_off_t;                                 /* begin */

    return sz;
}

/*----< hdr_len_NC_vararray() >----------------------------------------------*/
inline static MPI_Offset
hdr_len_NC_vararray(const NC_vararray *ncap,
                    int                sizeof_t,     /* NON_NEG */
                    int                sizeof_off_t) /* OFFSET */
{
    /* netCDF file format:
     * netcdf_file = header  data
     * header      = magic  numrecs  dim_list  gatt_list  var_list
     *  ...
     * var_list    = ABSENT | NC_VARIABLE   nelems  [var ...]
     * ABSENT      = ZERO  ZERO |  // list is not present for CDF-1 and 2
     *               ZERO  ZERO64  // for CDF-5
     * ZERO        = \x00 \x00 \x00 \x00                      // 32-bit zero
     * ZERO64      = \x00 \x00 \x00 \x00 \x00 \x00 \x00 \x00  // 64-bit zero
     * NC_VARIABLE = \x00 \x00 \x00 \x0B         // tag for list of variables
     * nelems      = NON_NEG       // number of elements in following sequence
     * NON_NEG     = <non-negative INT> |        // CDF-1 and CDF-2
     *               <non-negative INT64>        // CDF-5
     */
    int i;
    MPI_Offset xlen;

    xlen = X_SIZEOF_NCTYPE;           /* NC_VARIABLE */
    xlen += sizeof_t;                 /* nelems */

    if (ncap == NULL) /* ABSENT: no variable is defined */
        return xlen;

    /* for CDF-1, sizeof_off_t == 4 && sizeof_t == 4
     * for CDF-2, sizeof_off_t == 8 && sizeof_t == 4
     * for CDF-5, sizeof_off_t == 8 && sizeof_t == 8
     */
    for (i=0; i<ncap->ndefined; i++)  /* [var ...] */
        xlen += hdr_len_NC_var(ncap->value[i], sizeof_off_t, sizeof_t);

    return xlen;
}

/*----< ncmpii_hdr_len_NC() >------------------------------------------------*/
MPI_Offset
ncmpii_hdr_len_NC(const NC *ncp)
{
    /* netCDF file format:
     * netcdf_file = header  data
     * header      = magic  numrecs  dim_list  gatt_list  var_list
     *  ...
     * numrecs     = NON_NEG | STREAMING   // length of record dimension
     * NON_NEG     = <non-negative INT> |  // CDF-1 and CDF-2
     *               <non-negative INT64>  // CDF-5
     */

    int sizeof_t, sizeof_off_t;
    MPI_Offset xlen;

    assert(ncp != NULL);

    if (fIsSet(ncp->flags, NC_64BIT_DATA)) {        /* CDF-5 */
        sizeof_t     = X_SIZEOF_INT64; /* 8-byte integer for all integers */
        sizeof_off_t = X_SIZEOF_INT64; /* 8-byte integer for var begin */
    }
    else if (fIsSet(ncp->flags, NC_64BIT_OFFSET)) { /* CDF-2 */
        sizeof_t     = X_SIZEOF_INT; /* 4-byte integer in CDF-1 */
        sizeof_off_t = X_SIZEOF_INT64; /* 8-byte integer for var begin */
    }
    else { /* CDF-1 */
        sizeof_t     = X_SIZEOF_INT; /* 4-byte integer in CDF-1 */
        sizeof_off_t = X_SIZEOF_INT; /* 4-byte integer in CDF-1 */
    }

    xlen  = sizeof(ncmagic1);                                          /* magic */
    xlen += sizeof_t;                                                  /* numrecs */
    xlen += hdr_len_NC_dimarray(&ncp->dims,   sizeof_t);               /* dim_list */
    xlen += hdr_len_NC_attrarray(&ncp->attrs, sizeof_t);               /* gatt_list */
    xlen += hdr_len_NC_vararray(&ncp->vars,   sizeof_t, sizeof_off_t); /* var_list */

    return xlen; /* return the header size (not yet aligned) */
}

/* Begin Of put NC */

/*----< hdr_put_NC_name() >--------------------------------------------------*/
inline static int
hdr_put_NC_name(bufferinfo      *pbp,
                const NC_string *ncstrp)
{
    /* netCDF file format:
     *  ...
     * name       = nelems  namestring
     * nelems     = NON_NEG
     * namestring = ID1 [IDN ...] padding
     * ID1        = alphanumeric | '_'
     * IDN        = alphanumeric | special1 | special2
     * padding    = <0, 1, 2, or 3 bytes to next 4-byte boundary>
     * NON_NEG    = <non-negative INT> |  // CDF-1 and CDF-2
     *              <non-negative INT64>  // CDF-5
     */
    int status;

    /* copy nelems */
    if (pbp->version == 5)
        status = ncmpix_put_uint64((void**)(&pbp->pos), ncstrp->nchars);
    else {
        if (ncstrp->nchars != (uint)ncstrp->nchars) DEBUG_RETURN_ERROR(NC_EINTOVERFLOW)
        status = ncmpix_put_uint32((void**)(&pbp->pos), (uint)ncstrp->nchars);
    }
    if (status != NC_NOERR) return status;

    /* copy namestring */
    status = ncmpix_pad_putn_text(&pbp->pos, ncstrp->nchars, ncstrp->cp);
    if (status != NC_NOERR) return status;

    return NC_NOERR;
}

/*----< hdr_put_NC_attrV() >-------------------------------------------------*/
/*
 * Put the values of an attribute
 */
inline static int
hdr_put_NC_attrV(bufferinfo    *pbp,
                 const NC_attr *attrp)
{
    /* netCDF file format:
     *  ...
     * attr    = name  nc_type  nelems  [values ...]
     *  ...
     * values  = bytes | chars | shorts | ints | floats | doubles
     * bytes   = [BYTE ...]  padding
     * chars   = [CHAR ...]  padding
     * shorts  = [SHORT ...]  padding
     * ints    = [INT ...]
     * floats  = [FLOAT ...]
     * doubles = [DOUBLE ...]
     * padding = <0, 1, 2, or 3 bytes to next 4-byte boundary>
     */
    MPI_Offset padding, sz;

    /* ncmpix_len_nctype() returns the element size (unaligned) of attrp->type
       attrp->xsz is the aligned total size of attribute values
     */
    sz = ncmpix_len_nctype(attrp->type);
    sz *= attrp->nelems;
    padding = attrp->xsz - sz;

    if (sz != (size_t) sz) DEBUG_RETURN_ERROR(NC_EINTOVERFLOW)
    memcpy(pbp->pos, attrp->xvalue, (size_t)sz);
    pbp->pos = (void *)((char *)pbp->pos + sz);

    /* zero-padding is per buffer, not per element */
    memset(pbp->pos, 0, (size_t)padding);
    pbp->pos = (void *)((char *)pbp->pos + padding);

    return NC_NOERR;
}

/*----< hdr_put_NC_dim() >---------------------------------------------------*/
inline static int
hdr_put_NC_dim(bufferinfo   *pbp,
               const NC_dim *dimp)
{
    /* netCDF file format:
     *  ...
     * dim        = name  dim_length
     * dim_length = NON_NEG
     * NON_NEG    = <non-negative INT> |  // CDF-1 and CDF-2
     *              <non-negative INT64>  // CDF-5
     */
    int status;

    /* copy name */
    status = hdr_put_NC_name(pbp, dimp->name);
    if (status != NC_NOERR) return status;

    /* copy dim_length */
    if (pbp->version == 5)
        status = ncmpix_put_uint64((void**)(&pbp->pos), dimp->size);
    else {
        /* TODO: Isn't checking dimension size already done in def_dim()? */
        if (dimp->size != (uint)dimp->size) DEBUG_RETURN_ERROR(NC_EINTOVERFLOW)
        status = ncmpix_put_uint32((void**)(&pbp->pos), (uint)dimp->size);
    }
    if (status != NC_NOERR) return status;

    return NC_NOERR;
}

/*----< hdr_put_NC_dimarray() >----------------------------------------------*/
inline static int
hdr_put_NC_dimarray(bufferinfo        *pbp,
                    const NC_dimarray *ncap)
{
    /* netCDF file format:
     *  ...
     * dim_list     = ABSENT | NC_DIMENSION  nelems  [dim ...]
     * ABSENT       = ZERO  ZERO |  // list is not present for CDF-1 and 2
     *                ZERO  ZERO64  // for CDF-5
     * ZERO         = \x00 \x00 \x00 \x00                      // 32-bit zero
     * ZERO64       = \x00 \x00 \x00 \x00 \x00 \x00 \x00 \x00  // 64-bit zero
     * NC_DIMENSION = \x00 \x00 \x00 \x0A         // tag for list of dimensions
     * nelems       = NON_NEG       // number of elements in following sequence
     * NON_NEG      = <non-negative INT> |        // CDF-1 and CDF-2
     *                <non-negative INT64>        // CDF-5
     */
    int i, status;

    assert(pbp != NULL);

    if (ncap == NULL || ncap->ndefined == 0) { /* ABSENT */
        status = ncmpix_put_uint32((void**)(&pbp->pos), NC_UNSPECIFIED);
        if (status != NC_NOERR) return status;

        /* put a ZERO or ZERO64 depending on which CDF format */
        if (pbp->version == 5)
            status = ncmpix_put_uint64((void**)(&pbp->pos), 0);
        else
            status = ncmpix_put_uint32((void**)(&pbp->pos), 0);
        if (status != NC_NOERR) return status;
    }
    else {
        /* copy NC_DIMENSION */
        status = ncmpix_put_uint32((void**)(&pbp->pos), NC_DIMENSION);
        if (status != NC_NOERR) return status;

        /* copy nelems */
        if (pbp->version == 5)
            status = ncmpix_put_uint64((void**)(&pbp->pos), ncap->ndefined);
        else
            status = ncmpix_put_uint32((void**)(&pbp->pos), ncap->ndefined);
        if (status != NC_NOERR) return status;

        /* copy [dim ...] */
        for (i=0; i<ncap->ndefined; i++) {
            status = hdr_put_NC_dim(pbp, ncap->value[i]);
            if (status != NC_NOERR) return status;
        }
    }

    return NC_NOERR;
}

/*----< hdr_put_NC_attr() >--------------------------------------------------*/
inline static int
hdr_put_NC_attr(bufferinfo    *pbp,
                const NC_attr *attrp)
{
    /* netCDF file format:
     *  ...
     * attr    = name  nc_type  nelems  [values ...]
     * nc_type = NC_BYTE | NC_CHAR | NC_SHORT | ...
     * nelems  = NON_NEG       // number of elements in following sequence
     * NON_NEG = <non-negative INT> |  // CDF-1 and CDF-2
     *           <non-negative INT64>  // CDF-5
     */
    int status;

    /* copy name */
    status = hdr_put_NC_name(pbp, attrp->name);
    if (status != NC_NOERR) return status;

    /* copy nc_type */
    status = ncmpix_put_uint32((void**)(&pbp->pos), attrp->type);
    if (status != NC_NOERR) return status;

    /* copy nelems */
    if (pbp->version == 5)
        status = ncmpix_put_uint64((void**)(&pbp->pos), attrp->nelems);
    else {
        if (attrp->nelems != (uint)attrp->nelems) DEBUG_RETURN_ERROR(NC_EINTOVERFLOW)
        status = ncmpix_put_uint32((void**)(&pbp->pos), (uint)attrp->nelems);
    }
    if (status != NC_NOERR) return status;

    /* copy [values ...] */
    status = hdr_put_NC_attrV(pbp, attrp);
    if (status != NC_NOERR) return status;

    return NC_NOERR;
}

/*----< hdr_put_NC_attrarray() >---------------------------------------------*/
inline static int
hdr_put_NC_attrarray(bufferinfo         *pbp,
                     const NC_attrarray *ncap)
{
    /* netCDF file format:
     *  ...
     * att_list     = ABSENT | NC_ATTRIBUTE  nelems  [attr ...]
     * ABSENT       = ZERO  ZERO |  // list is not present for CDF-1 and 2
     *                ZERO  ZERO64  // for CDF-5
     * ZERO         = \x00 \x00 \x00 \x00                      // 32-bit zero
     * ZERO64       = \x00 \x00 \x00 \x00 \x00 \x00 \x00 \x00  // 64-bit zero
     * NC_ATTRIBUTE = \x00 \x00 \x00 \x0C         // tag for list of attributes
     * nelems       = NON_NEG       // number of elements in following sequence
     * NON_NEG      = <non-negative INT> |        // CDF-1 and CDF-2
     *                <non-negative INT64>        // CDF-5
     */
    int i, status;

    assert(pbp != NULL);

    if (ncap == NULL || ncap->ndefined == 0) { /* ABSENT */
        status = ncmpix_put_uint32((void**)(&pbp->pos), NC_UNSPECIFIED);
        if (status != NC_NOERR) return status;

        /* put a ZERO or ZERO64 depending on which CDF format */
        if (pbp->version == 5)
            status = ncmpix_put_uint64((void**)(&pbp->pos), 0);
        else
            status = ncmpix_put_uint32((void**)(&pbp->pos), 0);
        if (status != NC_NOERR) return status;
    }
    else {
        /* copy NC_ATTRIBUTE */
        status = ncmpix_put_uint32((void**)(&pbp->pos), NC_ATTRIBUTE);
        if (status != NC_NOERR) return status;

        /* copy nelems */
        if (pbp->version == 5)
            status = ncmpix_put_uint64((void**)(&pbp->pos), ncap->ndefined);
        else
            status = ncmpix_put_uint32((void**)(&pbp->pos), ncap->ndefined);
        if (status != NC_NOERR) return status;

        /* copy [attr ...] */
        for (i=0; i<ncap->ndefined; i++) {
            status = hdr_put_NC_attr(pbp, ncap->value[i]);
            if (status != NC_NOERR) return status;
        }
    }

    return NC_NOERR;
}

/*----< hdr_put_NC_var() >---------------------------------------------------*/
static int
hdr_put_NC_var(bufferinfo   *pbp,
               const NC_var *varp)
{
    /* netCDF file format:
     * netcdf_file = header data
     * header      = magic numrecs dim_list gatt_list var_list
     *  ...
     * var         = name nelems [dimid ...] vatt_list nc_type vsize begin
     * nelems      = NON_NEG
     * dimid       = NON_NEG
     * vatt_list   = att_list
     * nc_type     = NC_BYTE | NC_CHAR | NC_SHORT | ...
     * vsize       = NON_NEG
     * begin       = OFFSET        // Variable start location.
     * OFFSET      = <non-negative INT> |  // CDF-1
     *               <non-negative INT64>  // CDF-2 and CDF-5
     * NON_NEG     = <non-negative INT> |  // CDF-1 and CDF-2
     *               <non-negative INT64>  // CDF-5
     */
    int i, status;

    /* copy name */
    status = hdr_put_NC_name(pbp, varp->name);
    if (status != NC_NOERR) return status;

    /* copy nelems */
    if (pbp->version == 5)
        status = ncmpix_put_uint64((void**)(&pbp->pos), varp->ndims);
    else
        status = ncmpix_put_uint32((void**)(&pbp->pos), varp->ndims);
    if (status != NC_NOERR) return status;

    /* copy [dimid ...] */
    for (i=0; i<varp->ndims; i++) {
        if (pbp->version == 5)
            status = ncmpix_put_uint64((void**)(&pbp->pos), varp->dimids[i]);
        else
            status = ncmpix_put_uint32((void**)(&pbp->pos), varp->dimids[i]);
        if (status != NC_NOERR) return status;
    }

    /* copy vatt_list */
    status = hdr_put_NC_attrarray(pbp, &varp->attrs);
    if (status != NC_NOERR) return status;

    /* copy nc_type */
    status = ncmpix_put_uint32((void**)(&pbp->pos), varp->type);
    if (status != NC_NOERR) return status;

    /* copy vsize */
    /* in CDF-1 and CDF-2, a variable's size in the header is a 32-bit integer
     * in CDF-5, it is a 64-bit integer
     */
    if (pbp->version == 5)
        status = ncmpix_put_uint64((void**)(&pbp->pos), varp->len);
    else {
        /* Special case, when there is no record variable, the last fixed-size
         * variable can be larger than 2 GiB if its file starting offset is
         * less than 2 GiB. This checking has already been done in the call
         * to ncmpii_NC_check_vlens() in ncmpii_NC_enddef().
         *
         * if (varp->len != (int)varp->len) DEBUG_RETURN_ERROR(NC_EVARSIZE)
         */
        status = ncmpix_put_uint32((void**)(&pbp->pos), (uint)varp->len);
    }
    if (status != NC_NOERR) return status;

    /* copy begin */
    /* in CDF-1 header, a variable's starting file offset is a 32-bit integer
     * in CDF-2 and CDF-5, it is a 64-bit integer
     */
    if (pbp->version == 1) {
        if (varp->begin != (uint)varp->begin) DEBUG_RETURN_ERROR(NC_EINTOVERFLOW)
        status = ncmpix_put_uint32((void**)(&pbp->pos), (uint)varp->begin);
    }
    else
        status = ncmpix_put_uint64((void**)(&pbp->pos), varp->begin);
    if (status != NC_NOERR) return status;

    return NC_NOERR;
}

/*----< hdr_put_NC_vararray() >----------------------------------------------*/
static int
hdr_put_NC_vararray(bufferinfo        *pbp,
                    const NC_vararray *ncap)
{
    /* netCDF file format:
     * netcdf_file = header  data
     * header      = magic  numrecs  dim_list  gatt_list  var_list
     *  ...
     * var_list    = ABSENT | NC_VARIABLE   nelems  [var ...]
     * ABSENT      = ZERO  ZERO |  // list is not present for CDF-1 and 2
     *               ZERO  ZERO64  // for CDF-5
     * ZERO        = \x00 \x00 \x00 \x00                      // 32-bit zero
     * ZERO64      = \x00 \x00 \x00 \x00 \x00 \x00 \x00 \x00  // 64-bit zero
     * NC_VARIABLE = \x00 \x00 \x00 \x0B         // tag for list of variables
     * nelems      = NON_NEG       // number of elements in following sequence
     * NON_NEG     = <non-negative INT> |        // CDF-1 and CDF-2
     *               <non-negative INT64>        // CDF-5
     */
    int i, status;

    assert(pbp != NULL);

    if (ncap == NULL || ncap->ndefined == 0) { /* ABSENT */
        status = ncmpix_put_uint32((void**)(&pbp->pos), NC_UNSPECIFIED);
        if (status != NC_NOERR) return status;

        /* put a ZERO or ZERO64 depending on which CDF format */
        if (pbp->version == 5)
            status = ncmpix_put_uint64((void**)(&pbp->pos), 0);
        else
            status = ncmpix_put_uint32((void**)(&pbp->pos), 0);
        if (status != NC_NOERR) return status;
    }
    else {
        /* copy NC_VARIABLE */
        status = ncmpix_put_uint32((void**)(&pbp->pos), NC_VARIABLE);
        if (status != NC_NOERR) return status;

        /* copy nelems */
        if (pbp->version == 5)
            status = ncmpix_put_uint64((void**)(&pbp->pos), ncap->ndefined);
        else
            status = ncmpix_put_uint32((void**)(&pbp->pos), ncap->ndefined);
        if (status != NC_NOERR) return status;

        /* copy [var ...] */
        for (i=0; i<ncap->ndefined; i++) {
            status = hdr_put_NC_var(pbp, ncap->value[i]);
            if (status != NC_NOERR) return status;
        }
    }

    return NC_NOERR;
}

/*----< ncmpii_hdr_put_NC() >------------------------------------------------*/
/* fill the file header into the I/O buffer, buf
 * this function is collective */
int
ncmpii_hdr_put_NC(NC   *ncp,
                  void *buf) {
    int status;
    bufferinfo putbuf;
    MPI_Offset nrecs=0;

    putbuf.nciop     = NULL;
    putbuf.offset    = 0;
    putbuf.pos       = buf;
    putbuf.base      = buf;
    putbuf.size      = ncp->xsz;
    putbuf.put_size  = 0;
    putbuf.get_size  = 0;
    putbuf.safe_mode = ncp->safe_mode;

    /* netCDF file format:
     * netcdf_file  = header  data
     * header       = magic  numrecs  dim_list  gatt_list  var_list
     */

    /* copy "magic", 4 characters */
    if (ncp->flags & NC_64BIT_DATA) {
        putbuf.version = 5;
        status = ncmpix_putn_schar_schar(&putbuf.pos, sizeof(ncmagic5), ncmagic5);
    }
    else if (ncp->flags & NC_64BIT_OFFSET) {
        putbuf.version = 2;
        status = ncmpix_putn_schar_schar(&putbuf.pos, sizeof(ncmagic2), ncmagic2);
    }
    else {
        putbuf.version = 1;
        status = ncmpix_putn_schar_schar(&putbuf.pos, sizeof(ncmagic1), ncmagic1);
    }
    if (status != NC_NOERR) return status;

    /* copy numrecs, number of records */
    nrecs = ncp->numrecs;
    if (ncp->flags & NC_64BIT_DATA)
        status = ncmpix_put_uint64((void**)(&putbuf.pos), nrecs);
    else {
        if (nrecs != (uint)nrecs) DEBUG_RETURN_ERROR(NC_EINTOVERFLOW)
        status = ncmpix_put_uint32((void**)(&putbuf.pos), (uint)nrecs);
    }
    if (status != NC_NOERR) return status;

    /* copy dim_list */
    status = hdr_put_NC_dimarray(&putbuf, &ncp->dims);
    if (status != NC_NOERR) return status;

    /* copy gatt_list */
    status = hdr_put_NC_attrarray(&putbuf, &ncp->attrs);
    if (status != NC_NOERR) return status;

    /* copy var_list */
    status = hdr_put_NC_vararray(&putbuf, &ncp->vars);
    if (status != NC_NOERR) return status;

    ncp->nciop->put_size += putbuf.put_size;
    ncp->nciop->get_size += putbuf.get_size;

    return NC_NOERR;
}

/* End Of put NC */

/* Begin Of get NC */

/*
 * Fetch the next header chunk.  the chunk is 'gbp->size' bytes big
 * Takes care to not overwrite leftover (unused) data in the buffer before
 * fetching a new chunk: the current approach is to re-read the extra data.
 *
 * NOTE: An alternate approach (which we do not do) would be to save the old
 *       data, read the next chunk and then copy the old data into the new
 *       chunk.  This alternate approach might help if it is important for reads
 *       to be aligned.
 */
static int
hdr_fetch(bufferinfo *gbp) {
    int rank, err=NC_NOERR, mpireturn;
    MPI_Offset slack;        /* any leftover data in the buffer */
    MPI_Aint pos_addr, base_addr;

    assert(gbp->base != NULL);

#ifdef HAVE_MPI_GET_ADDRESS
    MPI_Get_address(gbp->pos,  &pos_addr);
    MPI_Get_address(gbp->base, &base_addr);
#else
    MPI_Address(gbp->pos,  &pos_addr);
    MPI_Address(gbp->base, &base_addr);
#endif
    slack = gbp->size - (pos_addr - base_addr);
    /* . if gbp->pos and gbp->base are the same, there is no leftover buffer
     *   data to worry about.
     * In the other extreme, where gbp->size == (gbp->pos - gbp->base), then
     * all data in the buffer has been consumed */
    if (slack == gbp->size) slack = 0;

    if (gbp->size != (size_t)gbp->size) DEBUG_RETURN_ERROR(NC_EINTOVERFLOW)
    if (gbp->size != (int)gbp->size) DEBUG_RETURN_ERROR(NC_EINTOVERFLOW)

    memset(gbp->base, 0, (size_t)gbp->size);
    gbp->pos = gbp->base;
    gbp->index = 0;

    MPI_Comm_rank(gbp->nciop->comm, &rank);
    if (rank == 0) {
        MPI_Status mpistatus;
        /* fileview is already entire file visible and MPI_File_read_at does
           not change the file pointer */
        TRACE_IO(MPI_File_read_at)(gbp->nciop->collective_fh,
                                   (gbp->offset)-slack, gbp->base,
                                   (int)gbp->size, MPI_BYTE, &mpistatus);
        if (mpireturn != MPI_SUCCESS) {
            err = ncmpii_handle_error(mpireturn, "MPI_File_read_at");
            if (err == NC_EFILE) DEBUG_ASSIGN_ERROR(err, NC_EREAD)
        }
        else {
            int get_size;
            MPI_Get_count(&mpistatus, MPI_BYTE, &get_size);
            gbp->get_size += get_size;
        }
    }
    /* we might have had to backtrack */
    gbp->offset += (gbp->size - slack);

    if (gbp->safe_mode == 1) {
        TRACE_COMM(MPI_Bcast)(&err, 1, MPI_INT, 0, gbp->nciop->comm);
        if (err != NC_NOERR) return err;
    }

    /* broadcast root's read to other processes */
    TRACE_COMM(MPI_Bcast)(gbp->base, (int)gbp->size, MPI_BYTE, 0, gbp->nciop->comm);

    return err;
}

/*----< hdr_check_buffer() >--------------------------------------------------*/
/* Ensure that 'nextread' bytes are available.  */
inline static int
hdr_check_buffer(bufferinfo *gbp,
                 MPI_Offset  nextread)
{
    MPI_Aint pos_addr, base_addr;

#ifdef HAVE_MPI_GET_ADDRESS
    MPI_Get_address(gbp->pos,  &pos_addr);
    MPI_Get_address(gbp->base, &base_addr);
#else
    MPI_Address(gbp->pos,  &pos_addr);
    MPI_Address(gbp->base, &base_addr);
#endif
    if (pos_addr + nextread <= base_addr + gbp->size)
        return NC_NOERR;

    return hdr_fetch(gbp);
}

/*----< hdr_get_NCtype() >----------------------------------------------------*/
inline static int
hdr_get_NCtype(bufferinfo *gbp,
               NCtype     *typep)
{
    /* NCtype is 4-byte integer */
    uint type = 0;
    int status = hdr_check_buffer(gbp, 4);
    if (status != NC_NOERR) return status;

    /* get an external unsigned 4-byte integer from the file */
    status = ncmpix_get_uint32((const void**)(&gbp->pos), &type);
    gbp->index += X_SIZEOF_INT;
    if (status != NC_NOERR) return status;

    *typep = (NCtype) type;
    return NC_NOERR;
}

/*----< hdr_get_uint32() >---------------------------------------------------*/
inline static int
hdr_get_uint32(bufferinfo *gbp,
               uint        *xp)
{
    /* in CDF-1 format, all integers are 32-bit
     * in CDF-2 format, only variable begin (starting file offset) is 64-bit
     * in CDF-5 format, both variable's begin and size are 64-bit
     */
    int status = hdr_check_buffer(gbp, 4); /* size of int32 == 4 */
    if (status != NC_NOERR) return status;
    gbp->index += 4;

    status = ncmpix_get_uint32((const void **)(&gbp->pos), xp);
    return status;
}

/*----< hdr_get_uint64() >---------------------------------------------------*/
inline static int
hdr_get_uint64(bufferinfo *gbp,
               uint64     *xp)
{
    /* in CDF-1 format, all integers are 32-bit
     * in CDF-2 format, only variable begin (starting file offset) is 64-bit
     * in CDF-5 format, both variable's begin and size are 64-bit
     */
    int status = hdr_check_buffer(gbp, 8); /* size of int64 == 8 */
    if (status != NC_NOERR) return status;
    gbp->index += 8;

    status = ncmpix_get_uint64((const void **)(&gbp->pos), xp);
    return status;
}

/*----< hdr_get_nc_type() >---------------------------------------------------*/
inline static int
hdr_get_nc_type(bufferinfo *gbp,
                nc_type    *typep)
{
    /* nc_type is 4-byte integer, X_SIZEOF_INT */
    int status;
    uint type;

    status = hdr_check_buffer(gbp, X_SIZEOF_INT);
    if (status != NC_NOERR) return status;

    status = ncmpix_get_uint32((const void**)(&gbp->pos), &type);
    gbp->index += X_SIZEOF_INT;
    if (status != NC_NOERR) return status;

    if (type != NC_BYTE    &&
        type != NC_CHAR    &&
        type != NC_UBYTE   &&
        type != NC_SHORT   &&
        type != NC_USHORT  &&
        type != NC_INT     &&
        type != NC_UINT    &&
        type != NC_FLOAT   &&
        type != NC_DOUBLE  &&
        type != NC_INT64   &&
        type != NC_UINT64
       )
        DEBUG_RETURN_ERROR(NC_EINVAL)

    *typep = (nc_type) type;
    return NC_NOERR;
}

/*----< ncmpix_len_nctype() >------------------------------------------------*/
int
ncmpix_len_nctype(nc_type type) {
    switch(type) {
        case NC_BYTE:
        case NC_CHAR:
        case NC_UBYTE:  return X_SIZEOF_CHAR;
        case NC_SHORT:
        case NC_USHORT: return X_SIZEOF_SHORT;
        case NC_INT:
        case NC_UINT:   return X_SIZEOF_INT;
        case NC_FLOAT:  return X_SIZEOF_FLOAT;
        case NC_DOUBLE: return X_SIZEOF_DOUBLE;
        case NC_INT64:
        case NC_UINT64: return X_SIZEOF_INT64;
        default: fprintf(stderr,"ncmpix_len_nctype bad type %d\n",type);
                 assert(0);
    }
    return 0;
}

/*----< hdr_get_NC_name() >---------------------------------------------------*/
static int
hdr_get_NC_name(bufferinfo  *gbp,
                NC_string  **ncstrpp)
{
    /* netCDF file format:
     *  ...
     * name       = nelems  namestring
     * nelems     = NON_NEG
     * namestring = ID1 [IDN ...] padding
     * ID1        = alphanumeric | '_'
     * IDN        = alphanumeric | special1 | special2
     * padding    = <0, 1, 2, or 3 bytes to next 4-byte boundary>
     * NON_NEG    = <non-negative INT> |  // CDF-1 and CDF-2
     *              <non-negative INT64>  // CDF-5
     */
    int status;
    MPI_Offset  nchars, nbytes, padding, bufremain, strcount;
    NC_string *ncstrp;
    char *cpos, pad[X_ALIGN-1];
    MPI_Aint pos_addr, base_addr;

    /* get nelems */
    if (gbp->version == 5) {
        uint64 tmp;
        status = hdr_get_uint64(gbp, &tmp);
        nchars = (MPI_Offset)tmp;
    }
    else {
        uint tmp;
        status = hdr_get_uint32(gbp, &tmp);
        nchars = (MPI_Offset)tmp;
    }
    if (status != NC_NOERR) return status;

    /* Allocate a NC_string structure large enough to hold nchars characters */
    ncstrp = ncmpii_new_NC_string((size_t)nchars, NULL);
    if (ncstrp == NULL) DEBUG_RETURN_ERROR(NC_ENOMEM)

    nbytes = nchars * X_SIZEOF_CHAR;
    padding = _RNDUP(X_SIZEOF_CHAR * ncstrp->nchars, X_ALIGN)
            - X_SIZEOF_CHAR * ncstrp->nchars;
#ifdef HAVE_MPI_GET_ADDRESS
    MPI_Get_address(gbp->pos,  &pos_addr);
    MPI_Get_address(gbp->base, &base_addr);
#else
    MPI_Address(gbp->pos,  &pos_addr);
    MPI_Address(gbp->base, &base_addr);
#endif
    bufremain = gbp->size - (pos_addr - base_addr);
    cpos = ncstrp->cp;

    /* get namestring with padding */
    while (nbytes > 0) {
        if (bufremain > 0) {
            strcount = MIN(bufremain, nbytes);
            if (strcount != (size_t)strcount) DEBUG_RETURN_ERROR(NC_EINTOVERFLOW)
            memcpy(cpos, gbp->pos, (size_t)strcount);
            nbytes -= strcount;
            gbp->pos = (void *)((char *)gbp->pos + strcount);
            gbp->index += strcount;
            cpos += strcount;
            bufremain -= strcount;
        } else {
            status = hdr_fetch(gbp);
            if (status != NC_NOERR) {
                ncmpii_free_NC_string(ncstrp);
                return status;
            }
            bufremain = gbp->size;
        }
    }

    /* handle the padding */
    if (padding > 0) {
        memset(pad, 0, X_ALIGN-1);
        if (memcmp(gbp->pos, pad, (size_t)padding) != 0) {
            ncmpii_free_NC_string(ncstrp);
            DEBUG_RETURN_ERROR(NC_EINVAL)
        }
        gbp->pos = (void *)((char *)gbp->pos + padding);
        gbp->index += padding;
    }

    *ncstrpp = ncstrp;

    return NC_NOERR;
}

/*----< hdr_get_NC_dim() >----------------------------------------------------*/
inline static int
hdr_get_NC_dim(bufferinfo  *gbp,
               NC_dim     **dimpp)
{
    /* netCDF file format:
     *  ...
     * dim        = name  dim_length
     * dim_length = NON_NEG
     * NON_NEG    = <non-negative INT> |  // CDF-1 and CDF-2
     *              <non-negative INT64>  // CDF-5
     */
    int status;
    NC_string *ncstrp;
    NC_dim *dimp;

    /* get name */
    status = hdr_get_NC_name(gbp, &ncstrp);
    if (status != NC_NOERR) return status;

    dimp = ncmpii_new_x_NC_dim(ncstrp);
    if (dimp == NULL) DEBUG_RETURN_ERROR(NC_ENOMEM)

    /* get dim_length */
    if (gbp->version == 5) {
        uint64 tmp;
        status = hdr_get_uint64(gbp, &tmp);
        dimp->size = (MPI_Offset)tmp;
    }
    else {
        uint tmp;
        status = hdr_get_uint32(gbp, &tmp);
        dimp->size = (MPI_Offset)tmp;
    }
    if (status != NC_NOERR) {
        ncmpii_free_NC_dim(dimp); /* frees name */
        return status;
    }

    *dimpp = dimp;
    return NC_NOERR;
}

/*----< hdr_get_NC_dimarray() >-----------------------------------------------*/
static int
hdr_get_NC_dimarray(bufferinfo  *gbp,
                    NC_dimarray *ncap)
{
    /* netCDF file format:
     *  ...
     * dim_list     = ABSENT | NC_DIMENSION  nelems  [dim ...]
     * ABSENT       = ZERO  ZERO |  // list is not present for CDF-1 and 2
     *                ZERO  ZERO64  // for CDF-5
     * ZERO         = \x00 \x00 \x00 \x00                      // 32-bit zero
     * ZERO64       = \x00 \x00 \x00 \x00 \x00 \x00 \x00 \x00  // 64-bit zero
     * NC_DIMENSION = \x00 \x00 \x00 \x0A         // tag for list of dimensions
     * nelems       = NON_NEG       // number of elements in following sequence
     * NON_NEG      = <non-negative INT> |        // CDF-1 and CDF-2
     *                <non-negative INT64>        // CDF-5
     */
    int i, status;
    NCtype type = NC_UNSPECIFIED;
    MPI_Offset ndefined;

    assert(gbp != NULL && gbp->pos != NULL);
    assert(ncap != NULL);
    assert(ncap->value == NULL);

    /* get NCtype (NC_DIMENSION) */
    status = hdr_get_NCtype(gbp, &type);
    if (status != NC_NOERR) return status;

    /* get nelems */
    if (gbp->version == 5) {
        uint64 tmp;
        status = hdr_get_uint64(gbp, &tmp);
        ndefined = (MPI_Offset)tmp;
    }
    else {
        uint tmp;
        status = hdr_get_uint32(gbp, &tmp);
        ndefined = (MPI_Offset)tmp;
    }
    if (status != NC_NOERR) return status;
    if (ndefined != (int)ndefined) DEBUG_RETURN_ERROR(NC_EINTOVERFLOW)
    ncap->ndefined = (int)ndefined;
    /* TODO: we should allow ndefined > 2^32, considering change the data type
     * of ndefined from int to MPI_Offset */

    if (ndefined == 0) {
        if (type != NC_DIMENSION && type != NC_UNSPECIFIED)
            DEBUG_RETURN_ERROR(NC_EINVAL)
    } else {
        if (type != NC_DIMENSION) DEBUG_RETURN_ERROR(NC_EINVAL)

        ncap->value = (NC_dim **) NCI_Malloc((size_t)ndefined * sizeof(NC_dim*));
        if (ncap->value == NULL) DEBUG_RETURN_ERROR(NC_ENOMEM)
        ncap->nalloc = (int)ndefined;
        /* TODO: we should allow nalloc > 2^32, considering change the data
         * type of nalloc from int to MPI_Offset */

        for (i=0; i<ndefined; i++) {
            status = hdr_get_NC_dim(gbp, ncap->value + i);
            if (status != NC_NOERR) { /* error: fail to get the next dim */
                ncap->ndefined = i;
                ncmpii_free_NC_dimarray(ncap);
                return status;
            }
        }
    }

    return NC_NOERR;
}

/*----< hdr_get_NC_attrV() >--------------------------------------------------*/
static int
hdr_get_NC_attrV(bufferinfo *gbp,
                 NC_attr    *attrp)
{
    /* netCDF file format:
     *  ...
     * attr    = name  nc_type  nelems  [values ...]
     *  ...
     * values  = bytes | chars | shorts | ints | floats | doubles
     * bytes   = [BYTE ...]  padding
     * chars   = [CHAR ...]  padding
     * shorts  = [SHORT ...]  padding
     * ints    = [INT ...]
     * floats  = [FLOAT ...]
     * doubles = [DOUBLE ...]
     * padding = <0, 1, 2, or 3 bytes to next 4-byte boundary>
     */
    int status;
    void *value = attrp->xvalue;
    char pad[X_ALIGN-1];
    MPI_Offset nbytes, padding, bufremain, attcount;
    MPI_Aint pos_addr, base_addr;

    nbytes = attrp->nelems * ncmpix_len_nctype(attrp->type);
    padding = attrp->xsz - nbytes;
#ifdef HAVE_MPI_GET_ADDRESS
    MPI_Get_address(gbp->pos,  &pos_addr);
    MPI_Get_address(gbp->base, &base_addr);
#else
    MPI_Address(gbp->pos,  &pos_addr);
    MPI_Address(gbp->base, &base_addr);
#endif
    bufremain = gbp->size - (pos_addr - base_addr);

    /* get values */
    while (nbytes > 0) {
        if (bufremain > 0) {
            attcount = MIN(bufremain, nbytes);
            if (attcount != (size_t)attcount) DEBUG_RETURN_ERROR(NC_EINTOVERFLOW)
            memcpy(value, gbp->pos, (size_t)attcount);
            nbytes -= attcount;
            gbp->pos = (void *)((char *)gbp->pos + attcount);
            gbp->index += attcount;
            value = (void *)((char *)value + attcount);
            bufremain -= attcount;
        } else {
            status = hdr_fetch(gbp);
            if (status != NC_NOERR) return status;
            bufremain = gbp->size;
        }
    }

    /* handle the padding */
    if (padding > 0) {
        memset(pad, 0, X_ALIGN-1);
        if (memcmp(gbp->pos, pad, (size_t)padding) != 0) DEBUG_RETURN_ERROR(NC_EINVAL)
        gbp->pos = (void *)((char *)gbp->pos + padding);
        gbp->index += padding;
    }

    return NC_NOERR;
}

/*----< hdr_get_NC_attr() >---------------------------------------------------*/
static int
hdr_get_NC_attr(bufferinfo  *gbp,
                NC_attr    **attrpp)
{
    /* netCDF file format:
     *  ...
     * attr    = name  nc_type  nelems  [values ...]
     * nc_type = NC_BYTE | NC_CHAR | NC_SHORT | ...
     * nelems  = NON_NEG       // number of elements in following sequence
     * NON_NEG = <non-negative INT> |  // CDF-1 and CDF-2
     *           <non-negative INT64>  // CDF-5
     */
    int status;
    NC_string *strp;
    nc_type type;
    MPI_Offset nelems;
    NC_attr *attrp;

    /* get name */
    status = hdr_get_NC_name(gbp, &strp);
    if (status != NC_NOERR) return status;

    /* get nc_type */
    status = hdr_get_nc_type(gbp, &type);
    if (status != NC_NOERR) {
        ncmpii_free_NC_string(strp);
        return status;
    }

    /* get nelems */
    if (gbp->version == 5) {
        uint64 tmp;
        status = hdr_get_uint64(gbp, &tmp);
        nelems = (MPI_Offset)tmp;
    }
    else {
        uint tmp;
        status = hdr_get_uint32(gbp, &tmp);
        nelems = (MPI_Offset)tmp;
    }
    if (status != NC_NOERR) {
        ncmpii_free_NC_string(strp);
        return status;
    }

    /* allocate space for attribute object */
    attrp = ncmpii_new_x_NC_attr(strp, type, nelems);
    if (attrp == NULL) {
        ncmpii_free_NC_string(strp);
        return status;
    }

    /* get [values ...] */
    status = hdr_get_NC_attrV(gbp, attrp);
    if (status != NC_NOERR) {
        ncmpii_free_NC_attr(attrp); /* frees strp */
        return status;
    }

    *attrpp = attrp;
    return NC_NOERR;
}

/*----< hdr_get_NC_attrarray() >----------------------------------------------*/
static int
hdr_get_NC_attrarray(bufferinfo   *gbp,
                     NC_attrarray *ncap)
{
    /* netCDF file format:
     *  ...
     * att_list     = ABSENT | NC_ATTRIBUTE  nelems  [attr ...]
     * ABSENT       = ZERO  ZERO |  // list is not present for CDF-1 and 2
     *                ZERO  ZERO64  // for CDF-5
     * ZERO         = \x00 \x00 \x00 \x00                      // 32-bit zero
     * ZERO64       = \x00 \x00 \x00 \x00 \x00 \x00 \x00 \x00  // 64-bit zero
     * NC_ATTRIBUTE = \x00 \x00 \x00 \x0C         // tag for list of attributes
     * nelems       = NON_NEG       // number of elements in following sequence
     * NON_NEG      = <non-negative INT> |        // CDF-1 and CDF-2
     *                <non-negative INT64>        // CDF-5
     */
    int i, status;
    NCtype type = NC_UNSPECIFIED;
    MPI_Offset ndefined;

    assert(gbp != NULL && gbp->pos != NULL);
    assert(ncap != NULL);
    assert(ncap->value == NULL);

    /* get NCtype (NC_ATTRIBUTE) */
    status = hdr_get_NCtype(gbp, &type);
    if (status != NC_NOERR) return status;

    /* get nelems */
    if (gbp->version == 5) {
        uint64 tmp;
        status = hdr_get_uint64(gbp, &tmp);
        ndefined = (MPI_Offset)tmp;
    }
    else {
        uint tmp;
        status = hdr_get_uint32(gbp, &tmp);
        ndefined = (MPI_Offset)tmp;
    }
    if (status != NC_NOERR) return status;
    if (ndefined != (int)ndefined) DEBUG_RETURN_ERROR(NC_EINTOVERFLOW)
    ncap->ndefined = (int)ndefined;

    if (ndefined == 0) {
        if (type != NC_ATTRIBUTE && type != NC_UNSPECIFIED)
            DEBUG_RETURN_ERROR(NC_EINVAL)
    } else {
        if (type != NC_ATTRIBUTE) DEBUG_RETURN_ERROR(NC_EINVAL)

        ncap->value = (NC_attr **)NCI_Malloc((size_t)ndefined * sizeof(NC_attr*));
        if (ncap->value == NULL) DEBUG_RETURN_ERROR(NC_ENOMEM)
        ncap->nalloc = (int)ndefined;

        /* get [attr ...] */
        for (i=0; i<ndefined; i++) {
            status = hdr_get_NC_attr(gbp, ncap->value + i);
            if (status != NC_NOERR) { /* Error: fail to get the next att */
                ncap->ndefined = i;
                ncmpii_free_NC_attrarray(ncap);
                return status;
            }
        }
    }

    return NC_NOERR;
}

/*----< hdr_get_NC_var() >---------------------------------------------------*/
static int
hdr_get_NC_var(bufferinfo  *gbp,
               NC_var     **varpp)
{
    /* netCDF file format:
     * netcdf_file = header data
     * header      = magic numrecs dim_list gatt_list var_list
     *  ...
     * var         = name nelems [dimid ...] vatt_list nc_type vsize begin
     * nelems      = NON_NEG
     * dimid       = NON_NEG
     * vatt_list   = att_list
     * nc_type     = NC_BYTE | NC_CHAR | NC_SHORT | ...
     * vsize       = NON_NEG
     * begin       = OFFSET        // Variable start location.
     * OFFSET      = <non-negative INT> |  // CDF-1
     *               <non-negative INT64>  // CDF-2 and CDF-5
     * NON_NEG     = <non-negative INT> |  // CDF-1 and CDF-2
     *               <non-negative INT64>  // CDF-5
     */
    int status;
    NC_string *strp;
    MPI_Offset ndims=0, dim;
    MPI_Offset tmp_dimids=0;
    NC_var *varp;

    /* get name */
    status = hdr_get_NC_name(gbp, &strp);
    if (status != NC_NOERR) return status;

    /* nelems */
    if (gbp->version == 5) {
        uint64 tmp;
        status = hdr_get_uint64(gbp, &tmp);
        ndims = (MPI_Offset)tmp;
    }
    else {
        uint tmp;
        status = hdr_get_uint32(gbp, &tmp);
        ndims = (MPI_Offset)tmp;
    }
    if (status != NC_NOERR) {
         ncmpii_free_NC_string(strp);
         return status;
    }
    if (ndims != (int)ndims) DEBUG_RETURN_ERROR(NC_EINTOVERFLOW)

    /* allocate space for var object */
    varp = ncmpii_new_x_NC_var(strp, (int)ndims);
    if (varp == NULL) {
        ncmpii_free_NC_string(strp);
        DEBUG_RETURN_ERROR(NC_ENOMEM)
    }

    /* get [dimid ...] */
    for (dim=0; dim<ndims; dim++) {
        status = hdr_check_buffer(gbp, (gbp->version == 5 ? 8 : 4));
        if (status != NC_NOERR) {
            ncmpii_free_NC_var(varp);
            return status;
        }
        if (gbp->version == 5) {
            uint64 tmp;
            status = hdr_get_uint64(gbp, &tmp);
            tmp_dimids = (MPI_Offset)tmp;
        }
        else {
            uint tmp;
            status = hdr_get_uint32(gbp, &tmp);
            tmp_dimids = (MPI_Offset)tmp;
        }
        /* TODO: consider change the data type of dimids from int to
         * MPI_Offset */
        varp->dimids[dim] = (int)tmp_dimids;
        if (status != NC_NOERR) {
           return status;
        }
    }

    /* get vatt_list */
    status = hdr_get_NC_attrarray(gbp, &varp->attrs);
    if (status != NC_NOERR) {
        ncmpii_free_NC_var(varp);
        return status;
    }

    /* get nc_type */
    status = hdr_get_nc_type(gbp, &varp->type);
    if (status != NC_NOERR) {
        ncmpii_free_NC_var(varp);
        return status;
    }

    /* get vsize */
    if (gbp->version == 5) {
        uint64 tmp;
        status = hdr_get_uint64(gbp, &tmp);
        varp->len = (MPI_Offset)tmp;
    }
    else {
        uint tmp;
        status = hdr_get_uint32(gbp, &tmp);
        varp->len = (MPI_Offset)tmp;
    }
    if (status != NC_NOERR) {
        ncmpii_free_NC_var(varp);
        return status;
    }
    /* As described in CDF-2 format specification, vsize is redundant.
       Its value may be computed from the product of dimension lengths.
       In CDF-2, vsize is a 4-byte integer. So, if we define a variable of
       less than 2^32 elements but size > 2^32-4 bytes, then vsize in CDF-2
       will overflow. Recompute varp->len can ignore an overflowed value in
       vsize stored in the file and hence bypass the limitation of CDF-2 on
       variable size of 2^32-4 bytes.

       Later on, back to ncmpii_hdr_get_NC(), ncmpii_NC_computeshapes() is
       called which recomputes varp->len using the dimension values and hence
       overwrites the value read from file above.

       In summary, PnetCDF now ignores the value of vsize stored in the file
       header.
     */

    /* next element is 'begin' */
    status = hdr_check_buffer(gbp, (gbp->version == 1 ? 4 : 8));
    if (status != NC_NOERR) {
        ncmpii_free_NC_var(varp);
        return status;
    }

    /* get begin */
    if (gbp->version == 1) {
        uint tmp=0;
        status = ncmpix_get_uint32((const void **)(&gbp->pos), &tmp);
        varp->begin = (MPI_Offset)tmp;
    }
    else {
        uint64 tmp=0;
        status = ncmpix_get_uint64((const void **)(&gbp->pos), &tmp);
        varp->begin = (MPI_Offset)tmp;
    }
    if (status != NC_NOERR) {
        ncmpii_free_NC_var(varp);
        return status;
    }

    *varpp = varp;
    return NC_NOERR;
}

/*----< hdr_get_NC_vararray() >----------------------------------------------*/
static int
hdr_get_NC_vararray(bufferinfo  *gbp,
                    NC_vararray *ncap)
{
    /* netCDF file format:
     * netcdf_file = header  data
     * header      = magic  numrecs  dim_list  gatt_list  var_list
     *  ...
     * var_list    = ABSENT | NC_VARIABLE   nelems  [var ...]
     * ABSENT      = ZERO  ZERO |  // list is not present for CDF-1 and 2
     *               ZERO  ZERO64  // for CDF-5
     * ZERO        = \x00 \x00 \x00 \x00                      // 32-bit zero
     * ZERO64      = \x00 \x00 \x00 \x00 \x00 \x00 \x00 \x00  // 64-bit zero
     * NC_VARIABLE = \x00 \x00 \x00 \x0B         // tag for list of variables
     * nelems      = NON_NEG       // number of elements in following sequence
     * NON_NEG     = <non-negative INT> |        // CDF-1 and CDF-2
     *               <non-negative INT64>        // CDF-5
     */
    int i, status;
    NCtype type = NC_UNSPECIFIED;
    MPI_Offset ndefined;

    assert(gbp != NULL && gbp->pos != NULL);
    assert(ncap != NULL);
    assert(ncap->value == NULL);

    /* get NCtype (NC_VARIABLE) from gbp buffer */
    status = hdr_get_NCtype(gbp, &type);
    if (status != NC_NOERR) return status;

    /* get nelems (number of variables) from gbp buffer */
    if (gbp->version == 5) {
        uint64 tmp;
        status = hdr_get_uint64(gbp, &tmp);
        ndefined = (MPI_Offset)tmp;
    }
    else {
        uint tmp;
        status = hdr_get_uint32(gbp, &tmp);
        ndefined = (MPI_Offset)tmp;
    }
    if (status != NC_NOERR) return status;
    if (ndefined != (int)ndefined) DEBUG_RETURN_ERROR(NC_EINTOVERFLOW)
    ncap->ndefined = (int)ndefined;
    /* TODO: we should allow ndefined > 2^32, considering change the data type
     * of ndefined from int to MPI_Offset */

    if (ndefined == 0) { /* no variable defined */
        if (type != NC_VARIABLE && type != NC_UNSPECIFIED)
            DEBUG_RETURN_ERROR(NC_EINVAL)
    } else {
        if (type != NC_VARIABLE) DEBUG_RETURN_ERROR(NC_EINVAL)

        ncap->value = (NC_var **) NCI_Malloc((size_t)ndefined * sizeof(NC_var*));
        if (ncap->value == NULL) DEBUG_RETURN_ERROR(NC_ENOMEM)
        ncap->nalloc = (int)ndefined;

        /* get [var ...] */
        for (i=0; i<ndefined; i++) {
            status = hdr_get_NC_var(gbp, ncap->value + i);
            if (status != NC_NOERR) { /* Error: fail to get the next var */
                ncap->ndefined = i;
                ncmpii_free_NC_vararray(ncap);
                return status;
            }
        }
    }

    return NC_NOERR;
}

/*----< ncmpii_hdr_get_NC() >------------------------------------------------*/
/*  CDF format specification
 *      netcdf_file  = header  data
 *      header       = magic  numrecs  dim_list  gatt_list  var_list
 *      magic        = 'C'  'D'  'F'  VERSION
 *      VERSION      = \x01 |                      // classic format
 *                     \x02 |                      // 64-bit offset format
 *                     \x05                        // 64-bit data format
 *      numrecs      = NON_NEG | STREAMING         // length of record dimension
 *      dim_list     = ABSENT | NC_DIMENSION  nelems  [dim ...]
 *      gatt_list    = att_list                    // global attributes
 *      att_list     = ABSENT | NC_ATTRIBUTE  nelems  [attr ...]
 *      var_list     = ABSENT | NC_VARIABLE   nelems  [var ...]
 */
int
ncmpii_hdr_get_NC(NC *ncp)
{
    int status;
    bufferinfo getbuf;
    schar magic[sizeof(ncmagic1)];
    MPI_Offset nrecs = 0;
    MPI_Aint pos_addr, base_addr;

    assert(ncp != NULL);

    /* Initialize the get buffer that stores the header read from the file */
    getbuf.nciop     = ncp->nciop;
    getbuf.offset    = 0;   /* read from start of the file */
    getbuf.put_size  = 0;   /* amount of writes so far in bytes */
    getbuf.get_size  = 0;   /* amount of reads  so far in bytes */
    getbuf.safe_mode = ncp->safe_mode;

    /* CDF-5's minimum header size is 4 bytes more than CDF-1 and CDF-2's */
    getbuf.size = _RNDUP( MAX(MIN_NC_XSZ+4, ncp->chunk), X_ALIGN );
    if (getbuf.size > NC_DEFAULT_CHUNKSIZE)
        getbuf.size = NC_DEFAULT_CHUNKSIZE;

    if (getbuf.size != (size_t)getbuf.size) DEBUG_RETURN_ERROR(NC_EINTOVERFLOW)
    getbuf.pos = getbuf.base = (void *)NCI_Malloc((size_t)getbuf.size);
    getbuf.index = 0;

    /* Fetch the next header chunk. The chunk is 'gbp->size' bytes big */
    status = hdr_fetch(&getbuf);
    if (status != NC_NOERR) return status;

    /* processing the header from getbuf, the get buffer */

    /* First get the file format information, magic */
    memset(magic, 0, sizeof(magic));
    status = ncmpix_getn_schar_schar((const void **)(&getbuf.pos),
                                     sizeof(magic), magic);
    if (status != NC_NOERR) return status;
    getbuf.index += (MPI_Offset)sizeof(magic);

    /* don't need to worry about CDF-1 or CDF-2
     * if the first bits are not 'CDF'
     */
    if (memcmp(magic, ncmagic1, sizeof(ncmagic1)-1) != 0) {
        NCI_Free(getbuf.base);
        DEBUG_RETURN_ERROR(NC_ENOTNC)
    }

    /* check version number in last byte of magic */
    if (magic[sizeof(ncmagic1)-1] == 0x1) {
        getbuf.version = 1;
        fSet(ncp->flags, NC_32BIT);
    } else if (magic[sizeof(ncmagic1)-1] == 0x2) {
        getbuf.version = 2;
        fSet(ncp->flags, NC_64BIT_OFFSET);
#if SIZEOF_MPI_OFFSET < 8
            /* take the easy way out: if we can't support all CDF-2
             * files, return immediately */
            NCI_Free(getbuf.base);
            DEBUG_RETURN_ERROR(NC_ESMALL)
#endif
    } else if (magic[sizeof(ncmagic1)-1] == 0x5) {
        getbuf.version = 5;
        fSet(ncp->flags, NC_64BIT_DATA);
#if SIZEOF_MPI_OFFSET < 8
            NCI_Free(getbuf.base);
            DEBUG_RETURN_ERROR(NC_ESMALL)
#endif
    } else {
        NCI_Free(getbuf.base);
        DEBUG_RETURN_ERROR(NC_ENOTNC) /* not a netCDF file */
    }

    /** Ensure that 'nextread' bytes (numrecs) are available. */
    status = hdr_check_buffer(&getbuf, (getbuf.version == 5) ? 8 : 4);
    if(status != NC_NOERR) {
        NCI_Free(getbuf.base);
        return status;
    }

    /* get numrecs from getbuf into ncp */
    if (getbuf.version == 5) {
        uint64 tmp=0;
        status = ncmpix_get_uint64((const void **)(&getbuf.pos), &tmp);
        nrecs = (MPI_Offset)tmp;
    }
    else {
        uint tmp=0;
        status = ncmpix_get_uint32((const void **)(&getbuf.pos), &tmp);
        nrecs = (MPI_Offset)tmp;
    }
    if (status != NC_NOERR) {
        NCI_Free(getbuf.base);
        return status;
    }

    if (getbuf.version == 5)
        getbuf.index += X_SIZEOF_INT64;
    else
        getbuf.index += X_SIZEOF_SIZE_T;

    ncp->numrecs = nrecs;

#ifdef HAVE_MPI_GET_ADDRESS
    MPI_Get_address(getbuf.pos,  &pos_addr);
    MPI_Get_address(getbuf.base, &base_addr);
#else
    MPI_Address(getbuf.pos,  &pos_addr);
    MPI_Address(getbuf.base, &base_addr);
#endif
    assert(pos_addr < base_addr + getbuf.size);

    /* get dim_list from getbuf into ncp */
    status = hdr_get_NC_dimarray(&getbuf, &ncp->dims);
    if (status != NC_NOERR) {
        NCI_Free(getbuf.base);
        return status;
    }

    /* get gatt_list from getbuf into ncp */
    status = hdr_get_NC_attrarray(&getbuf, &ncp->attrs);
    if (status != NC_NOERR) {
        NCI_Free(getbuf.base);
        return status;
    }

    /* get var_list from getbuf into ncp */
    status = hdr_get_NC_vararray(&getbuf, &ncp->vars);
    if (status != NC_NOERR) {
        NCI_Free(getbuf.base);
        return status;
    }

    /* get the un-aligned size occupied by the file header */
    ncp->xsz = ncmpii_hdr_len_NC(ncp);

    /* Recompute the shapes of all variables
     * Sets ncp->begin_var to start of first variable.
     * Sets ncp->begin_rec to start of first record variable.
     */
    status = ncmpii_NC_computeshapes(ncp);

    NCI_Free(getbuf.base);

    ncp->nciop->put_size += getbuf.put_size;
    ncp->nciop->get_size += getbuf.get_size;

    return status;
}

/* End Of get NC */

#define WARN_STR "Warning (inconsistent metadata):"

/*----< ncmpii_comp_dims() >--------------------------------------------------*/
/* compare the local copy of dim_list against root's
 * If inconsistency is detected, overwrite local's with root's
 * this function is collective.
 */
static int
ncmpii_comp_dims(int          safe_mode,
                 NC_dimarray *root_dim,
                 NC_dimarray *local_dim)
{
    int i, err, status=NC_NOERR;

    if (root_dim->ndefined != local_dim->ndefined) {
        if (safe_mode)
            printf("%s number of dimensions (local=%d, root=%d)\n",
                   WARN_STR, local_dim->ndefined, root_dim->ndefined);
        DEBUG_ASSIGN_ERROR(status, NC_EMULTIDEFINE_DIM_NUM)
    }

    for (i=0; i<root_dim->ndefined; i++) {

        if (i >= local_dim->ndefined) { /* if local list is shorter */
            /* copy root's dim to local */
            NC_dim *new_dim = dup_NC_dim(root_dim->value[i]);
            err = incr_NC_dimarray(local_dim, new_dim);
            if (err != NC_NOERR) return err; /* this is a fatal error */
            continue;
        }

        /* check dimension name */
        NC_string *root_name, *local_name;
         root_name =  root_dim->value[i]->name;
        local_name = local_dim->value[i]->name;

        err = NC_NOERR;
        if (root_name->nchars != local_name->nchars) {
            if (safe_mode)
                printf("%s dimension name length (local=%lld, root=%lld)\n",
                       WARN_STR, local_name->nchars, root_name->nchars);
            DEBUG_ASSIGN_ERROR(err, NC_EMULTIDEFINE_DIM_NAME)
        }
        else if (memcmp(root_name->cp, local_name->cp, (size_t)root_name->nchars) != 0) {
            if (safe_mode)
                printf("%s dimension name (local=%s, root=%s)\n",
                       WARN_STR, local_name->cp, root_name->cp);
            DEBUG_ASSIGN_ERROR(err, NC_EMULTIDEFINE_DIM_NAME)
        }
        else if (root_dim->value[i]->size != local_dim->value[i]->size) {
            /* check dimension size */
            if (safe_mode)
                printf("%s dimension %s's size (local=%lld, root=%lld)\n",
                       WARN_STR, root_dim->value[i]->name->cp,
                       root_dim->value[i]->size, local_dim->value[i]->size);
            DEBUG_ASSIGN_ERROR(err, NC_EMULTIDEFINE_DIM_SIZE)
        }
        if (status == NC_NOERR) status = err;

        /* overwrite local's dim with root's */
        if (err != NC_NOERR) {
            ncmpii_free_NC_dim(local_dim->value[i]);
            local_dim->value[i] = dup_NC_dim(root_dim->value[i]);
        }
    }

    /* delete extra dimensions defined only in local copy */
    for (; i<local_dim->ndefined; i++)
        ncmpii_free_NC_dim(local_dim->value[i]);

    local_dim->ndefined = root_dim->ndefined;

    return status;
}

/*----< ncmpii_comp_attrs() >-------------------------------------------------*/
/* compare the local copy of attr_list against root's
 * If inconsistency is detected, overwrite local's with root's
 */
static int
ncmpii_comp_attrs(int           safe_mode,
                  NC_attrarray *root_attr,
                  NC_attrarray *local_attr)
{
    int i, j, err, status=NC_NOERR;
    char *msg;

    /* check if the numbers of attributes are the same */
    if (root_attr->ndefined != local_attr->ndefined) {
        if (safe_mode)
            printf("%s number of attributes (root=%d, local=%d)\n",
                   WARN_STR, root_attr->ndefined, local_attr->ndefined);
        DEBUG_ASSIGN_ERROR(status, NC_EMULTIDEFINE_ATTR_NUM)
    }

    for (i=0; i<root_attr->ndefined; i++) {

        if (i >= local_attr->ndefined) { /* if local list is shorter */
            /* copy root's attr to local */
            NC_attr *new_attr = dup_NC_attr(root_attr->value[i]);
            err = incr_NC_attrarray(local_attr, new_attr);
            if (err != NC_NOERR) return err; /* a fatal error */
            continue;
        }

        NC_attr *v1 = root_attr->value[i];
        NC_attr *v2 = local_attr->value[i];
        char *name = v1->name->cp;

#define ATTR_WARN(msg, attr, root, local) \
    if (safe_mode) printf(msg, WARN_STR, attr, root, local);

#define ATTR_WARN_J(msg, attr, j, root, local) \
    if (safe_mode) printf(msg, WARN_STR, attr, j, root, local);

        err = NC_NOERR;
        if (v1->name->nchars != v2->name->nchars ||
            memcmp(name, v2->name->cp, (size_t)v1->name->nchars) != 0) {
            msg ="%s attribute %s (root=%s, local=%s)\n";
            ATTR_WARN(msg, "name", name, v2->name->cp)
            DEBUG_ASSIGN_ERROR(err, NC_EMULTIDEFINE_ATTR_NAME)
        }
        else if (v1->type != v2->type) {
            msg = "%s attribute \"%s\" type (root=%d, local=%d)\n";
            ATTR_WARN(msg, name, v1->type, v2->type)
            DEBUG_ASSIGN_ERROR(err, NC_EMULTIDEFINE_ATTR_TYPE)
        }
        else if (v1->nelems != v2->nelems) {
            msg = "%s attribute \"%s\" length (root=%lld, local=%lld)\n";
            ATTR_WARN(msg, name, lld(v1->nelems), lld(v2->nelems))
            DEBUG_ASSIGN_ERROR(err, NC_EMULTIDEFINE_ATTR_LEN)
        }
        else if (v1->xsz != v2->xsz) { /* internal check */
            msg = "%s attribute \"%s\" size (root=%lld, local=%lld)\n";
            ATTR_WARN(msg, name, lld(v1->xsz), lld(v2->xsz))
            DEBUG_ASSIGN_ERROR(err, NC_EMULTIDEFINE_ATTR_SIZE)
        }
        /* hereinafter, we have v1->nelems == v2->nelems */
        else if (v1->type == NC_CHAR) {
            if (memcmp(v1->xvalue, v2->xvalue, (size_t)v1->nelems)) {
                msg = "%s attribute \"%s\" CHAR (root=%s, local=%s)\n";
                ATTR_WARN(msg, name, (char*)v1->xvalue, (char*)v2->xvalue);
                DEBUG_ASSIGN_ERROR(err, NC_EMULTIDEFINE_ATTR_VAL)
            }
        }
        else if (v1->type == NC_BYTE) {
            schar *sba = (schar*) v1->xvalue;
            schar *sbb = (schar*) v2->xvalue;
            for (j=0; j<v1->nelems; j++) {
                if (sba[j] != sbb[j]) {
                    msg = "%s attribute \"%s\"[%d] BYTE (root=%hhdb, local=%hhdb)\n";
                    ATTR_WARN_J(msg, name, j, sba[j], sbb[j])
                    DEBUG_ASSIGN_ERROR(err, NC_EMULTIDEFINE_ATTR_VAL)
                    break;
                }
            }
        }
        else if (v1->type == NC_UBYTE) {
            uchar *uba = (uchar*) v1->xvalue;
            uchar *ubb = (uchar*) v2->xvalue;
            for (j=0; j<v1->nelems; j++) {
                if (uba[j] != ubb[j]) {
                    msg = "%s attribute \"%s\"[%d] UBYTE (root=%hhuub, local=%hhuub)\n";
                    ATTR_WARN_J(msg, name, j, uba[j], ubb[j])
                    DEBUG_ASSIGN_ERROR(err, NC_EMULTIDEFINE_ATTR_VAL)
                    break;
                }
            }
        }
        else if (v1->type == NC_SHORT) {
            short *ssa = (short*) v1->xvalue;
            short *ssb = (short*) v2->xvalue;
            for (j=0; j<v1->nelems; j++) {
                if (ssa[j] != ssb[j]) {
                    msg = "%s attribute \"%s\"[%d] SHORT (root=%hds, local=%hds)\n";
                    ATTR_WARN_J(msg, name, j, ssa[j], ssb[j])
                    DEBUG_ASSIGN_ERROR(err, NC_EMULTIDEFINE_ATTR_VAL)
                    break;
                }
            }
        }
        else if (v1->type == NC_USHORT) {
            ushort *usa = (ushort*) v1->xvalue;
            ushort *usb = (ushort*) v2->xvalue;
            for (j=0; j<v1->nelems; j++) {
                if (usa[j] != usb[j]) {
                    msg = "%s attribute \"%s\"[%d] USHORT (root=%huus, local=%huus)\n";
                    ATTR_WARN_J(msg, name, j, usa[j], usb[j])
                    DEBUG_ASSIGN_ERROR(err, NC_EMULTIDEFINE_ATTR_VAL)
                    break;
                }
            }
        }
        else if (v1->type == NC_INT) {
            int *sia = (int*) v1->xvalue;
            int *sib = (int*) v2->xvalue;
            for (j=0; j<v1->nelems; j++) {
                if (sia[j] != sib[j]) {
                    msg = "%s attribute \"%s\"[%d] INT (root=%d, local=%d)\n";
                    ATTR_WARN_J(msg, name, j, sia[j], sib[j])
                    DEBUG_ASSIGN_ERROR(err, NC_EMULTIDEFINE_ATTR_VAL)
                    break;
                }
            }
        }
        else if (v1->type == NC_UINT) {
            uint *uia = (uint*) v1->xvalue;
            uint *uib = (uint*) v2->xvalue;
            for (j=0; j<v1->nelems; j++) {
                if (uia[j] != uib[j]) {
                    msg = "%s attribute \"%s\"[%d] UINT (root=%uu, local=%uu)\n";
                    ATTR_WARN_J(msg, name, j, uia[j], uib[j])
                    DEBUG_ASSIGN_ERROR(err, NC_EMULTIDEFINE_ATTR_VAL)
                    break;
                }
            }
        }
        else if (v1->type == NC_FLOAT) {
            float *fa = (float*) v1->xvalue;
            float *fb = (float*) v2->xvalue;
            for (j=0; j<v1->nelems; j++) {
                /* floating-point inequality here but we genuinely do
                 * expect all processors to set bit-for-bit identical
                 * headers */
                if (fa[j] != fb[j]) {
                    msg = "%s attribute \"%s\"[%d] FLOAT (root=%f, local=%f)\n";
                    ATTR_WARN_J(msg, name, j, fa[j], fb[j])
                    DEBUG_ASSIGN_ERROR(err, NC_EMULTIDEFINE_ATTR_VAL)
                    break;
                }
            }
        }
        else if (v1->type == NC_DOUBLE) {
            double *da = (double*) v1->xvalue;
            double *db = (double*) v2->xvalue;
            for (j=0; j<v1->nelems; j++) {
                /* floating-point inequality here but we genuinely do
                 * expect all processors to set bit-for-bit identical
                 * headers */
                if (da[j] != db[j]) {
                    msg = "%s attribute \"%s\"[%d] DOUBLE (root=%f, local=%f)\n";
                    ATTR_WARN_J(msg, name, j, da[j], db[j])
                    DEBUG_ASSIGN_ERROR(err, NC_EMULTIDEFINE_ATTR_VAL)
                    break;
                }
            }
        }
        else if (v1->type == NC_INT64) {
            long long *slla = (long long*) v1->xvalue;
            long long *sllb = (long long*) v2->xvalue;
            for (j=0; j<v1->nelems; j++) {
                if (slla[j] != sllb[j]) {
                    msg = "%s attribute \"%s\"[%d] INT64 (root=%lldll, local=%lldll)\n";
                    ATTR_WARN_J(msg, name, j, slla[j], sllb[j])
                    DEBUG_ASSIGN_ERROR(err, NC_EMULTIDEFINE_ATTR_VAL)
                    break;
                }
            }
        }
        else if (v1->type == NC_UINT64) {
            unsigned long long *ulla = (unsigned long long*) v1->xvalue;
            unsigned long long *ullb = (unsigned long long*) v2->xvalue;
            for (j=0; j<v1->nelems; j++) {
                if (ulla[j] != ullb[j]) {
                    msg = "%s attribute \"%s\"[%d] UINT64 (root=%llull, local=%llull)\n";
                    ATTR_WARN_J(msg, name, j, ulla[j], ullb[j])
                    DEBUG_ASSIGN_ERROR(err, NC_EMULTIDEFINE_ATTR_VAL)
                    break;
                }
            }
        }
        if (status == NC_NOERR) status = err;

        /* overwrite local's attr with root's */
        if (ErrIsHeaderDiff(err)) {
            ncmpii_free_NC_attr(local_attr->value[i]);
            local_attr->value[i] = dup_NC_attr(root_attr->value[i]);
        }
    }

    /* delete extra attributes defined only in local copy */
    for (; i<local_attr->ndefined; i++)
        ncmpii_free_NC_attr(local_attr->value[i]);

    local_attr->ndefined = root_attr->ndefined;

    return status;
}

/*----< ncmpii_comp_vars() >--------------------------------------------------*/
/* compare the local copy of var_list against root's
 * If inconsistency is detected, overwrite local's with root's
 */
static int
ncmpii_comp_vars(int          safe_mode,
                 NC_vararray *root_var,
                 NC_vararray *local_var)
{
    int i, j, err, status=NC_NOERR;
    char *msg;

    /* check if the numbers of variables are the same */
    if (root_var->ndefined != local_var->ndefined) {
        if (safe_mode)
            printf("%s number of variables (root=%d, local=%d)\n",
                   WARN_STR, root_var->ndefined, local_var->ndefined);
        DEBUG_ASSIGN_ERROR(status, NC_EMULTIDEFINE_VAR_NUM)
    }

    for (i=0; i<root_var->ndefined; i++) {

        if (i >= local_var->ndefined) { /* if local list is shorter */
            /* copy root's variable to local */
            NC_var *new_var = dup_NC_var(root_var->value[i]);
            err = incr_NC_vararray(local_var, new_var);
            if (err != NC_NOERR) return err; /* a fatal error */
            /* local_var->ndefined is increased by 1 in incr_NC_vararray() */
            continue;
        }

        NC_var *v1 = root_var->value[i];
        NC_var *v2 = local_var->value[i];
        char *name = v1->name->cp;

#define VAR_WARN(msg, var, root, local) \
    if (safe_mode) printf(msg, WARN_STR, var, root, local);

#define VAR_WARN_J(msg, var, j, root, local) \
    if (safe_mode) printf(msg, WARN_STR, var, j, root, local);

        err = NC_NOERR;
        if (v1->name->nchars != v2->name->nchars ||
            strncmp(v1->name->cp, v2->name->cp, (size_t)v1->name->nchars) != 0) {
            msg = "%s variable %s (root=%s, local=%s)\n";
            VAR_WARN(msg, "name", name, v2->name->cp)
            DEBUG_ASSIGN_ERROR(err, NC_EMULTIDEFINE_VAR_NAME)
        }
        else if (v1->ndims != v2->ndims) {
            msg = "%s variable %s's ndims (root=%d, local=%d)\n";
            VAR_WARN(msg, name, v1->ndims, v2->ndims)
            DEBUG_ASSIGN_ERROR(err, NC_EMULTIDEFINE_VAR_NDIMS)
        }
        else if (v1->type != v2->type) {
            msg = "%s variable %s's type (root=%d, local=%d)\n";
            VAR_WARN(msg, name, v1->type, v2->type)
            DEBUG_ASSIGN_ERROR(err, NC_EMULTIDEFINE_VAR_TYPE)
        }
        else if (v1->len != v2->len) {
            msg = "%s variable %s's len (root=%lld, local=%lld)\n";
            VAR_WARN(msg, name, v1->len, v2->len)
            DEBUG_ASSIGN_ERROR(err, NC_EMULTIDEFINE_VAR_LEN)
        }
        else if (v1->begin != v2->begin) {
            msg = "%s variable %s's begin (root=%lld, local=%lld)\n";
            VAR_WARN(msg, name, v1->begin, v2->begin)
            DEBUG_ASSIGN_ERROR(err, NC_EMULTIDEFINE_VAR_BEGIN)
        }
        else {
            for (j=0; j<v1->ndims; j++) {
                if (v1->dimids[j] != v2->dimids[j]) {
                    msg = "%s variable %s's %dth dim ID (root=%d, local=%ld)\n";
                    VAR_WARN_J(msg, name, j, v1->dimids[j], v2->dimids[j])
                    DEBUG_ASSIGN_ERROR(err, NC_EMULTIDEFINE_VAR_DIMIDS)
                    break;
                }
            }
        }
        /* compare variable's attributes if by far no inconsistency is found */
        if (err == NC_NOERR) {
            err = ncmpii_comp_attrs(safe_mode, &(v1->attrs), &(v2->attrs));
            if (err != NC_NOERR && !ErrIsHeaderDiff(err))
                return err; /* a fatal error */
        }

        if (status == NC_NOERR) status = err;

        /* if there is any inconsistency, overwrite local's var with root's */
        if (ErrIsHeaderDiff(err)) {
            ncmpii_free_NC_var(local_var->value[i]);
            local_var->value[i] = dup_NC_var(root_var->value[i]);
            /* note once a new var is created, one must call
             * ncmpii_NC_computeshapes() to recalculate the shape */
        }
    }

    /* delete extra variables defined only in local copy */
    for (; i<local_var->ndefined; i++)
        ncmpii_free_NC_var(local_var->value[i]);

    local_var->ndefined = root_var->ndefined;

    return status;
}

/*----< ncmpii_hdr_check_NC() >-----------------------------------------------*/
/* This function is only called by NC_check_header()
 * It checks the header of local copy against root's and overwrites local's
 * header object, ncp, with root's header if any inconsistency is detected.
 * This function is called independently and should not contain any MPI
 * communication calls.
 */
int
ncmpii_hdr_check_NC(bufferinfo *getbuf, /* header from root */
                    NC         *ncp)
{
    int rank, err, status=NC_NOERR;
    schar magic[sizeof(ncmagic1)];
    MPI_Offset nrecs=0, chunksize=NC_DEFAULT_CHUNKSIZE;
    MPI_Aint pos_addr, base_addr;
    NC *root_ncp;

    assert(ncp != NULL);
    assert(getbuf != NULL);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* check header's magic */
    memset(magic, 0, sizeof(magic));
    err = ncmpix_getn_schar_schar((const void **)(&getbuf->pos),
                                  sizeof(magic), magic);
    if (err != NC_NOERR) {
        /* Fatal error, as root's header is significant */
        if (ncp->safe_mode) fprintf(stderr,"Error: CDF magic number from root's header\n");
        return err;
    }
    getbuf->index += (MPI_Offset)sizeof(magic);

    /* check if the first 3 letters are "CDF" */
    if (memcmp(magic, ncmagic1, sizeof(ncmagic1)-1) != 0) {
        /* Fatal error, as root's header is significant */
        if (ncp->safe_mode)
            fprintf(stderr,"Error: root's header indicates not a CDF file\n");
        DEBUG_RETURN_ERROR(NC_ENOTNC) /* should not continue */
    }

    /* allocate a header object and fill it with root's header */
    root_ncp = ncmpii_new_NC(&chunksize);

    /* consistency of magic numbers should have already been checked during
     * ncmpi_create()
     */
    if (magic[sizeof(ncmagic1)-1] == 0x5) {
        fSet(root_ncp->flags, NC_64BIT_DATA);
    }
    else if (magic[sizeof(ncmagic1)-1] == 0x2) {
        fSet(root_ncp->flags, NC_64BIT_OFFSET);
    }
    else if (magic[sizeof(ncmagic1)-1] != 0x1) {
        /* Fatal error, as root's header is significant */
        if (ncp->safe_mode)
            fprintf(stderr,"Error: root's header indicates not CDF 1/2/5 format\n");
        DEBUG_RETURN_ERROR(NC_ENOTNC) /* should not continue */
    }

    /* check version number in last byte of magic */
    int local_ver, root_ver;
         if (ncp->flags & NC_64BIT_DATA)   local_ver = 0x5;
    else if (ncp->flags & NC_64BIT_OFFSET) local_ver = 0x2;
    else                                   local_ver = 0x1;

    root_ver = magic[sizeof(ncmagic1)-1];
    if (local_ver != root_ver) {
        if (ncp->safe_mode)
            printf("%s CDF file format (local=CDF-%d, root=CDF-%d)\n",
                   WARN_STR, local_ver, root_ver);

        /* overwrite the local header object with root's */
             if (local_ver == 0x5) fClr(ncp->flags, NC_64BIT_DATA);
        else if (local_ver == 0x2) fClr(ncp->flags, NC_64BIT_OFFSET);

             if (root_ver  == 0x5) fSet(ncp->flags, NC_64BIT_DATA);
        else if (root_ver  == 0x2) fSet(ncp->flags, NC_64BIT_OFFSET);

        /* this inconsistency is not fatal */
        DEBUG_ASSIGN_ERROR(status, NC_EMULTIDEFINE_OMODE)
    }
    getbuf->version = root_ver; /* getbuf's version has not been set before */

#if SIZEOF_MPI_OFFSET < 8
    if (root_ver > 1) {
        /* for NC_64BIT_DATA or NC_64BIT_OFFSET, MPI_Offset must be 8 bytes */
        if (ncp->safe_mode) fprintf(stderr,"Error: cannot support CDF-2 and CDF-5 on this machine\n");
        DEBUG_RETURN_ERROR(NC_ESMALL) /* should not continue */
    }
#endif

    /* since getbuf contains the entire root's header, we do not need to check
     * for any next element in the buffer. Similarly, for all possible calls
     * to hdr_check_buffer() from this subroutine, hdr_fetch() will never be
     * called.
     * (move on to the next element in header: number of records)
     err = hdr_check_buffer(getbuf, (getbuf->version == 5) ? 8 : 4);
     if (err != NC_NOERR) {
         if (ncp->safe_mode)
             fprintf(stderr,"Error: root's header is too short\n");
         return err;
     }
     */

    if (getbuf->version == 5) {
        uint64 tmp=0;
        err = ncmpix_get_uint64((const void **)(&getbuf->pos), &tmp);
        nrecs = (MPI_Offset)tmp;
    }
    else {
        uint tmp=0;
        err = ncmpix_get_uint32((const void **)(&getbuf->pos), &tmp);
        nrecs = (MPI_Offset)tmp;
    }
    if (err != NC_NOERR) {
        if (ncp->safe_mode)
            fprintf(stderr,"Error: failed to read numrecs from root's header\n");
        return err; /* should not continue */
    }

    /* move the buffer point forward */
    if (getbuf->version == 5)
        getbuf->index += X_SIZEOF_INT64;
    else
        getbuf->index += X_SIZEOF_SIZE_T;

    root_ncp->numrecs = nrecs;
    if (root_ncp->numrecs != ncp->numrecs) {
        if (ncp->safe_mode)
            printf("%s number of records (local=%lld, root=%lld)\n",
                   WARN_STR, ncp->numrecs, root_ncp->numrecs);
        /* overwrite the local header's numrecs */
        ncp->numrecs = root_ncp->numrecs;
        if (status == NC_NOERR) DEBUG_ASSIGN_ERROR(status, NC_EMULTIDEFINE_NUMRECS)
    }

#ifdef HAVE_MPI_GET_ADDRESS
    MPI_Get_address(getbuf->pos,  &pos_addr);
    MPI_Get_address(getbuf->base, &base_addr);
#else
    MPI_Address(getbuf->pos,  &pos_addr);
    MPI_Address(getbuf->base, &base_addr);
#endif
    assert(pos_addr < base_addr + getbuf->size);

    /* get the next header element dim_list from getbuf to root_ncp */
    err = hdr_get_NC_dimarray(getbuf, &root_ncp->dims);
    if (err != NC_NOERR) return err; /* fatal error */

    /* compare local's and root's dim_list */
    err = ncmpii_comp_dims(ncp->safe_mode, &root_ncp->dims, &ncp->dims);
    if (err != NC_NOERR && !ErrIsHeaderDiff(err))
        return err; /* a fatal error */
    if (status == NC_NOERR) status = err;

    /* get the next header element gatt_list from getbuf to root_ncp */
    err = hdr_get_NC_attrarray(getbuf, &root_ncp->attrs);
    if (err != NC_NOERR) return err; /* fatal error */

    /* get the next header element att_list from getbuf to root_ncp */
    err = ncmpii_comp_attrs(ncp->safe_mode, &root_ncp->attrs, &ncp->attrs);
    if (err != NC_NOERR && !ErrIsHeaderDiff(err))
        return err; /* a fatal error */
    if (status == NC_NOERR) status = err;

    /* get the next header element var_list from getbuf to root_ncp */
    err = hdr_get_NC_vararray(getbuf, &root_ncp->vars);
    if (err != NC_NOERR) return err; /* fatal error */

    /* compare local's and root's var_list */
    err = ncmpii_comp_vars(ncp->safe_mode, &root_ncp->vars, &ncp->vars);
    if (err != NC_NOERR && !ErrIsHeaderDiff(err))
        return err; /* a fatal error */
    if (status == NC_NOERR) status = err;

    if (err != NC_NOERR) { /* header has been sync-ed with root */
        /* recompute shape is required for every new variable created */
        err = ncmpii_NC_computeshapes(ncp);
        if (err != NC_NOERR) return err; /* a fatal error */
    }
    /* now, the local header object has been sync-ed with root */

    if (ncp->safe_mode && ErrIsHeaderDiff(status)) {
        /* recompute header size */
        MPI_Offset root_xsz, local_xsz;
         root_xsz = ncmpii_hdr_len_NC(root_ncp);
        local_xsz = ncmpii_hdr_len_NC(ncp);
        /* root's header size is getbuf->size */
        assert( ncp->xsz == getbuf->size &&
                root_xsz == local_xsz    &&
               local_xsz == getbuf->size);
    }
    ncmpii_free_NC(root_ncp);
    return status;
}

/*----< ncmpii_write_header() >-----------------------------------------------*/
/* This function is called only in data mode (collective or independent) and by
 * 1. ncmpi_rename_att()
 * 2. ncmpi_copy_att()
 * 3. ncmpii_put_att()
 * 4. ncmpi_rename_dim()
 * 5. ncmpi_rename_var()
 *
 * This function is collective (even in independent data mode) */
int ncmpii_write_header(NC *ncp)
{
    int rank, status=NC_NOERR, mpireturn, err;
    MPI_File fh;

    /* Write the entire header to the file. This function may be called from
     * a rename API. In that case, we cannot just change the variable name in
     * the file header, because if the file space occupied by the name shrinks,
     * all metadata following the new name must be moved ahead.
     */

    fh = ncp->nciop->collective_fh;
    if (NC_indep(ncp))
        fh = ncp->nciop->independent_fh;

    MPI_Comm_rank(ncp->nciop->comm, &rank);
    if (rank == 0) {
        MPI_Status mpistatus;
        void *buf = NCI_Malloc((size_t)ncp->xsz); /* header's write buffer */

        /* copy header object to write buffer */
        status = ncmpii_hdr_put_NC(ncp, buf);

        if (ncp->xsz != (int)ncp->xsz) {
            NCI_Free(buf);
            DEBUG_RETURN_ERROR(NC_EINTOVERFLOW)
        }
        TRACE_IO(MPI_File_write_at)(fh, 0, buf, (int)ncp->xsz, MPI_BYTE, &mpistatus);
        if (mpireturn != MPI_SUCCESS) {
            err = ncmpii_handle_error(mpireturn, "MPI_File_write_at");
            if (status == NC_NOERR) {
                err = (err == NC_EFILE) ? NC_EWRITE : err;
                DEBUG_ASSIGN_ERROR(status, err)
            }
        }
        else {
            int put_size;
            MPI_Get_count(&mpistatus, MPI_BYTE, &put_size);
            ncp->nciop->put_size += put_size;
        }
        NCI_Free(buf);
    }

    if (ncp->safe_mode == 1) {
        /* broadcast root's status, because only root writes to the file */
        int root_status = status;
        TRACE_COMM(MPI_Bcast)(&root_status, 1, MPI_INT, 0, ncp->nciop->comm);
        /* root's write has failed, which is serious */
        if (root_status == NC_EWRITE) DEBUG_ASSIGN_ERROR(status, NC_EWRITE)
        if (mpireturn != MPI_SUCCESS) {
            ncmpii_handle_error(mpireturn,"MPI_Bcast");
            DEBUG_RETURN_ERROR(NC_EMPI)
        }
    }

    /* update file header size */
    ncp->xsz = ncmpii_hdr_len_NC(ncp);

    if (NC_doFsync(ncp)) { /* NC_SHARE is set */
        TRACE_IO(MPI_File_sync)(fh);
        if (mpireturn != MPI_SUCCESS) {
            ncmpii_handle_error(mpireturn,"MPI_File_sync");
            DEBUG_RETURN_ERROR(NC_EMPI)
        }
        TRACE_COMM(MPI_Barrier)(ncp->nciop->comm);
        if (mpireturn != MPI_SUCCESS) {
            ncmpii_handle_error(mpireturn,"MPI_Barrier");
            DEBUG_RETURN_ERROR(NC_EMPI)
        }
    }

    return status;
}


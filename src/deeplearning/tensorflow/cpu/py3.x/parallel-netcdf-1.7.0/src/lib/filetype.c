/*
 *  Copyright (C) 2003, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id: filetype.c 2335 2016-03-03 14:55:12Z wkliao $ */

#if HAVE_CONFIG_H
# include "ncconfig.h"
#endif

#include <stdio.h>
#include <unistd.h>
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#include <assert.h>

#include <mpi.h>

#include "nc.h"
#include "ncx.h"
#include "macro.h"


#if SIZEOF_MPI_AINT != SIZEOF_MPI_OFFSET
/*----< check_recsize_too_big() >--------------------------------------------*/
inline static int
check_recsize_too_big(NC *ncp)
{
    int ret = NC_NOERR;
    /* assertion: because recsize will be used to set up the file
     * view, we must ensure there is no overflow when specifying
     * how big a stride there is between items (think interleaved
     * records).
     *
     * note: 'recsize' is the sum of the record size of all record
     * variables in this dataset */
    if (ncp->recsize != (MPI_Aint)ncp->recsize) {
        fprintf(stderr, "Type overflow: unable to read/write multiple records in this dataset\non this platform. Please either access records of this record variable\none-at-a-time or run on a 64 bit platform\n");
        DEBUG_ASSIGN_ERROR(ret, NC_ESMALL)
    }
    /* the assert here might be harsh, but without it, users will get corrupt
     * data. Now, we just skip this request to avoid this assertion. */
    /* assert (ncp->recsize == (MPI_Aint)ncp->recsize); */
    return ret;
}
#endif

/*----< NC_start_count_stride_ck() >-----------------------------------------*/
/*
 * Check whether start, count, stride values are valid for the variable.
 * Note that even if the request size is zero, this check is enforced in both
 * netCDF and PnetCDF. Otherwise, many test cases under test directory can fail.
 * Arguments count and stride can be NULL.
 */
int
NC_start_count_stride_ck(const NC         *ncp,
                         const NC_var     *varp,
                         const MPI_Offset *start,
                         const MPI_Offset *count,
                         const MPI_Offset *stride,
                         const int         rw_flag) /* for read or write */
{
    int i=0;

    if (varp->ndims == 0) return NC_NOERR; /* 'scalar' variable */

    /* negative start[] is illegal */
    if (start[0] < 0) DEBUG_RETURN_ERROR(NC_EINVALCOORDS)

    if (IS_RECVAR(varp)) {
        if (!fIsSet(ncp->flags, NC_64BIT_DATA) && /* not CDF-5 */
            start[0] > X_UINT_MAX) /* sanity check */
            DEBUG_RETURN_ERROR(NC_EINVALCOORDS)

        if (count != NULL && count[0] < 0) /* no negative count[] */
            DEBUG_RETURN_ERROR(NC_ENEGATIVECNT)

        /* for record variable, [0] is the NC_UNLIMITED dimension */
        if (rw_flag == READ_REQ) { /* read cannot go beyond current numrecs */
            if (start[0] >= ncp->numrecs)
                DEBUG_RETURN_ERROR(NC_EINVALCOORDS)

            if (count != NULL) {
                if (stride == NULL) { /* for vara APIs */
                    if (start[0] + count[0] > ncp->numrecs)
                        DEBUG_RETURN_ERROR(NC_EEDGE)
                }
                else { /* for vars APIs */
                    if (count[0] > 0 &&
                        start[0] + (count[0]-1) * stride[0] >= ncp->numrecs)
                        DEBUG_RETURN_ERROR(NC_EEDGE)
                }
            }
            /* else is for var1 APIs */
        }

        if (stride != NULL && stride[0] == 0) DEBUG_RETURN_ERROR(NC_ESTRIDE)

        /* In collective data mode where numrecs is always kept consistent
         * across memory, then there is no need to update numrecs.
         * (If NC_SHARE is set, then numrecs is even sync-ed with file.)
         *
         * In independent data mode, numrecs in memory across processes
         * and file can be inconsistent. Even re-reading numrecs from file
         * cannot get the latest value, because in independent mode,
         * numrecs in file is not updated (due to race condition).
         * For example, a subset of processes write a new record and at
         * the same time another set writes 2 new records. Even if NC_SHARE
         * is set, new values of numrecs cannot be written to the file,
         * because it can cause a race condition (atomic read-modify IO is
         * required to solve this problem and MPI-IO cannot do it). Simply
         * said, numrecs is not automatically kept consistent in
         * independent mode. Users must call ncmpi_sync_numrecs()
         * collectively to sync the value. So, here what PnetCDF can do
         * best is just to check numrecs against the local value.
         */

        /* skip checking the record dimension */
        i = 1;
    }

    for (; i<varp->ndims; i++) {
        if (start[i] < 0 || start[i] >= varp->shape[i])
            DEBUG_RETURN_ERROR(NC_EINVALCOORDS)

        if (varp->shape[i] < 0) DEBUG_RETURN_ERROR(NC_EEDGE)

        if (count != NULL) {
            if (count[i] < 0) /* no negative count[] */
                DEBUG_RETURN_ERROR(NC_ENEGATIVECNT)

            if (stride == NULL) { /* for vara APIs */
                if (count[i] > varp->shape[i] ||
                    start[i] + count[i] > varp->shape[i])
                    DEBUG_RETURN_ERROR(NC_EEDGE)
            }
            else { /* for vars APIs */
                if (count[i] > 0 &&
                    start[i] + (count[i]-1) * stride[i] >= varp->shape[i])
                    DEBUG_RETURN_ERROR(NC_EEDGE)
                if (stride[i] == 0) DEBUG_RETURN_ERROR(NC_ESTRIDE)
            }
        }
        /* else is for var1 APIs */
    }
    return NC_NOERR;
}

/*----< ncmpii_get_offset() >------------------------------------------------*/
/* returns the file offset of the last byte accessed of this request */
int
ncmpii_get_offset(NC               *ncp,
                  NC_var           *varp,
                  const MPI_Offset  starts[],   /* [varp->ndims] */
                  const MPI_Offset  counts[],   /* [varp->ndims] */
                  const MPI_Offset  strides[],  /* [varp->ndims] */
                  const int         rw_flag,
                  MPI_Offset       *offset_ptr) /* return file offset */
{
    MPI_Offset offset, *end_off=NULL;
    int status, i, ndims;

    offset = varp->begin; /* beginning file offset of this variable */
    ndims  = varp->ndims; /* number of dimensions of this variable */

    if (counts != NULL) {
        end_off = (MPI_Offset*) NCI_Malloc((size_t)ndims * SIZEOF_MPI_OFFSET);

        if (strides != NULL) {
            for (i=0; i<ndims; i++)
                end_off[i] = starts[i] + (counts[i] - 1) * strides[i];
        }
        else { /* strides == NULL */
            for (i=0; i<ndims; i++)
                end_off[i] = starts[i] + counts[i] - 1;
        }
    }
    else { /* when counts == NULL strides is of no use */
        end_off = (MPI_Offset*) starts;
    }

    /* check whether end_off is valid */
    status = NC_start_count_stride_ck(ncp, varp, end_off, NULL, NULL, rw_flag);
    if (status != NC_NOERR) {
#ifdef CDEBUG
        printf("%s(): NC_start_count_stride_ck() fails\n",__func__);
#endif
        if (end_off != NULL && end_off != starts) NCI_Free(end_off);
        return status;
    }

    if (ndims > 0) {
        if (IS_RECVAR(varp))
            /* no need to check recsize here: if MPI_Offset is only 32 bits we
               will have had problems long before here */
            offset += end_off[0] * ncp->recsize;
        else
            offset += end_off[ndims-1] * varp->xsz;

        if (ndims > 1) {
            if (IS_RECVAR(varp))
                offset += end_off[ndims - 1] * varp->xsz;
            else
                offset += end_off[0] * varp->dsizes[1] * varp->xsz;

            for (i=1; i<ndims-1; i++)
                offset += end_off[i] * varp->dsizes[i+1] * varp->xsz;
        }
    }
    if (counts != NULL && end_off != NULL)
        NCI_Free(end_off);

    *offset_ptr = offset;
    return NC_NOERR;
}

/*----< ncmpii_is_request_contiguous() >-------------------------------------*/
int
ncmpii_is_request_contiguous(NC               *ncp,
                             NC_var           *varp,
                             const MPI_Offset  starts[],
                             const MPI_Offset  counts[])
{
    /* determine whether the get/put request to this variable using
       starts[] and counts[] is contiguous in file */
    int i, j, most_sig_dim, ndims=varp->ndims;

    /* this variable is a scalar */
    if (ndims == 0) return 1;

    for (i=0; i<ndims; i++)
         if (counts[i] == 0) /* zero length request */
             return 1;

    most_sig_dim = 0; /* record dimension */

    if (IS_RECVAR(varp)) {
        /* if there are more than one record variable, then the record
           dimensions, counts[0] must == 1. For now, we assume there
           are more than one record variable.
           TODO: we may need an API to inquire how many record variables
           are defined */
        if (ncp->vars.num_rec_vars > 1) {
            /* or if (ncp->recsize > varp->len) more than one record variable */
            if (counts[0] > 1) return 0;

            /* we need to check from dimension ndims-1 up to dimension 1 */
            most_sig_dim = 1;
        }
        /* if there is only one record variable, then we need to check from
         * dimension ndims-1 up to dimension 0 */
    }

    for (i=ndims-1; i>most_sig_dim; i--) {
        /* find the first counts[i] that is not the entire dimension */
        if (counts[i] < varp->shape[i]) {
            /* check dim from i-1, i-2, ..., most_sig_dim and
               their counts[] should all be 1 */
            for (j=i-1; j>=most_sig_dim; j--) {
                if (counts[j] > 1)
                    return 0;
            }
            break;
        }
        else { /* counts[i] == varp->shape[i] */
            /* when accessing the entire dimension, starts[i] must be 0 */
            if (starts[i] != 0) return 0; /* actually this should be error */
        }
    }
    return 1;
}

#ifndef HAVE_MPI_TYPE_CREATE_SUBARRAY
/*----< ncmpii_type_create_subarray() >--------------------------------------*/
/* this is to be used when MPI_Type_create_subarray() is not available,
 * typically for MPI-1 implementation only */
static int
ncmpii_type_create_subarray(int           ndims,
                            int          *array_of_sizes,    /* [ndims] */
                            int          *array_of_subsizes, /* [ndims] */
                            int          *array_of_starts,   /* [ndims] */
                            int           order,
                            MPI_Datatype  oldtype,
                            MPI_Datatype *newtype)
{
    int i, err, blklens[3] = {1, 1, 1};
    MPI_Datatype type1, type2;
    MPI_Aint extent, size, array_size, stride, disps[3];

    if (ndims == 0) DEBUG_RETURN_ERROR(NC_EDIMMETA)

#ifdef HAVE_MPI_TYPE_GET_EXTENT
    MPI_Aint lb;
    MPI_Type_get_extent(oldtype, &lb, &extent);
#else
    MPI_Type_extent(oldtype, &extent);
#endif
    array_size = extent;
    for (i=0; i<ndims; i++) array_size *= array_of_sizes[i];

    if (ndims == 1) {
        /* blklens argument in MPI_Type_create_hindexed() is of type int */
        blklens[1] = array_of_subsizes[0];
        disps[1] = extent * array_of_starts[0];

#if defined (HAVE_MPI_TYPE_CREATE_HINDEXED) && defined(HAVE_MPI_TYPE_CREATE_RESIZED)
        /* take advantage of disps argument is of type MPI_Aint */
        err = MPI_Type_create_hindexed(1, &blklens[1], &disps[1], oldtype, &type1);
        if (err != MPI_SUCCESS)
            return ncmpii_handle_error(err, "MPI_Type_create_hindexed");
        MPI_Type_commit(&type1);

        /* add holes in the beginning and tail of type1 */
        err = MPI_Type_create_resized(type1, 0, array_size, newtype);
        if (err != MPI_SUCCESS)
            return ncmpii_handle_error(err, "MPI_Type_create_resized");
        MPI_Type_free(&type1);
#else
        /* add holes in the beginning and tail of oldtype */
        MPI_Datatype types[3];
        types[0] = MPI_LB; types[1] = oldtype; types[2] = MPI_UB;
        disps[0] = 0;                          disps[2] = array_size;
        err = MPI_Type_struct(3, blklens, disps, types, newtype);
        if (err != MPI_SUCCESS)
            return ncmpii_handle_error(err, "MPI_Type_struct");
#endif
        return NC_NOERR;
    }
    /* now, ndims > 1 */

    /* first create a datatype for the least 2 significant dimensions */

    /* blklens argument in MPI_Type_create_hvector() is of type int */
    blklens[0] = array_of_subsizes[ndims-1];
    stride = array_of_sizes[ndims-1] * extent;
#ifdef HAVE_MPI_TYPE_CREATE_HVECTOR
    err = MPI_Type_create_hvector(array_of_subsizes[ndims-2], blklens[0],
                                  stride, oldtype, &type1);
    if (err != MPI_SUCCESS)
        return ncmpii_handle_error(err, "MPI_Type_create_hvector");
#else
    err = MPI_Type_hvector(array_of_subsizes[ndims-2], blklens[0],
                           stride, oldtype, &type1);
    if (err != MPI_SUCCESS)
        return ncmpii_handle_error(err, "MPI_Type_hvector");
#endif
    MPI_Type_commit(&type1);

    /* now iterate through the rest dimensions */
    for (i=ndims-3; i>=0; i--) {
        stride *= array_of_sizes[i+1];
#ifdef HAVE_MPI_TYPE_CREATE_HVECTOR
        err = MPI_Type_create_hvector(array_of_subsizes[i], 1, stride, type1, &type2);
        if (err != MPI_SUCCESS)
            return ncmpii_handle_error(err, "MPI_Type_create_hvector");
#else
        err = MPI_Type_hvector(array_of_subsizes[i], 1, stride, type1, &type2);
        if (err != MPI_SUCCESS)
            return ncmpii_handle_error(err, "MPI_Type_hvector");
#endif
        MPI_Type_commit(&type2);
        MPI_Type_free(&type1);
        type1 = type2;
    }

    /* disps[1] is the first byte displacement of the subarray */
    disps[1] = array_of_starts[ndims-1] * extent;
    size = 1;
    for (i=ndims-2; i>=0; i--) {
        size *= array_of_sizes[i+1];
        disps[1] += size * array_of_starts[i];
    }

    /* disps[2] is the size of the global array */
    disps[2] = array_size;

    /* disps[0] is the beginning of the global array */
    disps[0] = 0;

    /* make filetype the same as calling MPI_Type_create_subarray() */
    blklens[0] = 1;
#if defined (HAVE_MPI_TYPE_CREATE_HINDEXED) && defined(HAVE_MPI_TYPE_CREATE_RESIZED)
    /* adjust LB and UB without using MPI_LB or MPI_UB */
    err = MPI_Type_create_hindexed(1, blklens, &disps[1], type1, &type2);
    if (err != MPI_SUCCESS)
        return ncmpii_handle_error(err, "MPI_Type_create_hindexed");
    MPI_Type_commit(&type2);
    err = MPI_Type_create_resized(type2, disps[0], disps[2], newtype);
    if (err != MPI_SUCCESS)
        return ncmpii_handle_error(err, "MPI_Type_create_resized");
    MPI_Type_free(&type2);
#else
    MPI_Datatype types[3];
    types[0] = MPI_LB;
    types[1] = type1;
    types[2] = MPI_UB;
    err = MPI_Type_struct(3, blklens, disps, types, newtype);
    if (err != MPI_SUCCESS)
        return ncmpii_handle_error(err, "MPI_Type_struct");
#endif
    MPI_Type_free(&type1);

    return NC_NOERR;
}
#endif

/*----< ncmpii_type_create_subarray64() >------------------------------------*/
/* This subroutine is to achieve the same result as MPI_Type_create_subarray()
 * but it takes arguments in type of MPI_Offset, instead of int. It also
 * checked for any possible 4-byte integer overflow.
 */
static int
ncmpii_type_create_subarray64(int           ndims,
                              MPI_Offset   *array_of_sizes,    /* [ndims] */
                              MPI_Offset   *array_of_subsizes, /* [ndims] */
                              MPI_Offset   *array_of_starts,   /* [ndims] */
                              int           order,
                              MPI_Datatype  oldtype,
                              MPI_Datatype *newtype)
{
    int i, err, tag, blklens[3] = {1, 1, 1};
    MPI_Datatype type1, type2;
    MPI_Aint extent, size, array_size, stride, disps[3];
#ifdef HAVE_MPI_TYPE_GET_EXTENT
    MPI_Aint lb;
#endif

    if (ndims == 0) DEBUG_RETURN_ERROR(NC_EDIMMETA)

    /* check if any of the dimensions is larger than 2^31-1 */
    tag = 0;
    for (i=0; i<ndims; i++) {
        if (array_of_sizes[i] > 2147483647) {
            tag = 1;
            break;
        }
    }

    if (tag == 0) {
        /* none of dimensions > 2^31-1, we can safely use
         * MPI_Type_create_subarray */
        int *sizes    = (int*) NCI_Malloc((size_t)ndims * 3 * SIZEOF_INT);
        int *subsizes = sizes    + ndims;
        int *starts   = subsizes + ndims;
        for (i=0; i<ndims; i++) {
            sizes[i]    = (int)array_of_sizes[i];
            subsizes[i] = (int)array_of_subsizes[i];
            starts[i]   = (int)array_of_starts[i];
            if (array_of_sizes[i]    != sizes[i] ||
                array_of_subsizes[i] != subsizes[i] ||
                array_of_starts[i]   != starts[i])
                DEBUG_RETURN_ERROR(NC_EINTOVERFLOW)
        }
#ifdef HAVE_MPI_TYPE_CREATE_SUBARRAY
        err = MPI_Type_create_subarray(ndims, sizes, subsizes, starts,
                                       order, oldtype, newtype);
        NCI_Free(sizes);
        if (err != MPI_SUCCESS)
            return ncmpii_handle_error(err, "MPI_Type_create_subarray");
#else
        err = ncmpii_type_create_subarray(ndims, sizes, subsizes, starts,
                                          order, oldtype, newtype);
        NCI_Free(sizes);
#endif
        return err;
    }

    /* at least one dimension is of size > 2^31-1 and we cannot use
     * MPI_Type_create_subarray() to create the newtype,
     * as its arguments array_of_sizes[] and array_of_starts[] are of
     * type int. One solution is to use a combination of
     * MPI_Type_create_hvector(), MPI_Type_create_hindexed(),
     * MPI_Type_create_resized(), and MPI_Type_struct(), as one
     * of their arguments, stride and indices[], are of type MPI_Aint
     * (a possible 8-byte integer) that can be used to store the value
     * of the dimension whose size is > 2^31-1. However, on a machine
     * where MPI_Aint is 4-byte integer, those MPI_Aint arguments will
     * cause overflow.
     */
#if SIZEOF_MPI_AINT != SIZEOF_MPI_OFFSET
    DEBUG_RETURN_ERROR(NC_EAINT_TOO_SMALL)
#endif

#ifdef HAVE_MPI_TYPE_GET_EXTENT
    MPI_Type_get_extent(oldtype, &lb, &extent);
#else
    MPI_Type_extent(oldtype, &extent);
#endif
    array_size = extent;
    for (i=0; i<ndims; i++) array_size *= array_of_sizes[i];

    if (ndims == 1) {
        /* blklens argument in MPI_Type_create_hindexed() is of type int */
        blklens[1] = (int)array_of_subsizes[0];
        if (array_of_subsizes[0] != blklens[1]) /* check int overflow */
            DEBUG_RETURN_ERROR(NC_EINTOVERFLOW)
        disps[1] = extent * array_of_starts[0];

#if defined (HAVE_MPI_TYPE_CREATE_HINDEXED) && defined(HAVE_MPI_TYPE_CREATE_RESIZED)
        /* take advantage of disps argument is of type MPI_Aint */
        err = MPI_Type_create_hindexed(1, &blklens[1], &disps[1], oldtype, &type1);
        if (err != MPI_SUCCESS)
            return ncmpii_handle_error(err, "MPI_Type_create_hindexed");
        MPI_Type_commit(&type1);

        /* add holes in the beginning and tail of type1 */
        err = MPI_Type_create_resized(type1, 0, array_size, newtype);
        if (err != MPI_SUCCESS)
            return ncmpii_handle_error(err, "MPI_Type_create_resized");
        MPI_Type_free(&type1);
#else
        /* add holes in the beginning and tail of oldtype */
        MPI_Datatype types[3];
        types[0] = MPI_LB; types[1] = oldtype; types[2] = MPI_UB;
        disps[0] = 0;                          disps[2] = array_size;
        err = MPI_Type_struct(3, blklens, disps, types, newtype);
        if (err != MPI_SUCCESS)
            return ncmpii_handle_error(err, "MPI_Type_struct");
#endif
        return NC_NOERR;
    }
    /* now, ndims > 1 */

    /* first create a datatype for the least 2 significant dimensions */

    /* count and blocklength arguments in MPI_Type_create_hvector() are of
     * type int. We need to check for integer overflow */
    int count, blocklength;
    count = (int)array_of_subsizes[ndims-2];
    blocklength = (int)array_of_subsizes[ndims-1];
    if (array_of_subsizes[ndims-2] != count ||
        array_of_subsizes[ndims-1] != blocklength) /* check int overflow */
        DEBUG_RETURN_ERROR(NC_EINTOVERFLOW)
    stride = array_of_sizes[ndims-1] * extent;
#ifdef HAVE_MPI_TYPE_CREATE_HVECTOR
    err = MPI_Type_create_hvector(count, blocklength, stride, oldtype, &type1);
    if (err != MPI_SUCCESS)
        return ncmpii_handle_error(err, "MPI_Type_create_hvector");
#else
    err = MPI_Type_hvector(count, blocklength, stride, oldtype, &type1);
    if (err != MPI_SUCCESS)
        return ncmpii_handle_error(err, "MPI_Type_hvector");
#endif
    MPI_Type_commit(&type1);

    /* now iterate through the rest dimensions */
    for (i=ndims-3; i>=0; i--) {
        count = (int)array_of_subsizes[i];
        if (array_of_subsizes[i] != count) /* check int overflow */
            DEBUG_RETURN_ERROR(NC_EINTOVERFLOW)
        stride *= array_of_sizes[i+1];
#ifdef HAVE_MPI_TYPE_CREATE_HVECTOR
        err = MPI_Type_create_hvector(count, 1, stride, type1, &type2);
        if (err != MPI_SUCCESS)
            return ncmpii_handle_error(err, "MPI_Type_create_hvector");
#else
        err = MPI_Type_hvector(count, 1, stride, type1, &type2);
        if (err != MPI_SUCCESS)
            return ncmpii_handle_error(err, "MPI_Type_hvector");
#endif
        MPI_Type_commit(&type2);
        MPI_Type_free(&type1);
        type1 = type2;
    }

    /* disps[0] is the displacement to the beginning of the global array */
    disps[0] = 0;

    /* disps[1] is the first byte displacement of the subarray */
    disps[1] = array_of_starts[ndims-1] * extent;
    size = 1;
    for (i=ndims-2; i>=0; i--) {
        size *= array_of_sizes[i+1];
        disps[1] += size * array_of_starts[i];
    }

    /* disps[2] is the size of the global array */
    disps[2] = array_size;

    /* make filetype the same as calling MPI_Type_create_subarray() */
#if defined(HAVE_MPI_TYPE_CREATE_HINDEXED) && defined(HAVE_MPI_TYPE_CREATE_RESIZED)
    /* adjust LB and UB without using MPI_LB or MPI_UB */
    err = MPI_Type_create_hindexed(1, blklens, &disps[1], type1, &type2);
    if (err != MPI_SUCCESS)
        return ncmpii_handle_error(err, "MPI_Type_create_hindexed");
    MPI_Type_commit(&type2);
    err = MPI_Type_create_resized(type2, disps[0], disps[2], newtype);
    if (err != MPI_SUCCESS)
        return ncmpii_handle_error(err, "MPI_Type_create_resized");
    MPI_Type_free(&type2);
#else
    MPI_Datatype types[3];
    types[0] = MPI_LB;
    types[1] = type1;
    types[2] = MPI_UB;
    err = MPI_Type_struct(3, blklens, disps, types, newtype);
    if (err != MPI_SUCCESS)
        return ncmpii_handle_error(err, "MPI_Type_struct");
#endif
    MPI_Type_free(&type1);

    return NC_NOERR;
}

/*----< ncmpii_vara_create_filetype() >--------------------------------------*/
static int
ncmpii_vara_create_filetype(NC               *ncp,
                            NC_var           *varp,
                            const MPI_Offset *start,
                            const MPI_Offset *count,
                            int               rw_flag,
                            int              *blocklen,
                            MPI_Offset       *offset_ptr,
                            MPI_Datatype     *filetype_ptr,
                            int              *is_filetype_contig)
{
    int          dim, status, err;
    MPI_Offset   nbytes, offset;
    MPI_Datatype filetype;

    /* check whether start, count are valid */
    status = NC_start_count_stride_ck(ncp, varp, start, count, NULL, rw_flag);
    if (status != NC_NOERR) return status;

    /* calculate the request size */
    nbytes = varp->xsz;
    for (dim=0; dim<varp->ndims; dim++) nbytes *= count[dim];
    if (nbytes != (int)nbytes) DEBUG_RETURN_ERROR(NC_EINTOVERFLOW)
    if (blocklen != NULL) *blocklen = (int)nbytes;

    /* when nbytes == 0 or varp is a scalar, i.e. varp->ndims == 0, no need to
     * create a filetype
     */
    if (varp->ndims == 0 || nbytes == 0) {
        *offset_ptr   = varp->begin;
        *filetype_ptr = MPI_BYTE;
        if (is_filetype_contig != NULL) *is_filetype_contig = 1;
        return NC_NOERR;
    }

    /* if the request is contiguous in file, no need to create a filetype */
    if (ncmpii_is_request_contiguous(ncp, varp, start, count)) {
        status = ncmpii_get_offset(ncp, varp, start, NULL, NULL, rw_flag,
                                   &offset);
        *offset_ptr   = offset;
        *filetype_ptr = MPI_BYTE;
        if (is_filetype_contig != NULL) *is_filetype_contig = 1;
        return status;
    }

    /* hereinafter fileview is noncontiguous, i.e. filetype != MPI_BYTE.
     * Since we will construct a filetype, blocklen is set to 1.
     */
    if (blocklen           != NULL) *blocklen           = 1;
    if (is_filetype_contig != NULL) *is_filetype_contig = 0;
    offset = varp->begin;

    /* previously, request size has been checked and it must > 0 */
    if (IS_RECVAR(varp)) {
        int blocklength;
        MPI_Datatype rectype=MPI_BYTE;

#if SIZEOF_MPI_AINT != SIZEOF_MPI_OFFSET
        /* check overflow only if MPI_Aint is smaller than MPI_Offset */
        status = check_recsize_too_big(ncp);
        if (status != NC_NOERR) return status;
#endif
        /* check overflow, because 1st argument of hvector is of type int */
        if (count[0] != (int) count[0]) DEBUG_RETURN_ERROR(NC_EINTOVERFLOW)

        offset += start[0] * ncp->recsize;

        if (varp->ndims > 1) {
            /* when ndims > 1, we need to construct a subarray type for a
             * single record, i.e. for dimension 1 ... ndims-1 */
            MPI_Offset *shape64, *subcount64, *substart64;

            shape64 = (MPI_Offset*) NCI_Malloc((size_t)varp->ndims * 3 * SIZEOF_MPI_OFFSET);
            subcount64 = shape64    + varp->ndims;
            substart64 = subcount64 + varp->ndims;

            shape64[0]    = count[0];
            subcount64[0] = count[0];
            substart64[0] = 0;

            for (dim=1; dim<varp->ndims; dim++) {
                shape64[dim]    = varp->shape[dim];
                subcount64[dim] = count[dim];
                substart64[dim] = start[dim];
            }
            shape64[varp->ndims-1]    *= varp->xsz;
            subcount64[varp->ndims-1] *= varp->xsz;
            substart64[varp->ndims-1] *= varp->xsz;

            status = ncmpii_type_create_subarray64(varp->ndims-1, shape64+1,
                                 subcount64+1, substart64+1, MPI_ORDER_C,
                                 MPI_BYTE, &rectype);
            NCI_Free(shape64);
            if (status != NC_NOERR) return status;

            MPI_Type_commit(&rectype);
            blocklength = 1;
        }
        else { /* no subarray datatype is needed */
            blocklength = varp->xsz;
        }

#ifdef HAVE_MPI_TYPE_CREATE_HVECTOR
        err = MPI_Type_create_hvector((int)count[0], blocklength, ncp->recsize,
                                      rectype, &filetype);
        if (err != MPI_SUCCESS)
            return ncmpii_handle_error(err, "MPI_Type_create_hvector");
#else
        err = MPI_Type_hvector((int)count[0], blocklength, ncp->recsize,
                               rectype, &filetype);
        if (err != MPI_SUCCESS)
            return ncmpii_handle_error(err, "MPI_Type_hvector");
#endif
        if (rectype != MPI_BYTE) MPI_Type_free(&rectype);
    }
    else { /* for non-record variable, just create a subarray datatype */
        MPI_Offset *shape64, *subcount64, *substart64;
        shape64 = (MPI_Offset*) NCI_Malloc((size_t)varp->ndims * 3 * SIZEOF_MPI_OFFSET);
        subcount64 = shape64    + varp->ndims;
        substart64 = subcount64 + varp->ndims;

        for (dim=0; dim<varp->ndims; dim++) {
            shape64[dim]    = varp->shape[dim];
            subcount64[dim] = count[dim];
            substart64[dim] = start[dim];
        }
        shape64[varp->ndims-1]    *= varp->xsz;
        subcount64[varp->ndims-1] *= varp->xsz;
        substart64[varp->ndims-1] *= varp->xsz;

        status = ncmpii_type_create_subarray64(varp->ndims, shape64, subcount64,
                                               substart64, MPI_ORDER_C,
                                               MPI_BYTE, &filetype);
        NCI_Free(shape64);
        if (status != NC_NOERR) return status;
    }
    MPI_Type_commit(&filetype);

    *offset_ptr   = offset;
    *filetype_ptr = filetype;

    return NC_NOERR;
}

/*----< ncmpii_vars_create_filetype() >--------------------------------------*/
int
ncmpii_vars_create_filetype(NC               *ncp,
                            NC_var           *varp,
                            const MPI_Offset  start[],
                            const MPI_Offset  count[],
                            const MPI_Offset  stride[],
                            int               rw_flag,
                            int              *blocklen,
                            MPI_Offset       *offset_ptr,
                            MPI_Datatype     *filetype_ptr,
                            int              *is_filetype_contig)
{
    int          dim, status, err;
    MPI_Offset   offset, stride_off, nelems;
    MPI_Datatype filetype=MPI_BYTE;

    if (stride == NULL)
        return ncmpii_vara_create_filetype(ncp, varp, start, count, rw_flag,
                                           blocklen, offset_ptr, filetype_ptr,
                                           is_filetype_contig);

    /* check if all stride[] == 1 */
    for (dim=0; dim<varp->ndims && stride[dim]==1; dim++) ;
    if (dim == varp->ndims) /* all stride[] == 1, same as stride == NULL */
        return ncmpii_vara_create_filetype(ncp, varp, start, count, rw_flag,
                                           blocklen, offset_ptr, filetype_ptr,
                                           is_filetype_contig);

    /* now stride[] indicates a non-contiguous fileview */

    /* check whether start, count, stride are valid */
    status = NC_start_count_stride_ck(ncp, varp, start, count, stride, rw_flag);
    if (status != NC_NOERR) return status;

    /* calculate request amount */
    nelems = 1;
    for (dim=0; dim<varp->ndims; dim++) nelems *= count[dim];

    /* when nelems == 0 or varp is a scalar, i.e. varp->ndims == 0, no need to
     * create a filetype
     */
    if (varp->ndims == 0 || nelems == 0) {
        *offset_ptr   = varp->begin;
        *filetype_ptr = MPI_BYTE;
        if (blocklen           != NULL) *blocklen           = 0;
        if (is_filetype_contig != NULL) *is_filetype_contig = 1;
        return NC_NOERR;
    }

    /* hereinafter fileview is noncontiguous, i.e. filetype != MPI_BYTE.
     * Since we will construct a filetype, blocklen is set to 1.
     */
    if (blocklen           != NULL) *blocklen           = 1;
    if (is_filetype_contig != NULL) *is_filetype_contig = 0;
    offset = varp->begin;

    int ndims, *blockcounts, *blocklens;
    MPI_Aint *blockstride;
    MPI_Datatype tmptype;

    ndims       = varp->ndims;
    blockcounts = (int*) NCI_Malloc((size_t)ndims * 2 * SIZEOF_INT);
    blocklens   = blockcounts + ndims;

    blockstride = (MPI_Aint*) NCI_Malloc((size_t)ndims * SIZEOF_MPI_AINT);

    tmptype = MPI_BYTE;

    blockcounts[ndims-1] = (int)count[ndims-1];
    /* check 4-byte integer overflow (blockcounts in MPI_Type_hvector
       is of type int while count[] is of type MPI_Offset */
    if (count[ndims-1] != blockcounts[ndims-1]) {
        NCI_Free(blockstride);
        NCI_Free(blockcounts);
        DEBUG_RETURN_ERROR(NC_EINTOVERFLOW)
    }
    /* blocklens[] is unlikely a big value */
    blocklens[ndims-1] = varp->xsz;

    if (ndims == 1 && IS_RECVAR(varp)) {
#if SIZEOF_MPI_AINT != SIZEOF_MPI_OFFSET
        status = check_recsize_too_big(ncp);
        if (status != NC_NOERR) return status;
#endif
        stride_off = stride[ndims-1] * ncp->recsize;
        blockstride[ndims-1] = stride_off;
        offset += start[ndims-1] * ncp->recsize;
    } else {
        stride_off = stride[ndims-1] * varp->xsz;
        blockstride[ndims-1] = stride_off;
        offset += start[ndims-1] * varp->xsz;
    }
#if SIZEOF_MPI_AINT != SIZEOF_MPI_OFFSET
    /* check 4-byte integer overflow */
    if (stride_off != blockstride[ndims-1]) {
        if (blockstride) NCI_Free(blockstride);
        if (blockcounts) NCI_Free(blockcounts);
        DEBUG_RETURN_ERROR(NC_EINTOVERFLOW)
    }
#endif

    for (dim=ndims-1; dim>=0; dim--) {
#ifdef HAVE_MPI_TYPE_CREATE_HVECTOR
        err = MPI_Type_create_hvector(blockcounts[dim], blocklens[dim],
                                      blockstride[dim], tmptype, &filetype);
        if (err != MPI_SUCCESS) {
            NCI_Free(blockstride);
            NCI_Free(blockcounts);
            return ncmpii_handle_error(err, "MPI_Type_create_hvector");
        }
#else
        err = MPI_Type_hvector(blockcounts[dim], blocklens[dim],
                               blockstride[dim], tmptype, &filetype);
        if (err != MPI_SUCCESS) {
            NCI_Free(blockstride);
            NCI_Free(blockcounts);
            return ncmpii_handle_error(err, "MPI_Type_hvector");
        }
#endif
        MPI_Type_commit(&filetype);
        if (tmptype != MPI_BYTE)
            MPI_Type_free(&tmptype);
        tmptype = filetype;

        if (dim - 1 >= 0) {
            blocklens[dim-1]  = 1;
            blockcounts[dim-1] = (int)count[dim - 1];
            /* check 4-byte integer overflow */
            if (count[dim-1] != blockcounts[dim-1]) {
                NCI_Free(blockstride);
                NCI_Free(blockcounts);
                DEBUG_RETURN_ERROR(NC_EINTOVERFLOW)
            }

            if (dim-1 == 0 && IS_RECVAR(varp)) {
                stride_off = stride[dim-1] * ncp->recsize;
                blockstride[dim-1] = stride_off;
                offset += start[dim-1] * ncp->recsize;
            } else {
                stride_off = stride[dim-1] * varp->dsizes[dim] * varp->xsz;
                blockstride[dim-1] = stride_off;
                offset += start[dim-1] * varp->dsizes[dim] * varp->xsz;
            }
#if SIZEOF_MPI_AINT != SIZEOF_MPI_OFFSET
            /* check 4-byte integer overflow */
            if (stride_off != blockstride[dim-1]) {
                NCI_Free(blockstride);
                NCI_Free(blockcounts);
                DEBUG_RETURN_ERROR(NC_EINTOVERFLOW)
            }
#endif
        }
    }
    NCI_Free(blockstride);
    NCI_Free(blockcounts);

    *offset_ptr   = offset;
    *filetype_ptr = filetype;

    return NC_NOERR;
}

/*----< ncmpii_file_set_view() >---------------------------------------------*/
/* This function handles the special case for root process for setting its
 * file view: to keeps the whole file header visible to the root process.
 * This function is collective if called in collective data mode
 */
int
ncmpii_file_set_view(NC           *ncp,
                     MPI_File      fh,
                     MPI_Offset   *offset,
                     MPI_Datatype  filetype)
{
    int rank, err, mpireturn, status=NC_NOERR;

    if (filetype == MPI_BYTE) {
        /* filetype is a contiguous space, make the whole file visible */
        TRACE_IO(MPI_File_set_view)(fh, 0, MPI_BYTE, MPI_BYTE,
                                    "native", MPI_INFO_NULL);
        return NC_NOERR;
    }

    MPI_Comm_rank(ncp->nciop->comm, &rank);
    if (rank == 0) {
        /* prepend the whole file header to filetype */
        int blocklens[2];
        MPI_Aint disps[2];
        MPI_Datatype root_filetype, ftypes[2];

        /* first block is the header extent */
        blocklens[0] = (int)ncp->begin_var;
            disps[0] = 0;
           ftypes[0] = MPI_BYTE;

        /* check if header size > 2^31 */
        if (ncp->begin_var != blocklens[0]) DEBUG_ASSIGN_ERROR(status, NC_EINTOVERFLOW)

        /* second block is filetype, the subarray request(s) to the variable */
        blocklens[1] = 1;
            disps[1] = *offset;
           ftypes[1] = filetype;

#if SIZEOF_MPI_AINT != SIZEOF_MPI_OFFSET
        if (*offset != disps[1]) {
            blocklens[1] = 0;
            DEBUG_ASSIGN_ERROR(status, NC_EAINT_TOO_SMALL)
        }
#endif

#ifdef HAVE_MPI_TYPE_CREATE_STRUCT
        MPI_Type_create_struct(2, blocklens, disps, ftypes, &root_filetype);
#else
        MPI_Type_struct(2, blocklens, disps, ftypes, &root_filetype);
#endif
        MPI_Type_commit(&root_filetype);

        TRACE_IO(MPI_File_set_view)(fh, 0, MPI_BYTE, root_filetype,
                                        "native", MPI_INFO_NULL);
        MPI_Type_free(&root_filetype);

        /* now update the explicit offset to be used in MPI-IO call later */
        *offset = ncp->begin_var;
    }
    else {
        TRACE_IO(MPI_File_set_view)(fh, *offset, MPI_BYTE, filetype,
                                    "native", MPI_INFO_NULL);
        /* the explicit offset is already set in fileview */
        *offset = 0;
    }
    if (mpireturn != MPI_SUCCESS) {
        err = ncmpii_handle_error(mpireturn, "MPI_File_set_view");
        if (status == NC_NOERR) status = err;
    }

    return status;
}

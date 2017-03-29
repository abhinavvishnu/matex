/*
 *  Copyright (C) 2003, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id: macro.h 2196 2015-11-27 22:31:05Z wkliao $ */

#ifndef _H_MACRO
#define _H_MACRO

#ifndef MAX
#define MAX(mm,nn) (((mm) > (nn)) ? (mm) : (nn))
#endif
#ifndef MIN
#define MIN(mm,nn) (((mm) < (nn)) ? (mm) : (nn))
#endif

void *NCI_Malloc_fn(size_t size, int lineno, const char *func, const char *fname);
void *NCI_Calloc_fn(size_t nelem, size_t elsize, int lineno, const char *func, const char *fname);
void *NCI_Realloc_fn(void *ptr, size_t size, int lineno, const char *func, const char *fname);
void  NCI_Free_fn(void *ptr, int lineno, const char *func, const char *fname);

#define NCI_Malloc(a)    NCI_Malloc_fn(a,__LINE__,__func__,__FILE__)
#define NCI_Calloc(a,b)  NCI_Calloc_fn(a,b,__LINE__,__func__,__FILE__)
#define NCI_Realloc(a,b) NCI_Realloc_fn(a,b,__LINE__,__func__,__FILE__)
#define NCI_Free(a)      NCI_Free_fn(a,__LINE__,__func__,__FILE__)


#define CHECK_MPI_ERROR(mpi_errorcode, err_msg, nc_err) {                     \
    if (mpi_errorcode != MPI_SUCCESS) {                                       \
        char errorString[MPI_MAX_ERROR_STRING];                               \
        int rank, errorStringLen;                                             \
        MPI_Comm_rank(ncp->nciop->comm, &rank);                               \
        MPI_Error_string(mpi_errorcode, errorString, &errorStringLen);        \
        printf("%2d: MPI Failure at line %d of %s (%s : %s)\n",               \
               rank, __LINE__, __FILE__, err_msg, errorString);               \
        mpi_err = nc_err;                                                     \
    }                                                                         \
}

#ifdef PNC_DEBUG
#define DEBUG_RETURN_ERROR(err) {                                       \
    int _rank;                                                          \
    char *env_str = getenv("PNETCDF_VERBOSE_DEBUG_MODE");               \
    MPI_Comm_rank(MPI_COMM_WORLD, &_rank);                              \
    if (env_str != NULL && *env_str != '0')                             \
        fprintf(stderr, "Rank %d: %s error at line %d of %s in %s\n",   \
        _rank,ncmpii_err_code_name(err),__LINE__,__func__,__FILE__);    \
    return err;                                                         \
}
#define DEBUG_ASSIGN_ERROR(status, err) {                               \
    int _rank;                                                          \
    char *env_str = getenv("PNETCDF_VERBOSE_DEBUG_MODE");               \
    MPI_Comm_rank(MPI_COMM_WORLD, &_rank);                              \
    if (env_str != NULL && *env_str != '0')                             \
        fprintf(stderr, "Rank %d: %s error at line %d of %s in %s\n",   \
        _rank,ncmpii_err_code_name(err),__LINE__,__func__,__FILE__);    \
    status = err;                                                       \
}
#else
#define DEBUG_RETURN_ERROR(err) return err;
#define DEBUG_ASSIGN_ERROR(status, err) status = err;
#endif

#define GET_ONE_COUNT(count) {                                                \
    int _i;                                                                   \
    count = (MPI_Offset*) NCI_Malloc((size_t)varp->ndims * SIZEOF_MPI_OFFSET);\
    for (_i=0; _i<varp->ndims; _i++)                                          \
        count[_i] = 1;                                                        \
}

#ifdef ENABLE_SUBFILING
#define GET_FULL_DIMENSIONS(start, count) {                                   \
    int _i;                                                                   \
    int _ndims = (varp->num_subfiles>1?varp->ndims_org:varp->ndims);          \
    start = (MPI_Offset*) NCI_Malloc((size_t)_ndims*2*SIZEOF_MPI_OFFSET);     \
    count = start + _ndims;                                                   \
                                                                              \
    for (_i=0; _i<_ndims; _i++) {                                             \
        NC_dim *dimp;                                                         \
        int _dimid;                                                           \
        _dimid = (varp->num_subfiles>1)?varp->dimids_org[_i]:varp->dimids[_i];\
        dimp = ncmpii_elem_NC_dimarray(&ncp->dims, _dimid);                   \
        if (dimp->size == NC_UNLIMITED)                                       \
            count[_i] = NC_get_numrecs(ncp);                                  \
        else                                                                  \
            count[_i] = dimp->size;                                           \
        start[_i] = 0;                                                        \
    }                                                                         \
}
#else /* without subfiling */
#define GET_FULL_DIMENSIONS(start, count) {                                   \
    int _i=0;                                                                 \
    start = (MPI_Offset*) NCI_Malloc((size_t)varp->ndims*2*SIZEOF_MPI_OFFSET);\
    count = start + varp->ndims;                                              \
                                                                              \
    if (IS_RECVAR(varp)) { /* find current numrec if varp is record var */    \
        count[0] = NC_get_numrecs(ncp);                                       \
        start[0] = 0;                                                         \
        _i = 1;                                                               \
    }                                                                         \
    for (; _i<varp->ndims; _i++) {                                            \
        count[_i] = varp->shape[_i];                                          \
        start[_i] = 0;                                                        \
    }                                                                         \
}
#endif

#define DATATYPE_GET_CONVERT(vartype, inbuf, outbuf, cnelems, memtype, err) { \
    /* vartype is the variable's data type defined in the nc file             \
     * memtype is the I/O buffers data type (MPI_Datatype)  */                \
    switch(vartype) {                                                         \
        case NC_BYTE:                                                         \
            err = ncmpii_x_getn_schar(inbuf, outbuf, cnelems, memtype);       \
            break;                                                            \
        case NC_UBYTE:                                                        \
            err = ncmpii_x_getn_uchar(inbuf, outbuf, cnelems, memtype);       \
            break;                                                            \
        case NC_SHORT:                                                        \
            err = ncmpii_x_getn_short(inbuf, outbuf, cnelems, memtype);       \
            break;                                                            \
        case NC_USHORT:                                                       \
            err = ncmpii_x_getn_ushort(inbuf, outbuf, cnelems, memtype);      \
            break;                                                            \
        case NC_INT:                                                          \
            err = ncmpii_x_getn_int(inbuf, outbuf, cnelems, memtype);         \
            break;                                                            \
        case NC_UINT:                                                         \
            err = ncmpii_x_getn_uint(inbuf, outbuf, cnelems, memtype);        \
            break;                                                            \
        case NC_FLOAT:                                                        \
            err = ncmpii_x_getn_float(inbuf, outbuf, cnelems, memtype);       \
            break;                                                            \
        case NC_DOUBLE:                                                       \
            err = ncmpii_x_getn_double(inbuf, outbuf, cnelems, memtype);      \
            break;                                                            \
        case NC_INT64:                                                        \
            err = ncmpii_x_getn_int64(inbuf, outbuf, cnelems, memtype);       \
            break;                                                            \
        case NC_UINT64:                                                       \
            err = ncmpii_x_getn_uint64(inbuf, outbuf, cnelems, memtype);      \
            break;                                                            \
        default:                                                              \
            err = NC_EBADTYPE;                                                \
            break;                                                            \
    }                                                                         \
}

#define DATATYPE_PUT_CONVERT(vartype, outbuf, inbuf, cnelems, memtype, err) { \
    /* vartype is the variable's data type defined in the nc file             \
     * memtype is the I/O buffers data type (MPI_Datatype)  */                \
    switch(vartype) {                                                         \
        case NC_BYTE:                                                         \
            err = ncmpii_x_putn_schar(outbuf, inbuf, cnelems, memtype);       \
            break;                                                            \
        case NC_UBYTE:                                                        \
            err = ncmpii_x_putn_uchar(outbuf, inbuf, cnelems, memtype);       \
            break;                                                            \
        case NC_SHORT:                                                        \
            err = ncmpii_x_putn_short(outbuf, inbuf, cnelems, memtype);       \
            break;                                                            \
        case NC_USHORT:                                                       \
            err = ncmpii_x_putn_ushort(outbuf, inbuf, cnelems, memtype);      \
            break;                                                            \
        case NC_INT:                                                          \
            err = ncmpii_x_putn_int(outbuf, inbuf, cnelems, memtype);         \
            break;                                                            \
        case NC_UINT:                                                         \
            err = ncmpii_x_putn_uint(outbuf, inbuf, cnelems, memtype);        \
            break;                                                            \
        case NC_FLOAT:                                                        \
            err = ncmpii_x_putn_float(outbuf, inbuf, cnelems, memtype);       \
            break;                                                            \
        case NC_DOUBLE:                                                       \
            err = ncmpii_x_putn_double(outbuf, inbuf, cnelems, memtype);      \
            break;                                                            \
        case NC_INT64:                                                        \
            err = ncmpii_x_putn_int64(outbuf, inbuf, cnelems, memtype);       \
            break;                                                            \
        case NC_UINT64:                                                       \
            err = ncmpii_x_putn_uint64(outbuf, inbuf, cnelems, memtype);      \
            break;                                                            \
        default:                                                              \
            err = NC_EBADTYPE;                                                \
            break;                                                            \
    }                                                                         \
}

#endif

/*
 *  Copyright (C) 2015, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id: ncoffsets.c 2335 2016-03-03 14:55:12Z wkliao $ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>    /* strcpy() */
#include <errno.h>     /* errno, strerror() */
#include <sys/types.h> /* open() */
#include <sys/stat.h>  /* open() */
#include <fcntl.h>     /* open() */
#include <unistd.h>    /* read() */
#include <assert.h>    /* assert() */
#include <inttypes.h>  /* check for Endianness */

static int endianness;

#ifndef MIN
#define MIN(mm,nn) (((mm) < (nn)) ? (mm) : (nn))
#endif

static int verbose_debug;

#define DEBUG_RETURN_ERROR(err) {                             \
    if (verbose_debug)                                        \
        fprintf(stderr, "Error code %s at line %d of %s\n",   \
        ncmpii_err_code_name(err),__LINE__,__FILE__);         \
    return err;                                               \
}
#define DEBUG_ASSIGN_ERROR(status, err) {                     \
    if (verbose_debug)                                        \
        fprintf(stderr, "Error code %s at line %d of %s\n",   \
        ncmpii_err_code_name(err),__LINE__,__FILE__);         \
    status = err;                                             \
}

#define IS_RECVAR(vp) \
        ((vp)->shape != NULL ? (*(vp)->shape == NC_UNLIMITED) : 0 )

#define MALLOC_CHECK(ptr) { \
    if ((ptr) == NULL) { \
        fprintf(stderr, "Error at line %d: malloc out of memory",__LINE__); \
        exit(1); \
    } \
}

#define NC_NOERR 0
#define	NC_EINVAL	(-36)	/**< Invalid Argument */
#define NC_EBADDIM	(-46)	/**< Invalid dimension id or name */
#define NC_EUNLIMPOS	(-47)	/**< NC_UNLIMITED in the wrong index */
#define NC_ENOTNC	(-51)	/**< Not a netcdf file */
#define NC_EVARSIZE     (-62)   /**< One or more variable sizes violate format constraints */

#define NC_UNLIMITED 0L

typedef enum {
    NC_UNSPECIFIED =  0,
    NC_DIMENSION   = 10,
    NC_VARIABLE    = 11,
    NC_ATTRIBUTE   = 12
} NCtype;

#define NC_NAT          0       /**< Not A Type */
#define NC_BYTE         1       /**< signed 1 byte integer */
#define NC_CHAR         2       /**< ISO/ASCII character */
#define NC_SHORT        3       /**< signed 2 byte integer */
#define NC_INT          4       /**< signed 4 byte integer */
#define NC_LONG         NC_INT
#define NC_FLOAT        5       /**< single precision floating point number */
#define NC_DOUBLE       6       /**< double precision floating point number */
#define NC_UBYTE        7       /**< unsigned 1 byte int */
#define NC_USHORT       8       /**< unsigned 2-byte int */
#define NC_UINT         9       /**< unsigned 4-byte int */
#define NC_INT64        10      /**< signed 8-byte int */
#define NC_UINT64       11      /**< unsigned 8-byte int */

#define NC_DEFAULT_CHUNKSIZE 1048576

/* sizes of external data types */
#define X_SIZEOF_CHAR           1
#define X_SIZEOF_SHORT          2
#define X_SIZEOF_INT            4
#define X_SIZEOF_FLOAT          4
#define X_SIZEOF_DOUBLE         8
#define X_SIZEOF_UBYTE          1
#define X_SIZEOF_USHORT         2
#define X_SIZEOF_UINT           4
#define X_SIZEOF_LONGLONG       8
#define X_SIZEOF_ULONGLONG      8
#define X_SIZEOF_INT64          8
#define X_SIZEOF_UINT64         8

#define X_UINT_MAX	4294967295U

#define X_ALIGN                 4

/* useful for aligning memory */
#define _RNDUP(x, unit)  ((((x) + (unit) - 1) / (unit)) * (unit))
#define M_RND_UNIT       X_SIZEOF_DOUBLE
#define M_RNDUP(x)       _RNDUP(x, M_RND_UNIT)

#define ncmpix_len_char(nelems)   _RNDUP((nelems), X_ALIGN)
#define ncmpix_len_short(nelems)  (((nelems) + (nelems)%2)  * X_SIZEOF_SHORT)
#define ncmpix_len_int(nelems)    ((nelems) * X_SIZEOF_INT)
#define ncmpix_len_long(nelems)   ((nelems) * X_SIZEOF_LONG)
#define ncmpix_len_float(nelems)  ((nelems) * X_SIZEOF_FLOAT)
#define ncmpix_len_double(nelems) ((nelems) * X_SIZEOF_DOUBLE)
#define ncmpix_len_ubyte(nelems)  _RNDUP((nelems), X_ALIGN)
#define ncmpix_len_ushort(nelems) (((nelems) + (nelems)%2)  * X_SIZEOF_USHORT)
#define ncmpix_len_uint(nelems)   ((nelems) * X_SIZEOF_UINT)
#define ncmpix_len_int64(nelems)  ((nelems) * X_SIZEOF_INT64)
#define ncmpix_len_uint64(nelems) ((nelems) * X_SIZEOF_UINT64)

typedef int nc_type;

typedef struct {
    long long  nchars;
    char      *cp;     /* [nchars+1] one additional char for '\0' */
} NC_string;

typedef struct {
    NC_string *name;
    long long  size;
} NC_dim;

typedef struct NC_dimarray {
    int      nalloc;    /* number allocated >= ndefined */
    int      ndefined;  /* number of defined dimensions */
    NC_dim **value;
} NC_dimarray;

typedef struct {
    long long  xsz;      /* amount of space at xvalue (4-byte aligned) */
    NC_string *name;     /* name of the attributes */
    nc_type    type;     /* the discriminant */
    long long  nelems;   /* number of attribute elements */
    void      *xvalue;   /* the actual data, in external representation */
} NC_attr;

typedef struct NC_attrarray {
    int       nalloc;    /* number allocated >= ndefined */
    int       ndefined;  /* number of defined attributes */
    NC_attr **value;
} NC_attrarray;

typedef struct {
    int           xsz;    /* byte size of 1 array element */
    long long    *shape;  /* dim->size of each dim */
    long long    *dsizes; /* the right to left product of shape */
    NC_string    *name;   /* name of the variable */
    int           ndims;  /* number of dimensions */
    int          *dimids; /* array of dimension IDs */
    NC_attrarray  attrs;  /* attribute array */
    nc_type       type;   /* variable's data type */
    long long     len;    /* this is the "vsize" defined in header format, the
                             total size in bytes of the array variable.
                             For record variable, this is the record size */
    long long     begin;  /* starting file offset of this variable */
} NC_var;

typedef struct NC_vararray {
    int      nalloc;      /* number allocated >= ndefined */
    int      ndefined;    /* number of defined variables */
    int      num_rec_vars;/* number of defined record variables */
    NC_var **value;
} NC_vararray;

typedef struct NC {
    int           flags;
    char         *path;
    long long     xsz;      /* external size of this header, <= var[0].begin */
    long long     begin_var;/* file offset of the first (non-record) var */
    long long     begin_rec;/* file offset of the first 'record' */

    long long     recsize;  /* length of 'record': sum of single record sizes
                               of all the record variables */
    long long     numrecs;  /* number of 'records' allocated */
    NC_dimarray   dims;     /* dimensions defined */
    NC_attrarray  attrs;    /* global attributes defined */
    NC_vararray   vars;     /* variables defined */
} NC;

typedef struct bufferinfo {
    int        fd;
    off_t      offset;   /* current read/write offset in the file */
    int        version;  /* 1, 2, and 5 for CDF-1, 2, and 5 respectively */
    void      *base;     /* beginning of read/write buffer */
    void      *pos;      /* current position in buffer */
    long long  size;     /* size of the buffer */
} bufferinfo;

/*
 * "magic number" at beginning of file: 0x43444601 (big endian)
 */
static const char ncmagic1[] = {'C', 'D', 'F', 0x01};
static const char ncmagic2[] = {'C', 'D', 'F', 0x02};
static const char ncmagic5[] = {'C', 'D', 'F', 0x05};

const char * ncmpii_err_code_name(int err);

static int is_little_endian(void) {
    volatile uint32_t i=0x01234567;
    // return 0 for big endian, 1 for little endian.
    return (*((uint8_t*)(&i))) == 0x67;
}

static void
swap4b(int *val)
{
    unsigned int tmp = *val;
    char *ip = (char*) &tmp;
    char *op = (char*) val;
    op[0] = ip[3];
    op[1] = ip[2];
    op[2] = ip[1];
    op[3] = ip[0];
}

static void
swap8b(long long *val)
{
    unsigned long long tmp = *val;
    char *ip = (char*) &tmp;
    char *op = (char*) val;
    op[0] = ip[7];
    op[1] = ip[6];
    op[2] = ip[5];
    op[3] = ip[4];
    op[4] = ip[3];
    op[5] = ip[2];
    op[6] = ip[1];
    op[7] = ip[0];
}

static unsigned long long
get_uint8(bufferinfo *gbp) {
    unsigned long long tmp;
    memcpy(&tmp, gbp->pos, 8);
    if (endianness) swap8b(&tmp);
    gbp->pos += 8;
    return tmp;
}

static unsigned int
get_uint4(bufferinfo *gbp) {
    unsigned int tmp;
    memcpy(&tmp, gbp->pos, 4);
    if (endianness) swap4b(&tmp);
    gbp->pos += 4;
    return tmp;
}

static int
type_size(nc_type type) {
    switch (type) {
      case NC_BYTE:   return X_SIZEOF_CHAR;
      case NC_CHAR:   return X_SIZEOF_CHAR;
      case NC_SHORT:  return X_SIZEOF_SHORT;
      case NC_INT:    return X_SIZEOF_INT;
      case NC_FLOAT:  return X_SIZEOF_FLOAT;
      case NC_DOUBLE: return X_SIZEOF_DOUBLE;
      case NC_UBYTE:  return X_SIZEOF_UBYTE;
      case NC_USHORT: return X_SIZEOF_USHORT;
      case NC_UINT:   return X_SIZEOF_UINT;
      case NC_INT64:  return X_SIZEOF_INT64;
      case NC_UINT64: return X_SIZEOF_UINT64;
      default: return -1;
    }
}

static const char *
type_name(nc_type type) {
    switch (type) {
      case NC_BYTE:   return "byte";
      case NC_CHAR:   return "char";
      case NC_SHORT:  return "short";
      case NC_INT:    return "int";
      case NC_FLOAT:  return "float";
      case NC_DOUBLE: return "double";
      case NC_UBYTE:  return "ubyte";
      case NC_USHORT: return "ushort";
      case NC_UINT:   return "uint";
      case NC_INT64:  return "int64";
      case NC_UINT64: return "uint64";
      default: return "bogus";
    }
}

static NC_string *
ncmpii_new_NC_string(long long   slen,
                     const char *str)
{
    /* str may not be NULL terminated */
    NC_string *ncstrp;
    size_t sizeof_NC_string = M_RNDUP(sizeof(NC_string));
    size_t sz = slen + sizeof_NC_string + 1;
    /* one char more space for NULL terminate char */

    ncstrp = (NC_string *) calloc(sz, sizeof(char));
    if (ncstrp == NULL) return NULL;

    /* make space occupied by ncstrp->cp part of ncstrp */
    ncstrp->nchars = slen;
    ncstrp->cp = (char *)ncstrp + sizeof_NC_string;

    /* in PnetCDF, we want to make name->cp always NULL character terminated */
    if (str != NULL && *str != '\0') {
        strncpy(ncstrp->cp, str, slen);
        ncstrp->cp[slen] = '\0';  /* NULL terminated */
    }

    return(ncstrp);
}

static NC_dim *
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

/*----< ncmpix_len_nctype() >------------------------------------------------*/
static int
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
        default: assert("ncmpix_len_nctype bad type" == 0);
    }
    return 0;
}

static int
ncmpii_NC_var_shape64(NC                *ncp,
                      NC_var            *varp,
                      const NC_dimarray *dims)
{
    int i;
    long long product = 1;

    /* set the size of 1 element */
    varp->xsz = ncmpix_len_nctype(varp->type);

    if (varp->ndims == 0) goto out;

    /*
     * use the user supplied dimension indices to determine the shape
     */
    for (i=0; i<varp->ndims; i++) {
        const NC_dim *dimp;

        if (varp->dimids[i] < 0)
            DEBUG_RETURN_ERROR(NC_EBADDIM);

        if (varp->dimids[i] >= ((dims != NULL) ? dims->ndefined : 1))
            DEBUG_RETURN_ERROR(NC_EBADDIM);

        /* get the pointer to the dim object */
        dimp = ncmpii_elem_NC_dimarray(dims, varp->dimids[i]);
        varp->shape[i] = dimp->size;

        /* check for record variable, only the highest dimension can
         * be unlimited */
        if (varp->shape[i] == NC_UNLIMITED && i != 0)
            DEBUG_RETURN_ERROR(NC_EUNLIMPOS);
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
    if (ncp->flags != 5 && product >= X_UINT_MAX)
        DEBUG_RETURN_ERROR(NC_EVARSIZE);
     */

    /*
     * align variable size to 4 byte boundary, required by all netcdf file
     * formats
     */
    varp->len = product * varp->xsz;
    if (varp->len % 4 > 0)
        varp->len += 4 - varp->len % 4; /* round up */

    return NC_NOERR;
}

/*
 * Recompute the shapes of all variables
 * Sets ncp->begin_var to start of first variable.
 * Sets ncp->begin_rec to start of first record variable.
 * Returns -1 on error. The only possible error is an reference
 * to a non existent dimension, which would occur for a corrupt
 * netcdf file.
 */
static int
ncmpii_NC_computeshapes(NC *ncp)
{
    NC_var **vpp = (NC_var **)ncp->vars.value;
    NC_var *const *const end = &vpp[ncp->vars.ndefined];
    NC_var *first_var = NULL;       /* first "non-record" var */
    NC_var *first_rec = NULL;       /* first "record" var */
    int status;

    ncp->begin_var = ncp->xsz;
    ncp->begin_rec = ncp->xsz;
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
            ncp->begin_rec = (*vpp)->begin + (*vpp)->len;
        }
    }

    if (first_rec != NULL) {
        if (ncp->begin_rec > first_rec->begin)
            DEBUG_RETURN_ERROR(NC_ENOTNC); /* not a netCDF file or corrupted */

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
        DEBUG_RETURN_ERROR(NC_ENOTNC); /* not a netCDF file or corrupted */

    return NC_NOERR;
}

/*
 * To compute how much space will the xdr'd header take
 */

#define X_SIZEOF_NC_TYPE X_SIZEOF_INT
#define X_SIZEOF_NCTYPE X_SIZEOF_INT

/*----< hdr_len_NC_name() >--------------------------------------------------*/
static long long
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
    long long sz = sizeof_t; /* nelems */

    assert(ncstrp != NULL);

    if (ncstrp->nchars != 0)  /* namestring */
        sz += _RNDUP(ncstrp->nchars, X_ALIGN);

    return sz;
}

/*----< hdr_len_NC_dim() >---------------------------------------------------*/
static long long
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
    long long sz;

    assert(dimp != NULL);

    sz = hdr_len_NC_name(dimp->name, sizeof_t); /* name */
    sz += sizeof_t;                             /* dim_length */

    return sz;
}

/*----< hdr_len_NC_dimarray() >----------------------------------------------*/
static long long
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
    long long xlen;

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
static long long
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
    long long sz;

    assert(attrp != NULL);

    sz  = hdr_len_NC_name(attrp->name, sizeof_t); /* name */
    sz += X_SIZEOF_NC_TYPE;                       /* nc_type */
    sz += sizeof_t;                               /* nelems */
    sz += attrp->xsz;                             /* [values ...] */

    return sz;
}

/*----< hdr_len_NC_attrarray() >---------------------------------------------*/
static long long
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
    long long xlen;

    xlen = X_SIZEOF_NCTYPE;        /* NC_ATTRIBUTE */
    xlen += sizeof_t;              /* nelems */

    if (ncap == NULL) /* ABSENT: no attribute is defined */
        return xlen;

    for (i=0; i<ncap->ndefined; i++) /* [attr ...] */
        xlen += hdr_len_NC_attr(ncap->value[i], sizeof_t);

    return xlen;
}

/*----< hdr_len_NC_var() >---------------------------------------------------*/
static long long
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
    long long sz;

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
static long long
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
    long long xlen;

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
static long long
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
    long long xlen;

    assert(ncp != NULL);

    if (ncp->flags == 5) {        /* CDF-5 */
        sizeof_t     = X_SIZEOF_INT64; /* 8-byte integer for all integers */
        sizeof_off_t = X_SIZEOF_INT64; /* 8-byte integer for var begin */
    }
    else if (ncp->flags == 2) { /* CDF-2 */
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

static int
hdr_fetch(bufferinfo *gbp) {
    int err=NC_NOERR;
    size_t slack;        /* any leftover data in the buffer */

    assert(gbp->base != NULL);

    slack = gbp->size - (gbp->pos - gbp->base);
    /* if gbp->pos and gbp->base are the same, there is no leftover buffer
     * data to worry about.
     * In the other extreme, where gbp->size == (gbp->pos - gbp->base), then
     * all data in the buffer has been consumed */
    if (slack == gbp->size) slack = 0;

    memset(gbp->base, 0, (size_t)gbp->size);
    gbp->pos = gbp->base;

    lseek(gbp->fd, (gbp->offset)-slack, SEEK_SET);
    size_t read_amount = read(gbp->fd, gbp->base, gbp->size);
    if (read_amount == -1) {
        fprintf(stderr,"ERROR at line %d: read error %s\n",__LINE__,strerror(errno));
        exit(1);
    }
    /* we might have had to backtrack */
    gbp->offset += (gbp->size - slack);

    return err;
}

/*----< hdr_check_buffer() >--------------------------------------------------*/
/* Ensure that 'nextread' bytes are available.  */
static int
hdr_check_buffer(bufferinfo *gbp,
                 size_t      nextread)
{
    if (gbp->pos + nextread <= gbp->base + gbp->size)
        return NC_NOERR;

    /* read the next chunk from file */
    return hdr_fetch(gbp);
}

/*----< hdr_get_NCtype() >----------------------------------------------------*/
static int
hdr_get_NCtype(bufferinfo *gbp,
               NCtype     *typep)
{
    /* NCtype is 4-byte integer */
    int status = hdr_check_buffer(gbp, 4);
    if (status != NC_NOERR) return status;

    /* get a 4-byte integer */
    *typep = get_uint4(gbp);

    return NC_NOERR;
}

/*----< hdr_get_nc_type() >---------------------------------------------------*/
static int
hdr_get_nc_type(bufferinfo *gbp,
                nc_type    *typep)
{
    /* nc_type is 4-byte integer, X_SIZEOF_INT */
    int type, status;

    status = hdr_check_buffer(gbp, X_SIZEOF_INT);
    if (status != NC_NOERR) return status;

    type = get_uint4(gbp);

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
        DEBUG_RETURN_ERROR(NC_EINVAL);

    *typep = (nc_type) type;
    return NC_NOERR;
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
    size_t  nchars, nbytes, padding, bufremain, strcount;
    NC_string *ncstrp;
    char *cpos, pad[X_ALIGN-1];

    /* get nelems */
    if (gbp->version == 5) 
        nchars = get_uint8(gbp);
    else
        nchars = get_uint4(gbp);

    /* Allocate a NC_string structure large enough to hold nchars characters */
    ncstrp = ncmpii_new_NC_string(nchars, NULL);

    nbytes = nchars;
    padding = _RNDUP(ncstrp->nchars, X_ALIGN) - ncstrp->nchars;
    bufremain = gbp->size - (gbp->pos - gbp->base);
    cpos = ncstrp->cp;

    /* get namestring with padding */
    while (nbytes > 0) {
        if (bufremain > 0) {
            strcount = MIN(bufremain, nbytes);
            memcpy(cpos, gbp->pos, strcount);
            nbytes -= strcount;
            gbp->pos = (void *)((char *)gbp->pos + strcount);
            cpos += strcount;
            bufremain -= strcount;
        } else {
            status = hdr_fetch(gbp);
            if (status != NC_NOERR) {
                free(ncstrp);
                return status;
            }
            bufremain = gbp->size;
        }
    }

    /* handle the padding */
    if (padding > 0) {
        memset(pad, 0, X_ALIGN-1);
        if (memcmp(gbp->pos, pad, padding) != 0) {
            free(ncstrp);
            DEBUG_RETURN_ERROR(NC_EINVAL);
        }
        gbp->pos = (void *)((char *)gbp->pos + padding);
    }

    *ncstrpp = ncstrp;

    return NC_NOERR;
}

static NC_dim *
ncmpii_new_x_NC_dim(NC_string *name)
{
    NC_dim *dimp;

    dimp = (NC_dim *) malloc(sizeof(NC_dim));
    MALLOC_CHECK(dimp)

    dimp->name = name;
    dimp->size = 0;

    return(dimp);
}

/*----< hdr_get_NC_dim() >----------------------------------------------------*/
static int
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

    /* get dim_length */
    if (gbp->version == 5) 
        dimp->size = get_uint8(gbp);
    else
        dimp->size = get_uint4(gbp);

    *dimpp = dimp;
    return NC_NOERR;
}

static void
ncmpii_free_NC_dim(NC_dim *dimp)
{
    if (dimp == NULL) return;
    free(dimp->name);
    free(dimp);
}

static void
ncmpii_free_NC_dimarray(NC_dimarray *ncap)
{
    int i;

    assert(ncap != NULL);
    if (ncap->nalloc == 0) return;

    assert(ncap->value != NULL);
    for (i=0; i<ncap->ndefined; i++)
        ncmpii_free_NC_dim(ncap->value[i]);

    free(ncap->value);
    ncap->value    = NULL;
    ncap->nalloc   = 0;
    ncap->ndefined = 0;
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
    size_t ndefined;

    /* get NCtype (NC_DIMENSION) */
    status = hdr_get_NCtype(gbp, &type);
    if (status != NC_NOERR) return status;

    /* get nelems */
    if (gbp->version == 5) 
        ndefined = get_uint8(gbp);
    else
        ndefined = get_uint4(gbp);
    ncap->ndefined = (int)ndefined;

    if (ndefined == 0) {
        if (type != NC_DIMENSION && type != NC_UNSPECIFIED)
            DEBUG_RETURN_ERROR(NC_EINVAL);
    } else {
        if (type != NC_DIMENSION) DEBUG_RETURN_ERROR(NC_EINVAL);

        ncap->value = (NC_dim **) malloc(ndefined * sizeof(NC_dim*));
        MALLOC_CHECK(ncap->value)
        ncap->nalloc = (int)ndefined;

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
    size_t nbytes, padding, bufremain, attcount;

    nbytes = attrp->nelems * ncmpix_len_nctype(attrp->type);
    padding = attrp->xsz - nbytes;
    bufremain = gbp->size - (gbp->pos - gbp->base);

    /* get values */
    while (nbytes > 0) {
        if (bufremain > 0) {
            attcount = MIN(bufremain, nbytes);
            memcpy(value, gbp->pos, (size_t)attcount);
            nbytes -= attcount;
            gbp->pos = (void *)((char *)gbp->pos + attcount);
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
        if (memcmp(gbp->pos, pad, (size_t)padding) != 0)
            DEBUG_RETURN_ERROR(NC_EINVAL);
        gbp->pos = (void *)((char *)gbp->pos + padding);
    }

    return NC_NOERR;
}

static long long
ncmpix_len_NC_attrV(nc_type    type,
                    long long  nelems)
{
    switch(type) {
        case NC_BYTE:
        case NC_CHAR:
        case NC_UBYTE:  return ncmpix_len_char(nelems);
        case NC_SHORT:  return ncmpix_len_short(nelems);
        case NC_USHORT: return ncmpix_len_ushort(nelems);
        case NC_INT:    return ncmpix_len_int(nelems);
        case NC_UINT:   return ncmpix_len_uint(nelems);
        case NC_FLOAT:  return ncmpix_len_float(nelems);
        case NC_DOUBLE: return ncmpix_len_double(nelems);
        case NC_INT64:  return ncmpix_len_int64(nelems);
        case NC_UINT64: return ncmpix_len_uint64(nelems);
        default: assert("ncmpix_len_NC_attr bad type" == 0);
    }
    return 0;
}

static NC_attr *
ncmpii_new_x_NC_attr(NC_string  *strp,
                     nc_type     type,
                     size_t      nelems)
{
    NC_attr *attrp;
    const long long xsz = ncmpix_len_NC_attrV(type, nelems);
    size_t sz = M_RNDUP(sizeof(NC_attr));

    assert(!(xsz == 0 && nelems != 0));

    sz += (size_t)xsz;

    attrp = (NC_attr *) malloc(sz);
    MALLOC_CHECK(attrp)

    attrp->xsz    = xsz;
    attrp->name   = strp;
    attrp->type   = type;
    attrp->nelems = nelems;

    if (xsz != 0)
        attrp->xvalue = (char *)attrp + M_RNDUP(sizeof(NC_attr));
    else
        attrp->xvalue = NULL;

    return(attrp);
}

static void
ncmpii_free_NC_attr(NC_attr *attrp)
{
    if (attrp == NULL) return;
    free(attrp->name);
    free(attrp);
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
    size_t nelems;
    NC_attr *attrp;

    /* get name */
    status = hdr_get_NC_name(gbp, &strp);
    if (status != NC_NOERR) return status;

    /* get nc_type */
    status = hdr_get_nc_type(gbp, &type);
    if (status != NC_NOERR) {
        free(strp);
        return status;
    }

    /* get nelems */
    if (gbp->version == 5) 
        nelems = get_uint8(gbp);
    else
        nelems = get_uint4(gbp);

    /* allocate space for attribute object */
    attrp = ncmpii_new_x_NC_attr(strp, type, nelems);
    if (attrp == NULL) {
        free(strp);
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

static void
ncmpii_free_NC_attrarray(NC_attrarray *ncap)
{
    int i;

    assert(ncap != NULL);
    if (ncap->nalloc == 0) return;
    assert(ncap->value != NULL);
    for (i=0; i<ncap->ndefined; i++)
        ncmpii_free_NC_attr(ncap->value[i]);

    free(ncap->value);
    ncap->value    = NULL;
    ncap->nalloc   = 0;
    ncap->ndefined = 0;
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
    size_t ndefined;

    /* get NCtype (NC_ATTRIBUTE) */
    status = hdr_get_NCtype(gbp, &type);
    if (status != NC_NOERR) return status;

    /* get nelems */
    if (gbp->version == 5) 
        ndefined = get_uint8(gbp);
    else
        ndefined = get_uint4(gbp);
    ncap->ndefined = (int)ndefined;

    if (ndefined == 0) {
        if (type != NC_ATTRIBUTE && type != NC_UNSPECIFIED)
            DEBUG_RETURN_ERROR(NC_EINVAL);
    } else {
        if (type != NC_ATTRIBUTE)
            DEBUG_RETURN_ERROR(NC_EINVAL);

        ncap->value = (NC_attr **)malloc((size_t)ndefined * sizeof(NC_attr*));
        MALLOC_CHECK(ncap->value)
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

static NC_var *
ncmpii_new_x_NC_var(NC_string *strp,
                    int        ndims)
{
    NC_var *varp;
    int shape_space   = M_RNDUP(ndims * 8);
    int dsizes_space  = M_RNDUP(ndims * 8);
    int dimids_space  = M_RNDUP(ndims * 4);
    size_t sizeof_NC_var = M_RNDUP(sizeof(NC_var));
    size_t sz = sizeof_NC_var + (size_t)(shape_space + dsizes_space + dimids_space);

    varp = (NC_var *) malloc(sz);
    MALLOC_CHECK(varp)

    memset(varp, 0, sz);

    varp->name = strp;
    varp->ndims = ndims;

    if (ndims != 0) {
        varp->shape  = (long long *)((char *)varp + sizeof_NC_var);
        varp->dsizes = (long long *)((char *)varp->shape  + shape_space);
        varp->dimids = (int *)      ((char *)varp->dsizes + dsizes_space);
    }

    varp->xsz = 0;
    varp->len = 0;
    varp->begin = 0;
    return varp;
}

static void
ncmpii_free_NC_var(NC_var *varp)
{
    if (varp == NULL) return;
    ncmpii_free_NC_attrarray(&varp->attrs);
    free(varp->name);
    free(varp);
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
    size_t ndims=0, dim;
    NC_var *varp;

    /* get name */
    status = hdr_get_NC_name(gbp, &strp);
    if (status != NC_NOERR) return status;

    /* nelems */
    if (gbp->version == 5) 
        ndims = get_uint8(gbp);
    else
        ndims = get_uint4(gbp);

    /* allocate space for var object */
    varp = ncmpii_new_x_NC_var(strp, (int)ndims);

    /* get [dimid ...] */
    for (dim=0; dim<ndims; dim++) {
        status = hdr_check_buffer(gbp, (gbp->version == 5 ? 8 : 4));
        if (status != NC_NOERR) {
            ncmpii_free_NC_var(varp);
            return status;
        }
        if (gbp->version == 5) 
            varp->dimids[dim] = get_uint8(gbp);
        else
            varp->dimids[dim] = get_uint4(gbp);
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
    if (gbp->version == 5) 
        varp->len = get_uint8(gbp);
    else
        varp->len = get_uint4(gbp);

    /* next element is 'begin' */
    status = hdr_check_buffer(gbp, (gbp->version == 1 ? 4 : 8));
    if (status != NC_NOERR) {
        ncmpii_free_NC_var(varp);
        return status;
    }

    /* get begin */
    if (gbp->version == 1)
        varp->begin = get_uint4(gbp);
    else
        varp->begin = get_uint8(gbp);

    *varpp = varp;
    return NC_NOERR;
}

static void
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

    free(ncap->value);
    ncap->value    = NULL;
    ncap->nalloc   = 0;
    ncap->ndefined = 0;
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
    size_t ndefined;

    assert(gbp != NULL && gbp->pos != NULL);
    assert(ncap != NULL);
    assert(ncap->value == NULL);

    /* get NCtype (NC_VARIABLE) from gbp buffer */
    status = hdr_get_NCtype(gbp, &type);
    if (status != NC_NOERR) return status;

    /* get nelems (number of variables) from gbp buffer */
    if (gbp->version == 5) 
        ndefined = get_uint8(gbp);
    else
        ndefined = get_uint4(gbp);
    ncap->ndefined = (int)ndefined;

    if (ndefined == 0) { /* no variable defined */
        if (type != NC_VARIABLE && type != NC_UNSPECIFIED)
            DEBUG_RETURN_ERROR(NC_EINVAL);
    } else {
        if (type != NC_VARIABLE)
            DEBUG_RETURN_ERROR(NC_EINVAL);

        ncap->value = (NC_var **) malloc((size_t)ndefined * sizeof(NC_var*));
        MALLOC_CHECK(ncap->value)
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
static int
ncmpii_hdr_get_NC(int fd, NC *ncp)
{
    int i, status;
    bufferinfo getbuf;
    char *magic;

    /* Initialize the get buffer that stores the header read from the file */
    getbuf.fd     = fd;
    getbuf.offset = 0;   /* read from start of the file */
    getbuf.size   = NC_DEFAULT_CHUNKSIZE;
    getbuf.base   = (void *)malloc((size_t)getbuf.size);
    MALLOC_CHECK(getbuf.base)
    getbuf.pos    = getbuf.base;

    /* Fetch the next header chunk. The chunk is 'gbp->size' bytes big */
    status = hdr_fetch(&getbuf);
    if (status != NC_NOERR) return status;

    /* processing the header from getbuf, the get buffer */

    /* First get the file format information, magic */
    magic = getbuf.base;
    getbuf.pos += 4;

    /* don't need to worry about CDF-1 or CDF-2
     * if the first bits are not 'CDF'
     */
    if (memcmp(magic, ncmagic1, sizeof(ncmagic1)-1) != 0) {
        DEBUG_ASSIGN_ERROR(status, NC_ENOTNC)
        goto fn_exit;
    }

    /* check version number in last byte of magic */
    if (magic[sizeof(ncmagic1)-1] == 0x1) {
        getbuf.version = 1;
        ncp->flags = 1;
    } else if (magic[sizeof(ncmagic1)-1] == 0x2) {
        getbuf.version = 2;
        ncp->flags = 2;
    } else if (magic[sizeof(ncmagic1)-1] == 0x5) {
        getbuf.version = 5;
        ncp->flags = 5;
    } else {
        DEBUG_ASSIGN_ERROR(status, NC_ENOTNC); /* not an netCDF file */
        goto fn_exit;
    }

    /** Ensure that 'nextread' bytes (numrecs) are available. */
    status = hdr_check_buffer(&getbuf, (getbuf.version == 5) ? 8 : 4);
    if(status != NC_NOERR) goto fn_exit;

    /* get numrecs from getbuf into ncp */
    if (getbuf.version == 5) 
        ncp->numrecs = get_uint8(&getbuf);
    else
        ncp->numrecs = get_uint4(&getbuf);

    /* get dim_list from getbuf into ncp */
    status = hdr_get_NC_dimarray(&getbuf, &ncp->dims);
    if (status != NC_NOERR) goto fn_exit;

    /* get gatt_list from getbuf into ncp */
    status = hdr_get_NC_attrarray(&getbuf, &ncp->attrs);
    if (status != NC_NOERR) goto fn_exit;

    /* get var_list from getbuf into ncp */
    status = hdr_get_NC_vararray(&getbuf, &ncp->vars);
    if (status != NC_NOERR) goto fn_exit;

    /* get the un-aligned size occupied by the file header */
    ncp->xsz = ncmpii_hdr_len_NC(ncp);

    /* Recompute the shapes of all variables
     * Sets ncp->begin_var to start of first variable.
     * Sets ncp->begin_rec to start of first record variable.
     */
    status = ncmpii_NC_computeshapes(ncp);

    /* update the total number of record variables */
    ncp->vars.num_rec_vars = 0;
    for (i=0; i<ncp->vars.ndefined; i++)
        ncp->vars.num_rec_vars += IS_RECVAR(ncp->vars.value[i]);

fn_exit:
    free(getbuf.base);
    return status;
}

/*----< ncmpii_free_NC() >----------------------------------------------------*/
static void
ncmpii_free_NC(NC *ncp)
{
    if (ncp == NULL) return;
    ncmpii_free_NC_dimarray(&ncp->dims);
    ncmpii_free_NC_attrarray(&ncp->attrs);
    ncmpii_free_NC_vararray(&ncp->vars);
    free(ncp->path);
}

const char *
ncmpi_strerror(int err)
{
   switch(err)
   {
      case NC_NOERR:
	 return "No error";
      case NC_EINVAL:
	 return "NetCDF: Invalid argument";
      case NC_EBADDIM:
	 return "NetCDF: Invalid dimension ID or name";
      case NC_EUNLIMPOS:
	 return "NetCDF: NC_UNLIMITED in the wrong index";
      case NC_ENOTNC:
	 return "NetCDF: Unknown file format";
      case NC_EVARSIZE:
	 return "NetCDF: One or more variable sizes violate format constraints";
      default:
         printf("Unknown error code %d\n",err);
         return "Unknown error code";
   }
}

const char *
ncmpii_err_code_name(int err)
{
   switch(err)
   {
      case NC_NOERR:     return "NC_NOERR";
      case NC_EINVAL:    return "NC_EINVAL";
      case NC_EBADDIM:   return "NC_EBADDIM";
      case NC_EUNLIMPOS: return "NC_EUNLIMPOS";
      case NC_ENOTNC:    return "NC_ENOTNC";
      case NC_EVARSIZE : return "NC_EVARSIZE";
      default:
         printf("Unknown error code %d\n",err);
         return "Unknown error code";
   }
}

struct fspec {
    int nlvars;    /* Number of variables specified with -v
                    * option on command line */
    char** lvars;  /* list of variable names specified with -v
                    * option on command line */
    NC_var **varp;  /* [nlvars] */
    int    *varids; /* [nlvars] */
};

static void
make_lvars(char *optarg, struct fspec* fspecp)
{
    char *cp = optarg;
    int nvars = 1;
    char ** cpp;

    /* compute number of variable names in comma-delimited list */
    fspecp->nlvars = 1;
    while (*cp++)
        if (*cp == ',')
            nvars++;

    fspecp->lvars = (char **) malloc(nvars * sizeof(char*));
    MALLOC_CHECK(fspecp->lvars)

    cpp = fspecp->lvars;
    /* copy variable names into list */
    for (cp = strtok(optarg, ",");
         cp != NULL;
         cp = strtok((char *) NULL, ",")) {

        *cpp = (char *) malloc(strlen(cp) + 1);
        MALLOC_CHECK(*cpp)

        strcpy(*cpp, cp);
        cpp++;
    }
    fspecp->nlvars = nvars;
    fspecp->varp = (NC_var**) malloc(nvars * sizeof(NC_var*));
    MALLOC_CHECK(fspecp->varp)
}

static int
check_gap_in_fixed_vars(NC *ncp)
{
    int i, j;
    long long size, prev_end;
    NC_var *varp, *prev;

    /* check only fixed-size variables */
    for (i=1; i<ncp->vars.ndefined; i++) {
        varp = ncp->vars.value[i];

        if (IS_RECVAR(varp)) continue;

        /* calculate the size in bytes of this variable */
        size = type_size(varp->type);
        if (varp->ndims) size *= varp->dsizes[0];

        /* search for the previous fixed-size variable */
        prev = NULL;
        for (j=i-1; j>=0; j--) {
            if (!IS_RECVAR(ncp->vars.value[j])) {
                prev = ncp->vars.value[j];
                break;
            }
        }
        if (prev == NULL) /* first defined fixed-size variable */
            continue;

        /* not the first fixed-size variable */
        prev_end = prev->begin;
        if (prev->ndims == 0)
            prev_end += type_size(prev->type);
        else
            prev_end += type_size(prev->type) * prev->dsizes[0];

        /* check the gap between the begin of this variable from the end of
         * variable immediately before it */
        if (varp->begin - prev_end) return 1;
    }
    return 0;
}

static void
usage(char *cmd)
{
    char *help =
"Usage: %s [-h] | [-x] | [-sgr] [-v var1[,...]] file\n"
"       [-h]            Print help\n"
"       [-v var1[,...]] Output for variable(s) <var1>,... only\n"
"       [-s]            Output variable size. For record variables, output\n"
"                       the size of one record only\n"
"       [-g]            Output gap from the previous variable\n"
"       [-r]            Output offsets for all records\n"
"       [-x]            Check gaps in fixed-size variables, output 1 if gaps\n"
"                       are found, 0 for otherwise.\n"
"       file            Input netCDF file name\n"
"*Parallel netCDF library version 1.7.0\n";
    fprintf(stderr, help, cmd);
}

/*----< main() >-------------------------------------------------------------*/
int main(int argc, char *argv[])
{
    extern int optind;
    char *filename, *cmd, *env_str;
    int i, j, err, opt, nvars;
    int print_var_size=0, print_gap=0, check_gap=0, print_all_rec=0;
    NC *ncp;
    struct fspec *fspecp=NULL;

    fspecp = (struct fspec*) calloc(1, sizeof(struct fspec));
    cmd = (char*) malloc(strlen(argv[0])+1);
    strcpy(cmd,argv[0]);

    /* get command-line arguments */
    while ((opt = getopt(argc, argv, "v:sghqxr")) != EOF) {
        switch(opt) {
            case 'v': make_lvars (optarg, fspecp);
                      break;
            case 's': print_var_size = 1;
                      break;
            case 'g': print_gap = 1;
                      break;
            case 'r': print_all_rec = 1;
                      break;
            case 'x': check_gap = 1;
                      break;
            case 'h':
            default:  usage(cmd);
                      free(fspecp);
                      return 0;
        }
    }
    argc -= optind;
    argv += optind;
    if (argc != 1) {
        fprintf(stderr, "%s: missing file name\n", cmd);
        usage(cmd);
        if (fspecp->varp != NULL) free(fspecp->varp);
        for (i=0; i<fspecp->nlvars; i++)
            free(fspecp->lvars[i]);
        if (fspecp->lvars != NULL) free(fspecp->lvars);
        free(fspecp);
        return 1;
    }
    free(cmd);
    filename = argv[0]; /* required argument */

    verbose_debug = 0;
    env_str = getenv("PNETCDF_VERBOSE_DEBUG_MODE");
    if (env_str != NULL && *env_str != '0') verbose_debug = 1;

    /* find Endianness of the running machine */
    endianness = is_little_endian();

    /* open file */
    int fd = open(filename, O_RDONLY, 0666);
    if (fd == -1) {
        printf("Error: file open %s (%s)\n",filename,strerror(errno));
        exit(1);
    }

    ncp = (NC*) calloc(1, sizeof(NC));
    ncp->path = (char*) malloc(strlen(filename)+1);
    strcpy(ncp->path, filename);

    /* read the header from file */
    err = ncmpii_hdr_get_NC(fd, ncp);
    if (err != NC_NOERR) {
        fprintf(stderr,"Error: %s\n", ncmpii_err_code_name(err));
        exit(1);
    }

    if (check_gap) {
        int ret = check_gap_in_fixed_vars(ncp);
        ncmpii_free_NC(ncp);
        free(ncp);
        close(fd);
        printf("%d\n",ret);
        return 0;
    }

    /* First check if all selected variables can be found in input file. */
    if (fspecp->nlvars > 0) {
        /* print a selected list of variables */
        fspecp->varids = (int*) malloc(fspecp->nlvars * sizeof(int));
        MALLOC_CHECK(fspecp->varids)
        for (i=0; i<fspecp->nlvars; i++) {
            for (j=0; j<ncp->vars.ndefined; j++) {
                if (!strcmp(fspecp->lvars[i], ncp->vars.value[j]->name->cp)) {
                    fspecp->varp[i] = ncp->vars.value[j];
                    fspecp->varids[i] = j;
                    break;
                }
            }
            if (j == ncp->vars.ndefined) {
                printf("Error: variable %s not found\n",fspecp->lvars[i]);
                return 1;
            }
        }
    }

    /* print file name and format */
    printf("netcdf %s {\n",filename);
    printf("// file format: CDF-%d\n",ncp->flags);
    printf("\n");

    /* print file header size and extent */
    printf("file header:\n");
    printf("\tsize   = %lld bytes\n",ncp->xsz);
    printf("\textent = %lld bytes\n",ncp->begin_var);
    
    /* print dimensions */
    if (ncp->dims.ndefined > 0) printf("\ndimensions:\n");
    for (i=0; i<ncp->dims.ndefined; i++) {
        long long size;
        size = ncp->dims.value[i]->size;
        printf("\t%s = ",ncp->dims.value[i]->name->cp);
        if (size == NC_UNLIMITED)
            printf("UNLIMITED // (%lld currently)\n",ncp->numrecs);
        else
            printf("%lld\n",size);
    }

    if (fspecp->nlvars == 0) { /* print all variables */
        fspecp = (struct fspec*) malloc(sizeof(struct fspec));
        MALLOC_CHECK(fspecp)
        fspecp->varp = (NC_var**) malloc(ncp->vars.ndefined * sizeof(NC_var*));
        MALLOC_CHECK(fspecp->varp)
        fspecp->varids = (int*) malloc(ncp->vars.ndefined * sizeof(int));
        MALLOC_CHECK(fspecp->varids)
        fspecp->nlvars = ncp->vars.ndefined;
        fspecp->lvars = NULL;
        for (i=0; i<ncp->vars.ndefined; i++) {
            fspecp->varp[i] = ncp->vars.value[i];
            fspecp->varids[i] = i;
        }
    }

    /* find how many variables are record variables */
    int num_rec_vars = 0;
    int num_fix_vars = 0;
    for (i=0; i<fspecp->nlvars; i++) {
        if (IS_RECVAR(fspecp->varp[i])) num_rec_vars++;
        else                            num_fix_vars++;
    }

    int last_fix_varid = -1;
    for (i=0; i<ncp->vars.ndefined; i++) {
        if (IS_RECVAR(ncp->vars.value[i])) continue;
        last_fix_varid = i;
    }

    int first_rec_varid = -1;
    for (i=0; i<ncp->vars.ndefined; i++) {
        if (IS_RECVAR(ncp->vars.value[i])) {
            first_rec_varid = i;
            break;
        }
    }

    /* print fixed-size variables first */
    if (num_fix_vars) printf("\nfixed-size variables:\n");
    for (i=0; i<fspecp->nlvars; i++) {
        int j, ndims;
        char type_str[16], str[1024], line[1024];
        long long size;
        NC_var *varp = fspecp->varp[i];

        if (IS_RECVAR(varp)) continue;

        /* calculate the size in bytes of this variable */
        size = type_size(varp->type);
        if (varp->ndims) size *= varp->dsizes[0];

        line[0]='\0';
        sprintf(type_str,"%-6s", type_name(varp->type));
        sprintf(line,"%s", varp->name->cp);
        ndims = varp->ndims;
        if (ndims > 0) strcat(line,"(");
        for (j=0; j<ndims; j++) {
            NC_dim *dimp = ncp->dims.value[varp->dimids[j]];
            if (dimp->size == NC_UNLIMITED)
                size *= ncp->numrecs;
            sprintf(str, "%s%s", dimp->name->cp, j < ndims-1 ? ", " : ")");
            strcat(line, str);
        }

        /* print the data type, variable name, and its dimensions */
        printf("\t%6s %s:\n", type_str, line);

        /* print the starting and ending file offset of this variable */
        printf("\t       start file offset =%12lld\n", varp->begin);
        printf("\t       end   file offset =%12lld\n", varp->begin+size);

        /* print variable size in bytes */
        if (print_var_size)
            printf("\t       size in bytes     =%12lld\n", size);

        /* print the gap between the begin of this variable from the end of
         * variable immediately before it */
        if (print_gap) {
            NC_var *prev=NULL;
            for (j=fspecp->varids[i]-1; j>=0; j--) {
                /* search for the previous fixed-size variable */
                if (!IS_RECVAR(ncp->vars.value[j])) {
                    prev = ncp->vars.value[j];
                    break;
                }
            }
            if (fspecp->varids[i] == 0 || prev == NULL) {
                /* first defined fixed-size variable */
                printf("\t       gap from prev var =%12lld\n",
                       varp->begin - ncp->xsz);
            }
            else {
                /* not the first fixed-size variable */
                long long prev_end = prev->begin;
                if (prev->ndims == 0)
                    prev_end += type_size(prev->type);
                else
                    prev_end += type_size(prev->type) * prev->dsizes[0];
                printf("\t       gap from prev var =%12lld\n",
                       varp->begin - prev_end);
            }
        }
    }

    /* print record variables */
    if (num_rec_vars) printf("\nrecord variables:\n");
    for (i=0; i<fspecp->nlvars; i++) {
        int j, ndims;
        char type_str[16], str[1024], line[1024];
        long long var_begin, var_end, size, numrecs;
        NC_var *varp = fspecp->varp[i];

        if (!IS_RECVAR(varp)) continue;

        /* calculate the size in bytes of this variable */
        size = type_size(varp->type);
        if (varp->ndims) size *= varp->dsizes[0];

        line[0]='\0';
        sprintf(type_str,"%-6s", type_name(varp->type));
        sprintf(line,"%s", varp->name->cp);
        ndims = varp->ndims;
        if (ndims > 0) strcat(line,"(");
        for (j=0; j<ndims; j++) {
            NC_dim *dimp = ncp->dims.value[varp->dimids[j]];
            sprintf(str, "%s%s", dimp->name->cp, j < ndims-1 ? ", " : ")");
            strcat(line, str);
        }

        /* print the data type, variable name, and its dimensions */
        printf("\t%6s %s:\n", type_str, line);

        /* print the starting and ending file offset of this variable */
        numrecs = ncp->numrecs;
        var_begin = varp->begin;
        var_end   = varp->begin + size;
        if (print_all_rec == 0) numrecs = 1;
        for (j=0; j<numrecs; j++) {
            char rec_num[32];
                 if (j % 10 == 1) sprintf(rec_num, "%dst", j);
            else if (j % 10 == 2) sprintf(rec_num, "%dnd", j);
            else if (j % 10 == 3) sprintf(rec_num, "%drd", j);
            else                  sprintf(rec_num, "%dth", j);
            printf("\t       start file offset =%12lld    (%s record)\n", var_begin,rec_num);
            printf("\t       end   file offset =%12lld    (%s record)\n", var_end,rec_num);
            var_begin += ncp->recsize;
            var_end   += ncp->recsize;
        }

        /* print variable size in bytes (one record only)
         *   size *= ncp->numrecs;   (if for all records)
         */
        if (print_var_size)
            printf("\t       size in bytes     =%12lld    (of one record)\n", size);

        /* print the gap between the begin of this variable from the end of
         * variable immediately before it */
        if (print_gap) {
            NC_var *prev=NULL;
            long long prev_end;
            for (j=fspecp->varids[i]-1; j>=0; j--) {
                /* search for the previous record variable */
                if (IS_RECVAR(ncp->vars.value[j])) {
                    prev = ncp->vars.value[j];
                    break;
                }
            }
            if (fspecp->varids[i] == 0 && last_fix_varid == -1) {
                /* first record variable and no fixed-size variable */
                printf("\t       gap from prev var =%12lld\n",
                       varp->begin - ncp->xsz);
            }
            else if (fspecp->varids[i] == first_rec_varid) {
                /* first record variable and there are fixed-size variables */
                prev = ncp->vars.value[last_fix_varid];
                prev_end = prev->begin;
                if (prev->ndims == 0)
                    prev_end += type_size(prev->type);
                else
                    prev_end += type_size(prev->type) * prev->dsizes[0];
                printf("\t       gap from prev var =%12lld\n",
                       varp->begin - prev_end);
            }
            else {
                /* not the first record variable */
                prev_end = prev->begin;
                if (prev->ndims == 0)
                    prev_end += type_size(prev->type);
                else
                    prev_end += type_size(prev->type) * prev->dsizes[0];
                printf("\t       gap from prev var =%12lld\n",
                       varp->begin - prev_end);
            }
        }
    }
    printf("}\n");

    free(fspecp->varp);
    if (fspecp->lvars != NULL) {
        for (i=0; i<fspecp->nlvars; i++)
            free(fspecp->lvars[i]);
        free(fspecp->lvars);
    }
    free(fspecp->varids);
    free(fspecp);
    ncmpii_free_NC(ncp);
    free(ncp);
    close(fd);

    return 0;
}


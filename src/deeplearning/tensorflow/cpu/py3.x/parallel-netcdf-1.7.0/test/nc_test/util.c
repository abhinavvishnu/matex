/*
 *  Copyright (C) 2003, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *  $Id: util.c 2293 2016-01-06 03:43:04Z wkliao $
 */

#include <math.h> /* floor() */
#include "tests.h"

/* Prototypes */

void
print_nok(int nok)
{
    if (verbose || nfails > 0)
        print("\n");
    print("%4d good comparisons. ", nok);
}


/* Is value within external type range? */
int
inRange(const double value, const nc_type datatype)
{
    switch (datatype) {
        case NC_CHAR:   return value >= X_CHAR_MIN   && value <= X_CHAR_MAX;
        case NC_BYTE:   return value >= X_BYTE_MIN   && value <= X_BYTE_MAX;
        case NC_SHORT:  return value >= X_SHORT_MIN  && value <= X_SHORT_MAX;
        case NC_INT:    return value >= X_INT_MIN    && value <= X_INT_MAX;
        case NC_FLOAT:  return value >= X_FLOAT_MIN  && value <= X_FLOAT_MAX;
        case NC_DOUBLE: return value >= X_DOUBLE_MIN && value <= X_DOUBLE_MAX;
        case NC_UBYTE:  return value >= 0            && value <= X_UCHAR_MAX;
        case NC_USHORT: return value >= 0            && value <= X_USHORT_MAX;
        case NC_UINT:   return value >= 0            && value <= X_UINT_MAX;
        case NC_INT64:  return value >= X_INT64_MIN  && value <= X_INT64_MAX;
        case NC_UINT64: return value >= 0            && value <= X_UINT64_MAX;
        default:
            assert(0);
	    return(0);
    }
}

static int
inRange_uchar(const double value, const nc_type datatype)
{
    /* check value of type datatype if within uchar range */

    if (datatype == NC_BYTE) {
        /* netCDF specification make a special case for type conversion between
         * uchar and scahr: do not check for range error. See
         * http://www.unidata.ucar.edu/software/netcdf/docs_rc/data_type.html#type_conversion
         */
        return(value >= 0 && value <= 255);
        /* this is to ensure value is within the range of uchar */
    }
    /* else */
    return inRange(value, datatype);
}

static int
inRange_float(const double value, const nc_type datatype)
{
    double min, max;

    switch (datatype) {
        case NC_CHAR:   min = X_CHAR_MIN;   max = X_CHAR_MAX;  break;
        case NC_BYTE:   min = X_BYTE_MIN;   max = X_BYTE_MAX;  break;
        case NC_SHORT:  min = X_SHORT_MIN;  max = X_SHORT_MAX; break;
        case NC_INT:    min = X_INT_MIN;    max = X_INT_MAX;   break;
        case NC_FLOAT:
            if (FLT_MAX < X_FLOAT_MAX) {
                min = (-FLT_MAX);
                max = FLT_MAX;
            } else {
                min = X_FLOAT_MIN;
                max = X_FLOAT_MAX;
            }
            break;
        case NC_DOUBLE:
            if (FLT_MAX < X_DOUBLE_MAX) {
                min = (-FLT_MAX);
                max = FLT_MAX;
            } else {
                min = X_DOUBLE_MIN;
                max = X_DOUBLE_MAX;
            }
            break;
        case NC_UBYTE:  min = 0;            max = X_UCHAR_MAX;  break;
        case NC_USHORT: min = 0;            max = X_USHORT_MAX; break;
        case NC_UINT:   min = 0;            max = X_UINT_MAX;   break;
        case NC_INT64:  min = X_INT64_MIN;  max = X_INT64_MAX;  break;
        case NC_UINT64: min = 0;            max = X_UINT64_MAX; break;
        default:
            min = 0.0;
            max = 0.0;
            assert(0);
    }
    if (!( value >= min && value <= max)) {
#if 0    /* DEBUG */
        if (datatype == NC_FLOAT) {
            fprintf(stderr, "\n");
            fprintf(stderr, "min   % .17e\n", min);
            fprintf(stderr, "value % .17e\n", value);
            fprintf(stderr, "max   % .17e\n", max);
        }
#endif
        return 0;
    }
#if FLT_MANT_DIG != DBL_MANT_DIG
    /* else */
    {
        const float fvalue = value;
        return fvalue >= min && fvalue <= max;
    }
#else
    return 1;
#endif
}

/* wrapper for inRange to handle special NC_BYTE/uchar adjustment */
/* this function checks whether "value" to be casted to type "itype" is
 * within the range of external "datatype".
 */
int
inRange3(const double    value, 
         const nc_type   datatype,
         const nct_itype itype)
{
    /* netCDF specification make a special case for type conversion between
     * uchar and scahr: do not check for range error. See
     * http://www.unidata.ucar.edu/software/netcdf/docs_rc/data_type.html#type_conversion
     * The _uchar and _schar functions were introduced in netCDF-3 to eliminate
     * an ambiguity, and support both signed and unsigned byte data. In
     * netCDF-2, whether the external NC_BYTE type represented signed or
     * unsigned values was left up to the user. In netcdf-3, we treat NC_BYTE
     * as signed for the purposes of conversion to short, int, long, float, or
     * double. (Of course, no conversion takes place when the internal type is
     * signed char.) In the _uchar functions, we treat NC_BYTE as if it were
     * unsigned. Thus, no NC_ERANGE error can occur converting between NC_BYTE
     * and unsigned char.
     */
    switch (itype) {
        case NCT_UCHAR:
            return inRange_uchar(value, datatype);
        case NCT_FLOAT:
            return inRange_float(value, datatype);
        default:
            break;
    }
    return inRange(value, datatype);
}


/* 
 *  Does x == y, where one is internal and other external (netCDF)?  
 *  Use tolerant comparison based on IEEE FLT_EPSILON or DBL_EPSILON.
 */
int
equal(double x, 
      double y, 
      nc_type      extType,     /* external data type */
      nct_itype    itype)
{
    const double flt_epsilon = 1.19209290E-07;
    const double dbl_epsilon = 2.2204460492503131E-16;
    double epsilon;

    epsilon = extType == NC_FLOAT ||
              itype == NCT_FLOAT ? flt_epsilon : dbl_epsilon;

    if (extType == NC_CHAR && itype == NCT_TEXT) {
        /* because in-memory data type char can be signed or unsigned,
         * type cast the value from external NC_CHAR before the comparison
         */
        char x2 = (char) x;
        char y2 = (char) y;
        x = x2;
        y = y2;
    }
    return ABS(x-y) <= epsilon * MAX( ABS(x), ABS(y));
}

/* this function is for the APIs without itype, i.e. extType == itype */
int
equal2(
    double x,
    double y,
    nc_type extType)    /* external data type */
{
    const double flt_epsilon = 1.19209290E-07;
    const double dbl_epsilon = 2.2204460492503131E-16;
    double epsilon;

    epsilon = extType == NC_FLOAT ? flt_epsilon : dbl_epsilon;

    if (extType == NC_CHAR) {
        /* because in-memory data type char can be signed or unsigned,
         * type cast the value from external NC_CHAR before the comparison
         */
        char x2 = (char) x;
        char y2 = (char) y;
        x = x2;
        y = y2;
    }

    return ABS(x-y) <= epsilon * MAX( ABS(x), ABS(y));
}

/* Test whether two int vectors are equal. If so return 1, else 0  */
int
int_vec_eq(const int *v1, const int *v2, const int n)
{
    int i;
    for (i= 0; i < n && v1[i] == v2[i]; i++)
    ;
    return i == n;
}


/*
 *  Generate random integer from 0 to n-1
 *  Like throwing an n-sided dice marked 0, 1, 2, ..., n-1
 */
int roll( int n )
{
    int  r;

    do
    /*
     * Compute a pseudo-random value between 0.0 and 1.0, multiply
     * it by n-1, and then find the nearest integer.
     *
     * We don't use RAND_MAX here because not all compilation
     * environments define it (e.g. gcc(1) under SunOS 4.1.4).
     */
    r = ((rand() % 32768) / 32767.0) * (n - 1) + 0.5;
    while (r >= n);

    return r;
}


/*
 *      Convert number to mixed base
 *
 *      E.g. to convert 41 inches to yards, feet and inches:
 *      MPI_Offset base[] = {1, 3, 12};
 *      MPI_Offset result[3];
 *      status = toMixedBase(41, 3, base, result);
 *
 *      Author: Harvey Davies, Unidata/UCAR, Boulder, Colorado
 *
 *      wkliao: this is like finding the N-dimensional array index
 *              given a linearized length. For example, given a 2x3x4 array,
 *              if number==37, then the result=(2,2,5)
 *              because 37 = 2x(3x4) + 2x(4) + 5
 */
int
toMixedBase(size_t           number,   /* to be converted to mixed base */
            size_t           length,
            const MPI_Offset base[],   /* in:  [length], base[0] ignored */
            MPI_Offset       result[]) /* out: [length] */
{
    size_t i;

    if (length > 0) {
        for (i = length - 1; i > 0; i--) {
            if (base[i] == 0) return 1;
            result[i] = number % base[i];
            number = number / base[i];
        }
        result[0] = number;
    }
    return 0;
}

/*
 *      Convert number from mixed base
 *
 *      E.g. to convert 1 yard, 0 feet, 5 inches to inches:
 *      MPI_Offset number[] = {1, 0, 5};
 *      MPI_Offset base[] = {1, 3, 12};
 *      inches = fromMixedBase(3, number, base);
 *
 *      Author: Harvey Davies, Unidata/UCAR, Boulder, Colorado
 */
size_t
fromMixedBase(size_t     length,
              MPI_Offset number[],  /* [length] */
              MPI_Offset base[])    /* [length], base[0] ignored */
{
    size_t i;
    size_t result = 0;

    for (i = 1; i < length; i++) {
        result += number[i-1];
        result *= base[i];
    }
    if (length > 0)
        result += number[i-1];
    return result;
}


/* Convert any nc_type to double */
int nc2dbl ( const nc_type datatype, const void *p, double *result)
{
    if ( ! p ) return 2;
    if ( ! result ) return 3;
    switch (datatype) {
        case NC_CHAR:   *result = *((signed char *)    p); break;
        case NC_BYTE:   *result = *((signed char *)    p); break;
        case NC_UBYTE:  *result = *((unsigned char *)  p); break;
        case NC_SHORT:  *result = *((short *)          p); break;
        case NC_USHORT: *result = *((unsigned short *) p); break;
        case NC_INT:
#if INT_MAX >= X_INT_MAX
            *result = *((int *) p); break;
#else
            *result = *((long *) p); break;
#endif
        case NC_UINT:
#if UINT_MAX >= X_UINT_MAX
            *result = *((unsigned int *) p); break;
#else
            *result = *((unsigned long *) p); break;
#endif
        case NC_FLOAT:  *result = *((float *)              p); break;
        case NC_DOUBLE: *result = *((double *)             p); break;
        case NC_INT64:  *result = *((long long *)          p); break;
        case NC_UINT64: *result = *((unsigned long long *) p); break;
        default: return 1;
    }
    return 0;
}


/* Convert double to any nc_type */
int dbl2nc ( const double d, const nc_type datatype, void *p)
{
    double r;   /* rounded value */

    if (p == NULL) return 1;
    switch (datatype) {
        case NC_CHAR:
            r = floor(0.5+d);
            /* d is obtained from hash() which may be set to X_CHAR_MIN (0)
             * or X_CHAR_MAX (255). When in-memory data type char is signed
             * (i.e. ranged from -128 to 127), we should still allow a type
             * cast a unsigned value > 127 to a signed char without reporting
             * it as a range error.
             */
            if ( r < X_CHAR_MIN || r > X_CHAR_MAX ) return 2;
#if defined(__CHAR_UNSIGNED__) && __CHAR_UNSIGNED__ != 0
            *((signed char*) p) = r;
#else
            *((char   *) p) = r;
#endif
            break;
        case NC_BYTE:
            r = floor(0.5+d);
            if ( r < schar_min  ||  r > schar_max )  return 2;
            *((signed char *) p) = r;
            break;
        case NC_UBYTE:
            r = floor(0.5+d);
            if ( r < 0.0  ||  r > uchar_max )  return 2;
            *((unsigned char *) p) = r;
            break;
        case NC_SHORT:
            r = floor(0.5+d);
            if ( r < short_min  ||  r > short_max )  return 2;
            *((short *) p) = r;
            break;
        case NC_USHORT:
            r = floor(0.5+d);
            if ( r < 0.0  ||  r > ushort_max )  return 2;
            *((unsigned short *) p) = r;
            break;
        case NC_INT:
            r = floor(0.5+d);
            if ( r < long_min  ||  r > long_max )  return 2;
#if INT_MAX >= X_INT_MAX
            *((int   *) p) = r;
#else
            *((long   *) p) = r;
#endif
            break;
        case NC_UINT:
            r = floor(0.5+d);
            if ( r < 0.0  ||  r > uint_max )  return 2;
#if UINT_MAX >= X_UINT_MAX
            *((unsigned int  *) p) = r;
#else
            *((unsigned long *) p) = r;
#endif
            break;
        case NC_FLOAT:
            if ( fabs(d) > float_max )  return 2;
            *((float  *) p) = d;
            break;
        case NC_DOUBLE:
            *((double *) p) = d;
            break;
        case NC_INT64:
            r = floor(0.5+d);
            if ( r < int64_min  ||  r > int64_max )  return 2;
            *((long long *) p) = r;
            break;
        case NC_UINT64:
            r = floor(0.5+d);
            if ( r < 0.0  ||  r > uint64_max )  return 2;
            *((unsigned long long *) p) = r;
            break;
        default:
            return 1;
    }
    return 0;
}


#define FUZZ (1.19209290E-07)

/* Generate data values as function of type, rank (-1 for attribute), index */
double
hash( const nc_type type, const int rank, const MPI_Offset *index ) 
{
    double base;
    double result = 0.0;
    int  d;       /* index of dimension */

    /* If vector then elements 0 & 1 are min & max. Elements 2 & 3 are */
    /* just < min & > max (except for NC_CHAR & NC_DOUBLE) */
    if (abs(rank) == 1 && index[0] <= 3) {
        switch (index[0]) {
            case 0:
                switch (type) {  /* test if can get/put MIN value */
                    case NC_CHAR:   return X_CHAR_MIN;
                    case NC_BYTE:   return X_BYTE_MIN;
                    case NC_SHORT:  return X_SHORT_MIN;
                    case NC_INT:    return X_INT_MIN;
                    case NC_FLOAT:  return X_FLOAT_MIN;
                    case NC_DOUBLE: return X_DOUBLE_MIN;
                    case NC_UBYTE:  return 0;
                    case NC_USHORT: return 0;
                    case NC_UINT:   return 0;
                    case NC_INT64:  return X_INT_MIN - 128.0; /* slight smaller
                                                                 than INT_MIN */
                    case NC_UINT64: return 0;
                    default:  assert(0);
                }
            case 1:
                switch (type) {  /* test if can get/put MAX value */
                    case NC_CHAR:   return X_CHAR_MAX;
                    case NC_BYTE:   return X_BYTE_MAX;
                    case NC_SHORT:  return X_SHORT_MAX;
                    case NC_INT:    return X_INT_MAX;
                    case NC_FLOAT:  return X_FLOAT_MAX;
                    case NC_DOUBLE: return X_DOUBLE_MAX;
                    case NC_UBYTE:  return X_UCHAR_MAX;
                    case NC_USHORT: return X_USHORT_MAX;
                    case NC_UINT:   return X_UINT_MAX;
                    case NC_INT64:  return X_INT_MAX + 128.0;
                                    /* slightly bigger than INT_MAX */
                    case NC_UINT64: return X_UINT_MAX + 128.0;
                                    /* slightly bigger than UINT_MAX */
                    default:  assert(0);
                }
            case 2:
                switch (type) {  /* test if can detect out-of-boundary value */
                    case NC_CHAR:   return 'A';
                    case NC_BYTE:   return X_BYTE_MIN-1.0;
                    case NC_SHORT:  return X_SHORT_MIN-1.0;
                    case NC_INT:    return X_INT_MIN-1.0;
                    case NC_FLOAT:  return X_FLOAT_MIN * (1.0 + FUZZ);
                    case NC_DOUBLE: return -1.0;  /* skip test */
                    case NC_UBYTE:  return -1.0;
                    case NC_USHORT: return -1.0;
                    case NC_UINT:   return -1.0;
                    case NC_INT64:  return -1.0;  /* skip test */
                    case NC_UINT64: return -1.0;
                    default:  assert(0);
                }
            case 3:
                switch (type) {  /* test if can detect out-of-boundary value */
                    case NC_CHAR:   return 'Z';
                    case NC_BYTE:   return X_BYTE_MAX  +1.0;
                    case NC_SHORT:  return X_SHORT_MAX +1.0;
                    case NC_INT:    return X_INT_MAX   +1.0;
                    case NC_FLOAT:  return X_FLOAT_MAX * (1.0 + FUZZ);
                    case NC_DOUBLE: return 1.0;    /* skip test */
                    case NC_UBYTE:  return X_UCHAR_MAX +1.0;
                    case NC_USHORT: return X_USHORT_MAX+1.0;
                    case NC_UINT:   return X_UINT_MAX  +1.0;
                    case NC_INT64:  return 1.0;    /* skip test */
                    case NC_UINT64: return 1.0;    /* skip test */
                    default:  assert(0);
                }
        }
    } else { /* for array with more than 4 elements, pick some random numbers */
        /* wkliao: what is "base" ? */
        switch (type) {
            case NC_CHAR:   base =  2;  break;
            case NC_BYTE:   base = -2;  break;
            case NC_SHORT:  base = -5;  break;
            case NC_INT:    base = -20; break;
            case NC_FLOAT:  base = -9;  break;
            case NC_DOUBLE: base = -10; break;

            /* not sure what right values are */
            case NC_UBYTE:   base =   2;  break;
            case NC_USHORT:  base =   5;  break;
            case NC_UINT:    base =  20;  break;
            case NC_INT64:   base = -20;  break;
            case NC_UINT64:  base =  20;  break;
            default:
                base = 0;
                assert(0);
        }
        if (rank < 0)  /* attribute */
            result = base * 7;
        else
            result = base * (rank + 1);

        for (d=0; d<abs(rank); d++)
            result = base * (result + index[d]);
    }
    return result;
}

/* wrapper for hash to handle special NC_BYTE/uchar adjustment */
double
hash4(const nc_type     type, 
      const int         rank, 
      const MPI_Offset *index, 
      const nct_itype   itype)
{
    double result;

    result = hash(type, rank, index);
    if (itype == NCT_UCHAR && type == NC_BYTE && result >= -128 && result < 0)
        result += 256;
    return result;
}

static nc_type
char2type(char letter) {
    switch (letter) {
        case 'c': return NC_CHAR;
        case 'b': return NC_BYTE;
        case 's': return NC_SHORT;
        case 'i': return NC_INT;
        case 'f': return NC_FLOAT;
        case 'd': return NC_DOUBLE;
        case 'y': return NC_UBYTE;
        case 't': return NC_USHORT;
        case 'u': return NC_UINT;
        case 'x': return NC_INT64;
        case 'z': return NC_UINT64;
        default:  assert(0);
    }
    return NC_CHAR;  /* Just to keep compiler happy */
}


static void
init_dims(const char *digit)
{
    int dimid;    /* index of dimension */

    for (dimid = 0; dimid < NDIMS; dimid++) {
        dim_len[dimid] = dimid == 0 ? NRECS : dimid;
        /* lengths = 1, 2, 3, ...
         * names = Dr, D1, D1, D2, ...
         */
        dim_name[dimid][0] = 'D';
        dim_name[dimid][1] = digit[dimid];
        dim_name[dimid][2] = '\0';
    }
}

static void
init_gatts(const char *type_letter)
{
    int attid;
    for (attid = 0; attid < numGatts; attid++) {
        gatt_name[attid][0] = 'G';
        gatt_name[attid][1] = type_letter[attid];
        gatt_name[attid][2] = '\0';
        gatt_len[attid]     = 1 + attid;
        gatt_type[attid]    = char2type (type_letter[attid]);
    }
}

static MPI_Offset
product(size_t nn, const MPI_Offset *sp)
{
    MPI_Offset result = 1;
    while (nn-- > 0)
        result *= *sp++;
    return result;  /* == sp[0] * sp[1] * ... * sp[nn-1] */
}

/* 
   define global variables:
   dim_name, dim_len, 
   var_name, var_type, var_rank, var_shape, var_natts, var_dimid, var_nels
   att_name, gatt_name, att_type, gatt_type, att_len, gatt_len
 */
void
init_gvars (void)
{
    const MPI_Offset max_dim_len[MAX_RANK] = {MAX_DIM_LEN +1,
                                              MAX_DIM_LEN,
                                              MAX_DIM_LEN };
    const char type_letter[] = "cbsifdytuxz";
    /* c:char, b:byte, s:short, i:int, f:float, d:double, y:ubyte, t:ushort,
     * u:uint, x:int64, z:uint64
     */
    const char digit[] = "r123456789";

    size_t rank;
    int vn;           /* var number */
    int xtype;        /* index of type */
    int an;           /* attribute number */

    assert(sizeof(max_dim_len)/sizeof(max_dim_len[0]) >= MAX_RANK);

    init_dims(digit);

    for (vn=0; vn<numVars; vn++)
        memset(var_name[vn], 0, 2+MAX_RANK);

    vn    = 0;
    xtype = 0;
    an    = 0;
    for (rank=0; rank<=MAX_RANK; rank++) { /* dimension ranks */
        /* number variables of a type and rank
         * == max_dim_len[0] * max_dim_len[1] * ... * max_dim_len[rank-1] */
        const size_t nvars = product(rank, max_dim_len);

        int jj;
        for (jj=0; jj<nvars; jj++) {
            /* number types of this shape */
            const int ntypes = rank < 2 ? numTypes : 1;

            int ac, tc;
            for (tc=0; tc<ntypes; tc++, vn++, xtype = (xtype + 1) % numTypes) {
                MPI_Offset tmp[MAX_RANK];

                var_name[vn][0] = type_letter[xtype];
                var_type[vn]    = char2type(type_letter[xtype]);
                var_rank[vn]    = rank;
                var_natts[vn]   = rank == 0 ? vn % (MAX_NATTS + 1) : 0;

                /* ac block */
                for (ac=0; ac<var_natts[vn]; ac++, an++) {
                     att_name[vn][ac][0] = type_letter[an % numTypes];
                     att_name[vn][ac][1] = '\0';
                     att_len[vn][ac]     = an;
                     att_type[vn][ac]    = char2type(type_letter[an % numTypes]);
                }
#ifndef NDEBUG
                assert(toMixedBase(jj, rank, max_dim_len, tmp) == 0);
#else
                toMixedBase(jj, rank, max_dim_len, tmp);
#endif
                /* dn block */
                int dn; /* dimension number */
                for (dn=0; dn<rank; dn++)
                     var_dimid[vn][dn] = (int)tmp[dn];

                for (dn=0, var_nels[vn]=1; dn<rank; dn++) {
                     var_dimid[vn][dn]   += dn > 0;
                     assert (var_dimid[vn][dn] <= 9);
                     var_name[vn][dn + 1] = digit[var_dimid[vn][dn]];
                     var_shape[vn][dn]    = var_dimid[vn][dn] ?
                                            var_dimid[vn][dn] : NRECS;
                     var_nels[vn]        *= var_shape[vn][dn];
                }
            }
        }
    }
    init_gatts(type_letter);
}


/* define dims defined by global variables */
void                                                        
def_dims(int ncid)
{
    int  i, err, dimid;

    for (i=0; i<NDIMS; i++) {
        err = ncmpi_def_dim(ncid, dim_name[i],
                            i==0 ? NC_UNLIMITED : dim_len[i], &dimid);
        IF (err != NC_NOERR) error("ncmpi_def_dim: %s", ncmpi_strerror(err));
    }
}


/* define vars defined by global variables */
void                                                        
def_vars(int ncid)
{
    int i, err, var_id;

    for (i = 0; i < numVars; i++) {
        err = ncmpi_def_var(ncid, var_name[i], var_type[i], var_rank[i],
                            var_dimid[i], &var_id);
        IF (err != NC_NOERR) error("ncmpi_def_var: %s", ncmpi_strerror(err));
    }
}


/* put attributes defined by global variables */
void                                                        
put_atts(int ncid)
{
    int  i, j, allInRange, err;
    MPI_Offset  k;
    char catt[MAX_NELS];
    double att[MAX_NELS];

    for (i=-1; i<numVars; i++) {  /* -1: global attribute */
        for (j=0; j<NATTS(i); j++) {
            if (ATT_TYPE(i,j) == NC_CHAR) {
                for (k=0; k<ATT_LEN(i,j); k++) {
                    catt[k] = hash(ATT_TYPE(i,j), -1, &k);
                }
                err = ncmpi_put_att_text(ncid, i, ATT_NAME(i,j),
                                         ATT_LEN(i,j), catt);
                IF (err != NC_NOERR) 
                    error("ncmpi_put_att_text: %s", ncmpi_strerror(err));
            } else {
                for (allInRange=1, k=0; k<ATT_LEN(i,j); k++) {
                    att[k] = hash(ATT_TYPE(i,j), -1, &k);
                    allInRange = allInRange && inRange(att[k], ATT_TYPE(i,j));
                }
                err = ncmpi_put_att_double(ncid, i, ATT_NAME(i,j),
                                           ATT_TYPE(i,j), ATT_LEN(i,j), att);
                if (allInRange) {
                    IF (err != NC_NOERR)
                        error("ncmpi_put_att_double: %s", ncmpi_strerror(err));
                } else {
                    /* when ATT_LEN(i,j) >= 3, some values returned for 3rd,
                     * 4th, ... will be specially made to cause NC_ERANG
                     */
                    IF (err != NC_ERANGE)
                        error("type-conversion range error: status = %d", err);
                }
            }
        }
    }
}

/* put variables defined by global variables */
void                                                        
put_vars(int ncid)
{
    MPI_Offset start[MAX_RANK];
    MPI_Offset index[MAX_RANK];
    int  i, j, err, allInRange;
    double value[MAX_NELS];
    char text[MAX_NELS];

    for (j = 0; j < MAX_RANK; j++) start[j] = 0;
    for (i = 0; i < numVars; i++) {
        for (allInRange = 1, j = 0; j < var_nels[i]; j++) {
            err = toMixedBase(j, var_rank[i], var_shape[i], index);
            IF (err != NC_NOERR) error("toMixedBase");
            if (var_name[i][0] == 'c') { /* var_type[i] is NC_CHAR */
                text[j] = hash(var_type[i], var_rank[i], index);
            } else {
                value[j]  = hash(var_type[i], var_rank[i], index);
                allInRange = allInRange && inRange(value[j], var_type[i]);
            }
        }
        if (var_name[i][0] == 'c') {
            err = ncmpi_put_vara_text_all(ncid, i, start, var_shape[i], text);
            IF (err != NC_NOERR)
                error("ncmpi_put_vara_text_all: %s", ncmpi_strerror(err));
        } else {
            err = ncmpi_put_vara_double_all(ncid, i, start, var_shape[i],value);
            if (allInRange) {
                IF (err != NC_NOERR)
                    error("ncmpi_put_vara_double_all: %s", ncmpi_strerror(err));
            } else {
                IF (err != NC_ERANGE)
                    error("type-conversion range error: status = %d", err);
            }
        }
    }
}


/* Create & write all of specified file using global variables */
void
write_file(char *filename) 
{
    int  err, ncid;

    err = ncmpi_create(comm, filename, NC_CLOBBER|extra_flags, info,
                       &ncid);
    IF (err != NC_NOERR) error("ncmpi_create: %s", ncmpi_strerror(err));

    def_dims(ncid);
    def_vars(ncid);
    put_atts(ncid);

    err = ncmpi_enddef(ncid);
    IF (err != NC_NOERR) error("ncmpi_enddef: %s", ncmpi_strerror(err));

    put_vars(ncid);

    err = ncmpi_close (ncid);
    IF (err != NC_NOERR) error("ncmpi_close: %s", ncmpi_strerror(err));
}


/*
 * check dimensions of specified file have expected name & length
 */
void
check_dims(int  ncid)
{
    char name[NC_MAX_NAME];
    MPI_Offset length;
    int  i, err;

    for (i = 0; i < NDIMS; i++) {
        err = ncmpi_inq_dim(ncid, i, name, &length);
        IF (err != NC_NOERR)
            error("ncmpi_inq_dim: %s", ncmpi_strerror(err));
        IF (strcmp(name, dim_name[i]) != 0)
            error("Unexpected name of dimension %d", i);
        IF (length != dim_len[i])
            error("Unexpected length %d of dimension %d", length, i);
    }
}


/*
 * check variables of specified file have expected name, type, shape & values
 */
void
check_vars(int  ncid)
{
    MPI_Offset index[MAX_RANK];
    char text, name[NC_MAX_NAME];
    int  i, j, err;
    int  nok = 0;      /* count of valid comparisons */
    int  isChar, ndims, dimids[MAX_RANK];
    double value, expect;
    nc_type datatype;
    MPI_Offset length;

    for (i = 0; i < numVars; i++) {
        isChar = var_type[i] == NC_CHAR;
        err = ncmpi_inq_var(ncid, i, name, &datatype, &ndims, dimids, NULL);
        IF (err != NC_NOERR) 
            error("ncmpi_inq_var: %s", ncmpi_strerror(err));
        IF (strcmp(name, var_name[i]) != 0) 
            error("Unexpected var_name");
        IF (datatype != var_type[i]) 
            error("Unexpected type");
        IF (ndims != var_rank[i]) 
            error("Unexpected rank");
        for (j = 0; j < ndims; j++) {
            err = ncmpi_inq_dim(ncid, dimids[j], name, &length);
            IF (err != NC_NOERR) 
                error("ncmpi_inq_dim: %s", ncmpi_strerror(err));
            IF (length != var_shape[i][j]) 
                error("Unexpected shape");
        }
        for (j = 0; j < var_nels[i]; j++) {
            err = toMixedBase(j, var_rank[i], var_shape[i], index);
            IF (err != NC_NOERR) 
                error("error in toMixedBase 2");
            expect = hash( var_type[i], var_rank[i], index );
            if (isChar) {
                ncmpi_begin_indep_data(ncid);
                err = ncmpi_get_var1_text(ncid, i, index, &text);
                IF (err != NC_NOERR)
                    error("ncmpi_get_var1_text: %s", ncmpi_strerror(err));
                IF (text != (char)expect) {
                    error("Var %s (varid=%d) value read 0x%02x not that expected 0x%02x ",
                          var_name[i], i, text, (char)expect);
                    print_n_size_t(var_rank[i], index);
                } else {
                    nok++;
                }
                ncmpi_end_indep_data(ncid);
            } else {
                ncmpi_begin_indep_data(ncid);
                err = ncmpi_get_var1_double(ncid, i, index, &value); 
                if (inRange(expect,var_type[i])) {
                    IF (err != NC_NOERR) {
                        error("ncmpi_get_var1_double: %s", ncmpi_strerror(err));
                    } else {
                        IF (!equal(value,expect,var_type[i], NCT_DOUBLE)) {
                            error("Var %s (varid=%d) value read % 12.5e not that expected % 12.7e ",
                                  var_name[i], i, value, expect);
                            print_n_size_t(var_rank[i], index);
                        } else {
                            nok++;
                        }
                    }
                }
                ncmpi_end_indep_data(ncid);
            }
        }
    }
    /* print_nok(nok); */
}


/*
 * check attributes of specified file have expected name, type, length & values
 */
void
check_atts(int  ncid) 
{
    char name[NC_MAX_NAME], text[MAX_NELS];
    int  i, j, err;        /* status */
    nc_type datatype;
    MPI_Offset k, length;
    double expect, value[MAX_NELS];
    int nok = 0;      /* count of valid comparisons */

    for (i = -1; i < numVars; i++) {
        for (j = 0; j < NATTS(i); j++) {
            err = ncmpi_inq_attname(ncid, i, j, name);
            IF (err != NC_NOERR) 
                error("ncmpi_inq_attname: %s", ncmpi_strerror(err));
            IF (strcmp(name, ATT_NAME(i,j)) != 0)
                error("ncmpi_inq_attname: unexpected name");
            err = ncmpi_inq_att(ncid, i, name, &datatype, &length);
            IF (err != NC_NOERR) 
                error("ncmpi_inq_att: %s", ncmpi_strerror(err));
            IF (datatype != ATT_TYPE(i,j))
                error("ncmpi_inq_att: unexpected type");
            IF (length != ATT_LEN(i,j))
                error("ncmpi_inq_att: unexpected length");
            if (datatype == NC_CHAR) {
                err = ncmpi_get_att_text(ncid, i, name, text);
                IF (err != NC_NOERR)
                    error("ncmpi_get_att_text: %s", ncmpi_strerror(err));
                for (k = 0; k < ATT_LEN(i,j); k++) {
                    expect = hash(datatype, -1, &k);
                    if (text[k] != (char)expect) {
                        error("ncmpi_get_att_text: unexpected value");
                    } else {
                        nok++;
                    }
                }
            } else {
                err = ncmpi_get_att_double(ncid, i, name, value);
                for (k = 0; k < ATT_LEN(i,j); k++) {
                    expect = hash(datatype, -1, &k);
                    if (inRange(expect,ATT_TYPE(i,j))) {
                        IF (err != NC_NOERR)
                            error("ncmpi_get_att_double: %s", ncmpi_strerror(err));
                        IF (!equal(value[k], expect, ATT_TYPE(i,j), NCT_DOUBLE)) {
                            error("Att value read not that expected");
                        } else {
                            nok++;
                        }
                    }
                }
            }
        }
    }
    /* print_nok(nok); */
}


/* Check file (dims, vars, atts) corresponds to global variables */
void
check_file(char *filename) 
{
    int ncid, err;

    err = ncmpi_open(comm, filename, NC_NOWRITE, info, &ncid);
    IF (err != NC_NOERR) {
        error("ncmpi_open: %s", ncmpi_strerror(err));
    } else {
        check_dims(ncid);
        check_vars(ncid);
        check_atts(ncid);
        err = ncmpi_close (ncid);
        IF (err != NC_NOERR) 
            error("ncmpi_close: %s", ncmpi_strerror(err));
    }
}

/* TODO: Maybe this function belongs in the netcdf library. */
const char *
s_nc_type(nc_type type)
{
    switch((int)type){
        case NC_CHAR:   return "NC_CHAR";
        case NC_BYTE:   return "NC_BYTE";
        case NC_UBYTE:  return "NC_UBYTE";
        case NC_SHORT:  return "NC_SHORT";
        case NC_USHORT: return "NC_USHORT";
        case NC_INT:    return "NC_INT";
        case NC_UINT:   return "NC_UINT";
        case NC_FLOAT:  return "NC_FLOAT";
        case NC_DOUBLE: return "NC_DOUBLE";
        case NC_INT64:  return "NC_INT64";
        case NC_UINT64: return "NC_UINT64";
    }
    return "";
}

int
nctypelen(nc_type type)
{
    switch(type){
        case NC_BYTE :
        case NC_CHAR :   return((int)sizeof(char));
        case NC_SHORT :  return((int)sizeof(short));
        case NC_INT :    return((int)sizeof(int));
        case NC_FLOAT :  return((int)sizeof(float));
        case NC_DOUBLE : return((int)sizeof(double));
        case NC_UBYTE :  return(1);
        case NC_USHORT : return((int)sizeof(unsigned short));
        case NC_UINT :   return((int)sizeof(unsigned int));
        case NC_INT64 :  return((int)sizeof(long long));
        case NC_UINT64 : return((int)sizeof(unsigned long long));
        default:         return -1;
    }
}

MPI_Datatype
nc_mpi_type(nc_type type)
{
    switch(type){
        case NC_BYTE :   return MPI_SIGNED_CHAR;
        case NC_CHAR :   return MPI_CHAR;
        case NC_SHORT :  return MPI_SHORT;
        case NC_INT :    return MPI_INT;
        case NC_FLOAT :  return MPI_FLOAT;
        case NC_DOUBLE : return MPI_DOUBLE;
        case NC_UBYTE :  return MPI_UNSIGNED_CHAR;
        case NC_USHORT : return MPI_UNSIGNED_SHORT;
        case NC_UINT :   return MPI_UNSIGNED;
        case NC_INT64 :  return MPI_LONG_LONG_INT;
        case NC_UINT64 : return MPI_UNSIGNED_LONG_LONG;
        default:         return MPI_DATATYPE_NULL;
    }
}


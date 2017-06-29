/*********************************************************************
 *   Copyright 1993, University Corporation for Atmospheric Research
 *   See netcdf/README file for copying and redistribution conditions.
 *   $Header$
 *********************************************************************/
/* $Id: ncmpidump.c 2322 2016-02-26 22:47:06Z wkliao $ */

#if HAVE_CONFIG_H
# include <ncconfig.h>
#endif

#include <stdio.h>
#include <stdlib.h>  /* strtol() */
#include <string.h>  /* strrchr() */
#include <ctype.h>
#include <stdlib.h>
#include <unistd.h>

#include <mpi.h>
#include <pnetcdf.h>
#include "ncmpidump.h"
#include "dumplib.h"
#include "vardata.h"

static void usage(void);
static char* name_path(const char* path);
static const char* type_name(nc_type  type);
static void tztrim(char* ss);
static void pr_att_string(size_t len, const char* string);
static void pr_att_vals(nc_type  type, size_t len, const double* vals);
static void pr_att(int ncid, int varid, const char *varname, int ia);
static void do_ncdump(const char* path, struct fspec* specp);
static void make_lvars(char* optarg, struct fspec* fspecp);
static void set_sigdigs( const char* optarg);
static void set_precision( const char *optarg);
int main(int argc, char** argv);

#define    STREQ(a, b)    (*(a) == *(b) && strcmp((a), (b)) == 0)

char *progname;

static void
usage(void)
{
#define USAGE   "\
  [-c]             Coordinate variable data and header information\n\
  [-h]             Header information only, no data\n\
  [-v var1[,...]]  Data for variable(s) <var1>,... only\n\
  [-b [c|f]]       Brief annotations for C or Fortran indices in data\n\
  [-f [c|f]]       Full annotations for C or Fortran indices in data\n\
  [-l len]         Line length maximum in data section (default 80)\n\
  [-n name]        Name for netCDF (default derived from file name)\n\
  [-p n[,n]]       Display floating-point values with less precision\n\
  [-V]             Print the file format (CDF-1, CDF-2, or CDF-5)\n\
  [-k]             Print kind of file (classic, 64-bit offset, or 64-bit data)\n\
  file             File name of input netCDF file\n"

    fprintf(stderr,
           "%s [-c|-h] [-v ...] [[-b|-f] [c|f]] [-l len] [-n name] [-p n[,n]] [-k] [-V] file\n%s",
           progname, USAGE);
    
    fprintf(stderr, "PnetCDF library version %s\n", ncmpi_inq_libvers());
}


/* 
 * convert pathname of netcdf file into name for cdl unit, by taking 
 * last component of path and stripping off any extension.
 */
static char *
name_path(const char *path)
{
    const char *cp;
    char *new;
    char *sp;

#ifdef vms
#define FILE_DELIMITER ']'
#endif    
#ifdef MSDOS
#define FILE_DELIMITER '\\'
#endif    
#ifndef FILE_DELIMITER /* default to unix */
#define FILE_DELIMITER '/'
#endif
    cp = strrchr(path, FILE_DELIMITER);
    if (cp == 0)        /* no delimiter */
        cp = path;
    else                /* skip delimeter */
        cp++;

    new = (char *) malloc((unsigned) (strlen(cp)+1));
    if (new == 0) error("out of memory!");

    strcpy(new, cp);    /* copy last component of path */
    if ((sp = strrchr(new, '.')) != NULL)
        *sp = '\0';     /* strip off any extension */
    return new;
}


static const char *
type_name(nc_type type)
{
    switch (type) {  /* conform with netcdf4 */
        case NC_BYTE:    return "byte";
        case NC_CHAR:    return "char";
        case NC_SHORT:   return "short";
        case NC_INT:     return "int";
        case NC_FLOAT:   return "float";
        case NC_DOUBLE:  return "double";
        case NC_UBYTE:   return "ubyte";
        case NC_USHORT:  return "ushort";
        case NC_UINT:    return "uint";
        case NC_INT64:   return "int64";
        case NC_UINT64:  return "uint64";
        default:
            error("type_name: bad type %d", type);
            return "bogus";
    }
}


/*
 * Remove trailing zeros (after decimal point) but not trailing decimal
 * point from ss, a string representation of a floating-point number that
 * might include an exponent part.
 */
static void
tztrim(char *ss)
{
    char *cp, *ep;
    
    cp = ss;
    if (*cp == '-')
        cp++;
    while(isdigit((int)*cp) || *cp == '.')
        cp++;
    if (*--cp == '.')
        return;
    ep = cp+1;
    while (*cp == '0')
        cp--;
    cp++;
    if (cp == ep)
        return;
    while (*ep)
        *cp++ = *ep++;
    *cp = '\0';
    return;
}


/*
 * Print attribute string, for text attributes.
 */
static void
pr_att_string(size_t      len,
              const char *string)
{
    int iel;
    const char *cp;
    const char *sp;
    unsigned char uc;

    cp = string;
    Printf ("\"");
    /* adjust len so trailing nulls don't get printed */
    sp = cp + len - 1;
    while (len != 0 && *sp-- == '\0')
        len--;

    for (iel = 0; iel < len; iel++)
        switch (uc = *cp++ & 0377) {
            case '\b':
                Printf ("\\b");
                break;
            case '\f':
                Printf ("\\f");
                break;
            case '\n':        /* generate linebreaks after new-lines */
                Printf ("\\n\",\n    \"");
                break;
            case '\r':
                Printf ("\\r");
                break;
            case '\t':
                Printf ("\\t");
                break;
            case '\v':
                Printf ("\\v");
                break;
            case '\\':
                Printf ("\\\\");
                break;
            case '\'':
                Printf ("\\'");
                break;
            case '\"':
                Printf ("\\\"");
                break;
            default:
                Printf ("%c",uc);
                break;
        }

    Printf ("\"");
}


/*
 * Print list of attribute values, for numeric attributes.  Attribute values
 * must be printed with explicit type tags, because CDL doesn't have explicit
 * syntax to declare an attribute type.
 */
static void
pr_att_vals(nc_type       type,
            size_t        len,
            const double *vals)
{
    int iel;
    char gps[30];
    signed char        sc;
    unsigned char      uc;
    short              ss;
    unsigned short     us;
    int                si;
    unsigned int       ui;
    float              ff;
    double             dd;
    long long          sll;
    unsigned long long ull;

    if (len == 0) return;

    for (iel=0; iel<len; iel++) {
        switch (type) {
            case NC_BYTE:
                sc = (signed char) vals[iel] & 0377;
                Printf ("%hhdb", sc);
                break;
            case NC_SHORT:
                ss = vals[iel];
                Printf ("%hds", ss);
                break;
            case NC_INT:
                si = (int) vals[iel];
                Printf ("%d", si);
                break;
            case NC_FLOAT:
                ff = vals[iel];
                (void) sprintf(gps, float_att_fmt, ff);
                tztrim(gps);    /* trim trailing 0's after '.' */
                Printf ("%s", gps);
                break;
            case NC_DOUBLE:
                dd = vals[iel];
                (void) sprintf(gps, double_att_fmt, dd);
                tztrim(gps);
                Printf ("%s", gps);
                break;
            case NC_UBYTE:
                uc = (unsigned char) vals[iel] & 0377;
                Printf ("%hhuUB", uc); /* match netCDF4 ncdump type indicator */
                break;
            case NC_USHORT:
                us = vals[iel];
                Printf ("%huUS", us); /* match netCDF4 ncdump type indicator */
                break;
            case NC_UINT:
                ui = vals[iel];
                Printf ("%uU", ui); /* match netCDF4 ncdump type indicator */
                break;
            case NC_INT64:
                sll = vals[iel];
                Printf ("%lldL", sll); /* match netCDF4 ncdump type indicator */
                break;
            case NC_UINT64:
                ull = vals[iel];
                Printf ("%lluUL", ull);/* match netCDF4 ncdump type indicator */
                break;
            default:
                error("pr_att_vals: bad type");
        }
        /* if not the last element */
        if (iel < len-1) Printf (", ");
    }
}


static void
pr_att(int         ncid,
       int         varid,
       const char *varname,
       int         ia)
{
    struct ncatt att;        /* attribute */
        
    NC_CHECK(ncmpi_inq_attname(ncid, varid, ia, att.name));

    Printf ("\t\t%s:%s = ", varname, att.name);

    NC_CHECK(ncmpi_inq_att(ncid, varid, att.name, &att.type, &att.len));

    if (att.len == 0) {    /* show 0-length attributes as empty strings */
        Printf ("\"\" ;\n");
        return;
    }
    switch (att.type) {
        case NC_CHAR:
            att.string = (char *) malloc(att.len);
            if (!att.string) {
                error("Out of memory!");
                NC_CHECK(ncmpi_close(ncid));
                return;
            }
            NC_CHECK(ncmpi_get_att_text(ncid, varid, att.name, att.string));
            pr_att_string(att.len, att.string);
            free(att.string);
            break;
        default:
            att.vals = (double *) malloc(att.len * sizeof(double));
            if (!att.vals) {
                error("Out of memory!");
                NC_CHECK( ncmpi_close(ncid) );
                return;
            }
            NC_CHECK(ncmpi_get_att_double(ncid, varid, att.name, att.vals));
            pr_att_vals(att.type, att.len, att.vals);
            free(att.vals);
            break;
    }
    Printf (" ;\n");
}


static void
do_ncdump(const char *path, struct fspec* specp)
{
    int ndims;            /* number of dimensions */
    int nvars;            /* number of variables */
    int ngatts;           /* number of global attributes */
    int xdimid;           /* id of unlimited dimension */
    int dimid;            /* dimension id */
    int varid;            /* variable id */
    struct ncdim dims[NC_MAX_DIMS]; /* dimensions */
    size_t vdims[NC_MAX_DIMS];    /* dimension sizes for a single variable */
    struct ncvar var;     /* variable */
    struct ncatt att;     /* attribute */
    int id;               /* dimension number per variable */
    int ia;               /* attribute number */
    int iv;               /* variable number */
    int is_coord;         /* true if variable is a coordinate variable */
    int ncid;             /* netCDF id */
    vnode* vlist = 0;     /* list for vars specified with -v option */
    int ncmpi_status;     /* return from netcdf calls */
    int NC_mode;
    MPI_Info info;

    /* Nov. 18, 2014 -- disable subfiling as it does not correctly handle the
     * cases when  nprocs < num_subfiles */
    MPI_Info_create (&info);
    MPI_Info_set (info, "pnetcdf_subfiling", "disable");

    ncmpi_status = ncmpi_open(MPI_COMM_WORLD, path, NC_NOWRITE,
                              info, &ncid);
    if (ncmpi_status != NC_NOERR)
        error("%s: %s", path, ncmpi_strerror(ncmpi_status));
    MPI_Info_free(&info);

    /*
     * If any vars were specified with -v option, get list of associated
     * variable ids
     */
    if (specp->nlvars > 0) {
        vlist = newvlist();    /* list for vars specified with -v option */
        for (iv=0; iv < specp->nlvars; iv++) {
            NC_CHECK(ncmpi_inq_varid(ncid, specp->lvars[iv], &varid));
            varadd(vlist, varid);
        }
    }

    /* if name not specified, derive it from path */
    if (specp->name == (char *)0)
        specp->name = name_path (path);

    ncmpi_inq_version(ncid, &NC_mode);
    if (specp->version) {
        if (NC_mode == NC_64BIT_DATA) 
            Printf ("%s file format: CDF-5 (big variables)\n", specp->name);
        else if (NC_mode == NC_64BIT_OFFSET) 
            Printf ("%s file format: CDF-2 (large file)\n", specp->name);
        else
            Printf ("%s file format: CDF-1\n", specp->name);
    } else if (specp->kind) {
        if (NC_mode == NC_64BIT_DATA) 
            Printf ("64-bit data\n");
        else if (NC_mode == NC_64BIT_OFFSET) 
            Printf ("64-bit offset\n");
        else
            Printf ("classic\n");
    } else {
        Printf ("netcdf %s {\n", specp->name);

        if (NC_mode == NC_64BIT_DATA) 
            Printf ("// file format: CDF-5 (big variables)\n");
        else if (NC_mode == NC_64BIT_OFFSET) 
            Printf ("// file format: CDF-2 (large file)\n");
        else
            Printf ("// file format: CDF-1\n");
        /*
         * get number of dimensions, number of variables, number of global
         * atts, and dimension id of unlimited dimension, if any
         */
        NC_CHECK(ncmpi_inq(ncid, &ndims, &nvars, &ngatts, &xdimid));

        /* print dimension info */
        if (ndims > 0) Printf ("dimensions:\n");

        for (dimid = 0; dimid < ndims; dimid++) {
            NC_CHECK(ncmpi_inq_dim(ncid, dimid, dims[dimid].name,
                                   &dims[dimid].size) );
            if (dimid == xdimid)
                Printf ("\t%s = %s ; // (%lld currently)\n",dims[dimid].name,
                        "UNLIMITED", (long long int)(dims[dimid].size));
            else
                Printf ("\t%s = %lld ;\n", dims[dimid].name, 
			(long long int)(dims[dimid].size));
        }

        if (nvars > 0) Printf ("variables:\n");

        /* get variable info, with variable attributes */
        for (varid = 0; varid < nvars; varid++) {
            NC_CHECK(ncmpi_inq_var(ncid, varid, var.name, &var.type,
                                   &var.ndims, var.dims, &var.natts) );
            Printf ("\t%s %s", type_name(var.type), var.name);
            if (var.ndims > 0) Printf ("(");
            for (id = 0; id < var.ndims; id++) {
                Printf ("%s%s", dims[var.dims[id]].name,
                        id < var.ndims-1 ? ", " : ")");
            }
            Printf (" ;\n");

            /* get variable attributes */
            for (ia = 0; ia < var.natts; ia++)
                pr_att(ncid, varid, var.name, ia); /* print ia-th attribute */
        }


        /* get global attributes */
        if (ngatts > 0) Printf ("\n// global attributes:\n");

        for (ia = 0; ia < ngatts; ia++)
            pr_att(ncid, NC_GLOBAL, "", ia); /* print ia-th global attribute */
    
        if (! specp->header_only) {
            if (nvars > 0) Printf ("data:\n");
            /* output variable data */
            for (varid = 0; varid < nvars; varid++) {
                /* if var list specified, test for membership */
                if (specp->nlvars > 0 && ! varmember(vlist, varid))
                    continue;
                NC_CHECK(ncmpi_inq_var(ncid, varid, var.name, &var.type,
                                       &var.ndims, var.dims, &var.natts));
                if (specp->coord_vals) {
                    /* Find out if this is a coordinate variable */
                    is_coord = 0;
                    for (dimid = 0; dimid < ndims; dimid++) {
                        if (strcmp(dims[dimid].name, var.name) == 0 &&
                            var.ndims == 1) {
                            is_coord = 1;
                            break;
                        }
                    }
                    if (! is_coord) /* don't get data for non-coordinate vars */
                        continue;
                }
                /*
                 * Only get data for variable if it is not a record variable,
                 * or if it is a record variable and at least one record has
                 * been written.
                 */
                if (var.ndims == 0 || var.dims[0] != xdimid ||
                    dims[xdimid].size != 0) { 
                    /* Collect variable's dim sizes */
                    for (id = 0; id < var.ndims; id++)
                         vdims[id] = dims[var.dims[id]].size;

                    var.has_fillval = 1; /* by default, but turn off for bytes */

                    /* get _FillValue attribute */
                    ncmpi_status = ncmpi_inq_att(ncid, varid, _FillValue,
                                                 &att.type, &att.len);
                    if (ncmpi_status == NC_NOERR &&
                        att.type == var.type && att.len == 1) {
                        if (var.type == NC_CHAR) {
                            char fillc;
                            NC_CHECK(ncmpi_get_att_text(ncid, varid, _FillValue,
                                                        &fillc));
                            var.fillval = fillc;
                        } else
                            NC_CHECK(ncmpi_get_att_double(ncid, varid, _FillValue,
                                                          &var.fillval));
                    } else {
                        switch (var.type) {
                            case NC_BYTE:
                                /* don't do default fill-values for bytes, too risky */
                                var.has_fillval = 0;
                                break;
                            case NC_UBYTE:
                                var.fillval = NC_FILL_UBYTE;
                                break;
                            case NC_CHAR:
                                var.fillval = NC_FILL_CHAR;
                                break;
                            case NC_SHORT:
                                var.fillval = NC_FILL_SHORT;
                                break;
                            case NC_USHORT:
                                var.fillval = NC_FILL_USHORT;
                                break;
                            case NC_INT:
                                var.fillval = NC_FILL_INT;
                                break;
                            case NC_UINT:
                                var.fillval = NC_FILL_UINT;
                                break;
                            case NC_FLOAT:
                                var.fillval = NC_FILL_FLOAT;
                                break;
                            case NC_DOUBLE:
                                var.fillval = NC_FILL_DOUBLE;
                                break;
                            case NC_INT64:
                                var.fillval = NC_FILL_INT64;
                                break;
                            case NC_UINT64:
                                var.fillval = NC_FILL_UINT64;
                                break;
                            default:
                                break;
                        }
                    }
                    if (vardata(&var, vdims, ncid, varid, specp) == -1) {
                        error("can't output data for variable %s", var.name);
                        NC_CHECK(ncmpi_close(ncid));
                        if (vlist) free(vlist);
                        return;
                    }
                }
            }
        }
        Printf ("}\n");
    }
    NC_CHECK(ncmpi_close(ncid));
    if (vlist) free(vlist);
}


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
    if (!fspecp->lvars) error("out of memory");

    cpp = fspecp->lvars;
    /* copy variable names into list */
    for (cp = strtok(optarg, ",");
         cp != NULL;
         cp = strtok((char *) NULL, ",")) {
    
        *cpp = (char *) malloc(strlen(cp) + 1);
        if (!*cpp) error("out of memory");

        strcpy(*cpp, cp);
        cpp++;
    }
    fspecp->nlvars = nvars;
}


/*
 * Extract the significant-digits specifiers from the -d argument on the
 * command-line and update the default data formats appropriately.
 */
static void
set_sigdigs(const char *optarg)
{
    char *ptr1 = 0;
    char *ptr2 = 0;
    int flt_digits = FLT_DIGITS; /* default floating-point digits */
    int dbl_digits = DBL_DIGITS; /* default double-precision digits */

    if (optarg != 0 && (int) strlen(optarg) > 0 && optarg[0] != ',')
        flt_digits = (int)strtol(optarg, &ptr1, 10);

    if (flt_digits < 1 || flt_digits > 20)
        error("unreasonable value for float significant digits: %d",
              flt_digits);

    if (ptr1 && *ptr1 == ',') {
        dbl_digits = (int)strtol(ptr1+1, &ptr2, 10);

        if (ptr2 == ptr1+1 || dbl_digits < 1 || dbl_digits > 20) {
            error("unreasonable value for double significant digits: %d",
                  dbl_digits);
        }
    }
    set_formats(flt_digits, dbl_digits);
}


/*
 * Extract the significant-digits specifiers from the -p argument on the
 * command-line, set flags so we can override C_format attributes (if any),
 * and update the default data formats appropriately.
 */
static void
set_precision(const char *optarg)
{
    char *ptr1 = 0;
    char *ptr2 = 0;
    int flt_digits = FLT_DIGITS;    /* default floating-point digits */
    int dbl_digits = DBL_DIGITS;    /* default double-precision digits */

    if (optarg != 0 && (int) strlen(optarg) > 0 && optarg[0] != ',') {
        flt_digits = (int)strtol(optarg, &ptr1, 10);
        float_precision_specified = 1;
    }

    if (flt_digits < 1 || flt_digits > 20)
        error("unreasonable value for float significant digits: %d",
              flt_digits);

    if (ptr1 && *ptr1 == ',') {
        dbl_digits = (int) strtol(ptr1+1, &ptr2, 10);
        double_precision_specified = 1;
        if (ptr2 == ptr1+1 || dbl_digits < 1 || dbl_digits > 20) {
            error("unreasonable value for double significant digits: %d",
                  dbl_digits);
        }
    }

    set_formats(flt_digits, dbl_digits);
}


int
main(int argc, char *argv[])
{
    extern int optind;
    extern int opterr;
    extern char *optarg;
    static struct fspec fspec =    /* defaults, overridden on command line */
    {
      0,            /* construct netcdf name from file name */
      false,        /* print header info only, no data? */
      false,        /* print CDF version, no data, no header */
      false,        /* print file kind, no data, no header */
      false,        /* just print coord vars? */
      false,        /* brief  comments in data section? */
      false,        /* full annotations in data section?  */
      LANG_C,       /* language conventions for indices */
      0,            /* if -v specified, number of variables */
      0             /* if -v specified, list of variable names */
    };
    int c, rank, err=EXIT_SUCCESS;
    int max_len = 80;        /* default maximum line length */
    int nameopt = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* If the user called ncmpidump without arguments, print the usage
     * message and return peacefully. */
    if (argc <= 1) {
        if (rank == 0) usage();
        goto fn_exit;
    }

    opterr = 1;
    progname = argv[0];
    set_formats(FLT_DIGITS, DBL_DIGITS); /* default for float, double data */

    while ((c = getopt(argc, argv, "b:cf:hVkl:n:v:d:p:")) != EOF)
        switch(c) {
            case 'h':        /* dump header only, no data */
                fspec.header_only = true;
                break;
            case 'V':        /* dump CDF version, no data, no header */
                fspec.version = true;
                break;
            case 'k':        /* dump CDF version, no data, no header */
                fspec.kind = true;
                break;
            case 'c':        /* header, data only for coordinate dims */
                fspec.coord_vals = true;
                break;
            case 'n':        /* provide different name than derived from
                              * file name
                              */
                fspec.name = optarg;
                nameopt = 1;
                break;
            case 'b':        /* brief comments in data section */
                fspec.brief_data_cmnts = true;
                switch (tolower(optarg[0])) {
                    case 'c':
                        fspec.data_lang = LANG_C;
                        break;
                    case 'f':
                        fspec.data_lang = LANG_F;
                        break;
                    default:
                        if (rank == 0)
                            fprintf(stderr,"invalid value for -b option: %s",
                                    optarg);
                        err = EXIT_FAILURE;
                        goto fn_exit;
                }
                break;
            case 'f':        /* full comments in data section */
                fspec.full_data_cmnts = true;
                switch (tolower(optarg[0])) {
                    case 'c':
                        fspec.data_lang = LANG_C;
                        break;
                    case 'f':
                        fspec.data_lang = LANG_F;
                        break;
                    default:
                        if (rank == 0)
                            fprintf(stderr,"invalid value for -f option: %s",
                                    optarg);
                        err = EXIT_FAILURE;
                        goto fn_exit;
                }
                break;
            case 'l':        /* maximum line length */
                max_len = (int) strtol(optarg, 0, 0);
                if (max_len < 10) {
                    if (rank == 0)
                        fprintf(stderr,
                                "unreasonably small line length specified: %d",
                                max_len);
                    err = EXIT_FAILURE;
                    goto fn_exit;
                }
                break;
            case 'v':        /* variable names */
                /* make list of names of variables specified */
                make_lvars (optarg, &fspec);
                break;
            case 'd':        /* specify precision for floats (old option) */
                set_sigdigs(optarg);
                break;
            case 'p':        /* specify precision for floats */
                set_precision(optarg);
                break;
            case '?':
                if (rank == 0) usage();
                goto fn_exit;
            default:
                break;
        }

    set_max_len(max_len);
    
    argc -= optind;
    argv += optind;

#ifndef MULTI_FILE_DUMP
    /* If no input file, print usage message. */
    if (argc != 1) {
        if (rank == 0) {
            if (argc == 0)
                fprintf(stderr,"Error: input filename is missing\n\n");
            else
                fprintf(stderr,"Error: only one input file is allowed\n\n");
            usage();
        }
        err = EXIT_FAILURE;
        goto fn_exit;
    }

    if (!nameopt) fspec.name = (char *)0;
    do_ncdump(argv[0], &fspec);
#else
    /* support multiple input files */
    if (argc < 1) {
        if (rank == 0) {
            fprintf(stderr,"Error: input filename(s) is missing\n\n");
            usage();
        }
        err = EXIT_FAILURE;
        goto fn_exit;
    }
    int i = 0;
    do {
        if (!nameopt) fspec.name = (char *)0;
        if (argc > 0)
              do_ncdump(argv[i], &fspec);
    } while (++i < argc);
#endif

fn_exit:
    MPI_Finalize();
#ifdef vms
    exit(err);
#else
    return err;
#endif
}

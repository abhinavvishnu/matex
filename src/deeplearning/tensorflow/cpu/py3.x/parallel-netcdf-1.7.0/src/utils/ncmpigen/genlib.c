/*********************************************************************
 *   Copyright 1993, UCAR/Unidata
 *   See netcdf/COPYRIGHT file for copying and redistribution conditions.
 *   $Header$
 *********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>	/* for isprint() */
#ifndef NO_STDARG
#include	<stdarg.h>
#else
/* try varargs instead */
#include	<varargs.h>
#endif /* !NO_STDARG */
#include <pnetcdf.h>
#include "generic.h"
#include "ncmpigen.h"
#include "genlib.h"

#include <mpi.h>
extern char *netcdf_name; /* output netCDF filename, if on command line. */
extern int netcdf_flag;
extern int c_flag;
extern int fortran_flag;
extern int giantfile_flag;
extern int giantvar_flag;
extern int nofill_flag;

int	lineno = 1;
int	derror_count = 0;



/* create netCDF from in-memory structure */
static void
gen_netcdf(
     char *filename)		/* name for output netcdf file */
{
    int idim, ivar, iatt;
    int dimid;
    int varid;
    int stat;

    if (giantfile_flag) {
	    stat = ncmpi_create(MPI_COMM_WORLD, filename, 
			    NC_CLOBBER|NC_64BIT_OFFSET, MPI_INFO_NULL, &ncid);
    } else if (giantvar_flag) {
	    stat = ncmpi_create(MPI_COMM_WORLD, filename,
			    NC_CLOBBER|NC_64BIT_DATA, MPI_INFO_NULL, &ncid);
    } else {
	    stat = ncmpi_create(MPI_COMM_WORLD, filename, 
			    NC_CLOBBER, MPI_INFO_NULL, &ncid);
    }
    check_err(stat, "ncmpi_create", __func__, __LINE__, __FILE__);

    /* define dimensions from info in dims array */
    for (idim = 0; idim < ndims; idim++) {
	stat = ncmpi_def_dim(ncid, dims[idim].name, dims[idim].size, &dimid);
	check_err(stat, "ncmpi_def_dim", __func__, __LINE__, __FILE__);
    }

    /* define variables from info in vars array */
    for (ivar = 0; ivar < nvars; ivar++) {
	stat = ncmpi_def_var(ncid,
			  vars[ivar].name,
			  vars[ivar].type,
			  vars[ivar].ndims,
			  vars[ivar].dims,
			  &varid);
	check_err(stat, "ncmpi_def_var", __func__, __LINE__, __FILE__);
    }

    /* define attributes from info in atts array */
    for (iatt = 0; iatt < natts; iatt++) {
	varid = (atts[iatt].var == -1) ? NC_GLOBAL : atts[iatt].var;
	switch(atts[iatt].type) {
	case NC_BYTE:
	    stat = ncmpi_put_att_schar(ncid, varid, atts[iatt].name,
				    atts[iatt].type, atts[iatt].len,
				    (signed char *) atts[iatt].val);
	    break;
	case NC_CHAR:
	    stat = ncmpi_put_att_text(ncid, varid, atts[iatt].name,
				   atts[iatt].len,
				   (char *) atts[iatt].val);
	    break;
	case NC_SHORT:
	    stat = ncmpi_put_att_short(ncid, varid, atts[iatt].name,
				    atts[iatt].type, atts[iatt].len,
				    (short *) atts[iatt].val);
	    break;
	case NC_INT:
	    stat = ncmpi_put_att_int(ncid, varid, atts[iatt].name,
				    atts[iatt].type, atts[iatt].len,
				    (int *) atts[iatt].val);
	    break;
	case NC_FLOAT:
	    stat = ncmpi_put_att_float(ncid, varid, atts[iatt].name,
				    atts[iatt].type, atts[iatt].len,
				    (float *) atts[iatt].val);
	    break;
	case NC_DOUBLE:
	    stat = ncmpi_put_att_double(ncid, varid, atts[iatt].name,
				    atts[iatt].type, atts[iatt].len,
				    (double *) atts[iatt].val);
	    break;
	case NC_UBYTE:
	    stat = ncmpi_put_att_ubyte(ncid, varid, atts[iatt].name,
				    atts[iatt].type, atts[iatt].len,
				    (unsigned char *) atts[iatt].val);
	    break;
	case NC_USHORT:
	    stat = ncmpi_put_att_ushort(ncid, varid, atts[iatt].name,
				    atts[iatt].type, atts[iatt].len,
				    (unsigned short *) atts[iatt].val);
	    break;
	case NC_UINT:
	    stat = ncmpi_put_att_uint(ncid, varid, atts[iatt].name,
				    atts[iatt].type, atts[iatt].len,
				    (unsigned int *) atts[iatt].val);
	    break;
	case NC_INT64:
	    stat = ncmpi_put_att_longlong(ncid, varid, atts[iatt].name,
				    atts[iatt].type, atts[iatt].len,
				    (long long *) atts[iatt].val);
	    break;
	case NC_UINT64:
	    stat = ncmpi_put_att_ulonglong(ncid, varid, atts[iatt].name,
				    atts[iatt].type, atts[iatt].len,
				    (unsigned long long *) atts[iatt].val);
	    break;
	default:
	    stat = NC_EBADTYPE;
	}
	check_err(stat, "ncmpi_put_att_xxx", __func__, __LINE__, __FILE__);
    }

    /* serial netcdf calls nc_set_fill(NC_NOFILL) here, but we do not
     * implement that routine (NC_NOFILL is our default and only behavior) */

    stat = ncmpi_enddef(ncid);
    check_err(stat, "ncmpi_enddef", __func__, __LINE__, __FILE__);
}


/*
 * Given a netcdf type, a pointer to a vector of values of that type,
 * and the index of the vector element desired, returns a pointer to a
 * malloced string representing the value in C.
 */
static char *
cstring(
     nc_type type,		/* netCDF type code */
     void *valp,		/* pointer to vector of values */
     int num)			/* element of vector desired */
{
    static char *cp, *sp, ch;
    signed char *bytep;
    short *shortp;
    int *intp;
    float *floatp;
    double *doublep;
    unsigned char *ubytep;
    unsigned short *ushortp;
    unsigned int *uintp;
    long long *int64p;
    unsigned long long *uint64p;

    switch (type) {
      case NC_CHAR:
	sp = cp = (char *) emalloc (7);
	*cp++ = '\'';
	ch = *((char *)valp + num);
	switch (ch) {
	  case '\b': *cp++ = '\\'; *cp++ = 'b'; break;
	  case '\f': *cp++ = '\\'; *cp++ = 'f'; break;
	  case '\n': *cp++ = '\\'; *cp++ = 'n'; break;
	  case '\r': *cp++ = '\\'; *cp++ = 'r'; break;
	  case '\t': *cp++ = '\\'; *cp++ = 't'; break;
	  case '\v': *cp++ = '\\'; *cp++ = 'v'; break;
	  case '\\': *cp++ = '\\'; *cp++ = '\\'; break;
	  case '\'': *cp++ = '\\'; *cp++ = '\''; break;
	  default:
	    if (!isprint(ch)) {
		static char octs[] = "01234567";
		int rem = ((unsigned char)ch)%64;
		*cp++ = '\\';
		*cp++ = octs[((unsigned char)ch)/64]; /* to get, e.g. '\177' */
		*cp++ = octs[rem/8];
		*cp++ = octs[rem%8];
	    } else {
		*cp++ = ch;
	    }
	    break;
	}
	*cp++ = '\'';
	*cp = '\0';
	return sp;
	
      case NC_BYTE:
	cp = (char *) emalloc (7);
	bytep = (signed char *)valp;
	/* Need to convert '\377' to -1, for example, on all platforms */
	(void) sprintf(cp,"%d", (signed char) *(bytep+num));
	return cp;

      case NC_SHORT:
	cp = (char *) emalloc (10);
	shortp = (short *)valp;
	(void) sprintf(cp,"%d",* (shortp + num));
	return cp;

      case NC_INT:
	cp = (char *) emalloc (20);
	intp = (int *)valp;
	(void) sprintf(cp,"%d",* (intp + num));
	return cp;

      case NC_FLOAT:
	cp = (char *) emalloc (20);
	floatp = (float *)valp;
	(void) sprintf(cp,"%.8g",* (floatp + num));
	return cp;

      case NC_DOUBLE:
	cp = (char *) emalloc (20);
	doublep = (double *)valp;
	(void) sprintf(cp,"%.16g",* (doublep + num));
	return cp;

      case NC_UBYTE:
	cp = (char *) emalloc (7);
	ubytep = (unsigned char *)valp;
	/* Need to convert '\377' to -1, for example, on all platforms */
	(void) sprintf(cp,"%hhu", (unsigned char) *(ubytep+num));
	return cp;

      case NC_USHORT:
	cp = (char *) emalloc (10);
	ushortp = (unsigned short *)valp;
	(void) sprintf(cp,"%hu",* (ushortp + num));
	return cp;

      case NC_UINT:
	cp = (char *) emalloc (20);
	uintp = (unsigned int *)valp;
	(void) sprintf(cp,"%u",* (uintp + num));
	return cp;

      case NC_INT64:
	cp = (char *) emalloc (20);
	int64p = (long long *)valp;
	(void) sprintf(cp,"%lld",* (int64p + num));
	return cp;

      case NC_UINT64:
	cp = (char *) emalloc (20);
	uint64p = (unsigned long long *)valp;
	(void) sprintf(cp,"%llu",* (uint64p + num));
	return cp;

      default:
	derror("cstring: bad type code");
	return 0;
    }
}


/*
 * Generate C code for creating netCDF from in-memory structure.
 */
static void
gen_c(
     const char *filename)
{
    int idim, ivar, iatt, jatt, maxdims;
    int vector_atts;
    char *val_string;
    char stmnt[C_MAX_STMNT];

    /* wrap in main program */
    cline("#include <stdio.h>");
    cline("#include <stdlib.h>");
    cline("#include <pnetcdf.h>");
    cline("#include <mpi.h>");
    cline("");
    cline("void");
    cline("check_err(int stat, const char *ncmpi_func, const char *calling_func, int lineno, const char *calling_file) {");
    cline("    if (stat != NC_NOERR) {");
    cline("        fprintf(stderr, \"ncmpigen error when calling %s in %s() at line %d of %s: %s\\n\", ncmpi_func, calling_func, lineno, calling_file, ncmpi_strerror(stat));");
    cline("        exit(1);");
    cline("    }");
    cline("}");
    cline("");
    cline("int");
    sprintf(stmnt, "main(int argc, char **argv) {\t\t\t/* create %s */", filename);
    cline(stmnt);

    /* create necessary declarations */
    cline("");
    cline("   int  stat;\t\t\t/* return status */");
    cline("   int  ncid;\t\t\t/* netCDF id */");

    if (ndims > 0) {
	cline("");
	cline("   /* dimension ids */");
	for (idim = 0; idim < ndims; idim++) {
	    sprintf(stmnt, "   int %s_dim;", dims[idim].lname);
	    cline(stmnt);
	    }

	cline("");
	cline("   /* dimension lengths */");
	for (idim = 0; idim < ndims; idim++) {
	    if (dims[idim].size == NC_UNLIMITED) {
		sprintf(stmnt, "   MPI_Offset %s_len = NC_UNLIMITED;",
			dims[idim].lname);
	    } else {
		sprintf(stmnt, "   MPI_Offset %s_len = %lu;",
			dims[idim].lname,
			(unsigned long) dims[idim].size);
	    }
	    cline(stmnt);
	}
    }

    maxdims = 0;	/* most dimensions of any variable */
    for (ivar = 0; ivar < nvars; ivar++)
      if (vars[ivar].ndims > maxdims)
	maxdims = vars[ivar].ndims;

    if (nvars > 0) {
	cline("");
	cline("   /* variable ids */");
	for (ivar = 0; ivar < nvars; ivar++) {
	    sprintf(stmnt, "   int %s_id;", vars[ivar].lname);
	    cline(stmnt);
	}

	cline("");
	cline("   /* rank (number of dimensions) for each variable */");
	for (ivar = 0; ivar < nvars; ivar++) {
	    sprintf(stmnt, "#  define RANK_%s %d", vars[ivar].lname,
		    vars[ivar].ndims);
	    cline(stmnt);
	}
	if (maxdims > 0) {	/* we have dimensioned variables */
	    cline("");
	    cline("   /* variable shapes */");
	    for (ivar = 0; ivar < nvars; ivar++) {
		if (vars[ivar].ndims > 0) {
		    sprintf(stmnt, "   int %s_dims[RANK_%s];",
			    vars[ivar].lname, vars[ivar].lname);
		    cline(stmnt);
		}
	    }
	}
    }

    /* determine if we need any attribute vectors */
    vector_atts = 0;
    for (iatt = 0; iatt < natts; iatt++) {
	if (atts[iatt].type != NC_CHAR) {
	    vector_atts = 1;
	    break;
	}
    }
    if (vector_atts) {
	cline("");
	cline("   /* attribute vectors */");
	for (iatt = 0; iatt < natts; iatt++) {
	    if (atts[iatt].type != NC_CHAR) {
		sprintf(stmnt,
		    "   %s %s_%s[%lu];",
		    ncatype(atts[iatt].type),
		    atts[iatt].var == -1 ? "cdf" : vars[atts[iatt].var].lname,
		    atts[iatt].lname,
		    (unsigned long) atts[iatt].len);
		cline(stmnt);
	    }
	}
    }

    /* create netCDF file, uses NC_CLOBBER mode */
    cline("");
    /* stat already declared above */
    cline("   MPI_Init(&argc, &argv);");
    cline("   /* enter define mode */");

    if (giantfile_flag) {
	    sprintf(stmnt,
		    "   stat = ncmpi_create(MPI_COMM_WORLD, \"%s\", NC_CLOBBER|NC_64BIT_OFFSET, MPI_INFO_NULL, &ncid);",
	    filename);
    } else if (giantvar_flag) {
	    sprintf(stmnt,
		    "   stat = ncmpi_create(MPI_COMM_WORLD, \"%s\", NC_CLOBBER|NC_64BIT_DATA, MPI_INFO_NULL, &ncid);",
		    filename);
    } else {
	    sprintf(stmnt, "   stat = ncmpi_create(MPI_COMM_WORLD, \"%s\", NC_CLOBBER, MPI_INFO_NULL, &ncid);", 
			    filename);
    }
    cline(stmnt);
    cline("   check_err(stat,\"ncmpi_create\", __func__, __LINE__,__FILE__);");
    
    /* define dimensions from info in dims array */
    if (ndims > 0) {
	cline("");
	cline("   /* define dimensions */");
    }
    for (idim = 0; idim < ndims; idim++) {
	sprintf(stmnt,
		"   stat = ncmpi_def_dim(ncid, \"%s\", %s_len, &%s_dim);",
		dims[idim].name, dims[idim].lname, dims[idim].lname);
	cline(stmnt);
	cline("   check_err(stat,\"ncmpi_def_dim\", __func__, __LINE__,__FILE__);");
    }

    /* define variables from info in vars array */
    if (nvars > 0) {
	cline("");
	cline("   /* define variables */");
	for (ivar = 0; ivar < nvars; ivar++) {
	    cline("");
	    for (idim = 0; idim < vars[ivar].ndims; idim++) {
		sprintf(stmnt,
			"   %s_dims[%d] = %s_dim;",
			vars[ivar].lname,
			idim,
			dims[vars[ivar].dims[idim]].lname);
		cline(stmnt);
	    }
	    if (vars[ivar].ndims > 0) {	/* a dimensioned variable */
		sprintf(stmnt,
			"   stat = ncmpi_def_var(ncid, \"%s\", %s, RANK_%s, %s_dims, &%s_id);",
			vars[ivar].name,
			nctype(vars[ivar].type),
			vars[ivar].lname,
			vars[ivar].lname,
			vars[ivar].lname);
	    } else {		/* a scalar */
		sprintf(stmnt,
			"   stat = ncmpi_def_var(ncid, \"%s\", %s, RANK_%s, 0, &%s_id);",
			vars[ivar].name,
			nctype(vars[ivar].type),
			vars[ivar].lname,
			vars[ivar].lname);
	    }
	    cline(stmnt);
	    cline("   check_err(stat,\"ncmpi_def_var\", __func__, __LINE__,__FILE__);");
	}
    }
    
    /* define attributes from info in atts array */
    if (natts > 0) {
	cline("");
	cline("   /* assign attributes */");
	for (iatt = 0; iatt < natts; iatt++) {
	    if (atts[iatt].type == NC_CHAR) { /* string */
		val_string = cstrstr((char *) atts[iatt].val, atts[iatt].len);
		sprintf(stmnt,
			"   stat = ncmpi_put_att_text(ncid, %s%s, \"%s\", %lu, %s);",
			atts[iatt].var == -1 ? "NC_GLOBAL" : vars[atts[iatt].var].lname,
			atts[iatt].var == -1 ? "" : "_id",
			atts[iatt].name,
			(unsigned long) atts[iatt].len,
			val_string);
		cline(stmnt);
		free (val_string);
	    }
	    else {			/* vector attribute */
		for (jatt = 0; jatt < atts[iatt].len ; jatt++) {
		    val_string = cstring(atts[iatt].type,atts[iatt].val,jatt);
		    sprintf(stmnt, "   %s_%s[%d] = %s;",
			    atts[iatt].var == -1 ? "cdf" : vars[atts[iatt].var].lname,
			    atts[iatt].lname,
			    jatt, 
			    val_string);
		    cline(stmnt);
		    free (val_string);
		}
		
		sprintf(stmnt,
			"   stat = ncmpi_put_att_%s(ncid, %s%s, \"%s\", %s, %lu, %s_%s);",
			ncatype(atts[iatt].type),
			atts[iatt].var == -1 ? "NC_GLOBAL" : vars[atts[iatt].var].lname,
			atts[iatt].var == -1 ? "" : "_id",
			atts[iatt].name,
			nctype(atts[iatt].type),
			(unsigned long) atts[iatt].len,
			atts[iatt].var == -1 ? "cdf" : vars[atts[iatt].var].lname,
			atts[iatt].lname);
		cline(stmnt);
	    }
	    cline("   check_err(stat,\"ncmpi_put_att_xxx\", __func__, __LINE__,__FILE__);");
	}
    }
    /* here's another place where serial netcdf would insert a call to
     * setfill(NOFILL), but that's our only supported behavior, so skip */
    cline("");
    cline("   /* leave define mode */");
    cline("   stat = ncmpi_enddef (ncid);");
    cline("   check_err(stat,\"ncmpi_enddef\", __func__, __LINE__,__FILE__);");
}


/* return Fortran type name for netCDF type, given type code */
static const char *
ncftype(
     nc_type type)		/* netCDF type code */
{
    switch (type) {

      case NC_BYTE:
	return "integer";
      case NC_CHAR:
	return "character";
      case NC_SHORT:
	return "integer";
      case NC_INT:
#ifdef MSDOS
	return "integer*4";
#else
	return "integer";
#endif
      case NC_FLOAT:
	return "real";
#ifdef _CRAY
      case NC_DOUBLE:
	return "real";		/* we don't support CRAY 128-bit doubles */
#else
      case NC_DOUBLE:
	return "double precision";
#endif
      default:
	derror("ncftype: bad type code");
	return 0;

    }
}


/* return Fortran type suffix for netCDF type, given type code */
const char *
nfstype(
     nc_type type)		/* netCDF type code */
{
    switch (type) {
      case NC_BYTE:
	return "int1";
      case NC_CHAR:
	return "text";
      case NC_SHORT:
	return "int2";
      case NC_INT:
	return "int";
      case NC_FLOAT:
	return "real";
      case NC_DOUBLE:
	return "double";
      default:
	derror("nfstype: bad type code");
	return 0;

    }
}


/* Return Fortran function suffix for netCDF type, given type code.
 * This should correspond to the Fortran type name in ncftype().
 */
const char *
nfftype(
     nc_type type)		/* netCDF type code */
{
    switch (type) {
      case NC_BYTE:
	return "int";
      case NC_CHAR:
	return "text";
      case NC_SHORT:
	return "int";
      case NC_INT:
	return "int";
      case NC_FLOAT:
	return "real";
#ifdef _CRAY
      case NC_DOUBLE:
	return "real";		/* we don't support CRAY 128-bit doubles */
#else
      case NC_DOUBLE:
	return "double";
#endif
      default:
	derror("nfstype: bad type code");
	return 0;

    }
}


/* return FORTRAN name for netCDF type, given type code */
static const char *
ftypename(
     nc_type type)			/* netCDF type code */
{
    switch (type) {
      case NC_BYTE:
	return "NF_INT1";
      case NC_CHAR:
	return "NF_CHAR";
      case NC_SHORT:
	return "NF_INT2";
      case NC_INT:
	return "NF_INT";
      case NC_FLOAT:
	return "NF_REAL";
      case NC_DOUBLE:
	return "NF_DOUBLE";
      default:
	derror("ftypename: bad type code");
	return 0;
    }
}


/*
 * Generate FORTRAN code for creating netCDF from in-memory structure.
 */
static void
gen_fortran(
     const char *filename)
{
    int idim, ivar, iatt, jatt, itype, maxdims;
    int vector_atts;
    char *val_string;
    char stmnt[FORT_MAX_STMNT];
    char s2[NC_MAX_NAME + 10];
    char *sp;
    /* Need how many netCDF types there are, because we create an array
     * for each type of attribute. */
    int ntypes = 6;		/* number of netCDF types, NC_BYTE, ... */
    nc_type types[6];		/* at least ntypes */
    MPI_Offset max_atts[NC_DOUBLE + 1];

    types[0] = NC_BYTE;
    types[1] = NC_CHAR;
    types[2] = NC_SHORT;
    types[3] = NC_INT;
    types[4] = NC_FLOAT;
    types[5] = NC_DOUBLE;

    fline("program fgennc");

    fline("include 'pnetcdf.inc'");

    /* create necessary declarations */
    fline("* error status return");
    fline("integer  iret");
    fline("* netCDF id");
    fline("integer  ncid");
    if (nofill_flag) {
	    fline(" * to save old fill mode before changing it temporarily");
	    fline ("integer oldmode");
    }

    if (ndims > 0) {
	fline("* dimension ids");
	for (idim = 0; idim < ndims; idim++) {
	    sprintf(stmnt, "integer  %s_dim", dims[idim].lname);
	    fline(stmnt);
	}

	fline("* dimension lengths");
	for (idim = 0; idim < ndims; idim++) {
	    sprintf(stmnt, "integer  %s_len", dims[idim].lname);
	    fline(stmnt);
	}
	for (idim = 0; idim < ndims; idim++) {
	    if (dims[idim].size == NC_UNLIMITED) {
		sprintf(stmnt, "parameter (%s_len = NFMPI_UNLIMITED)",
			dims[idim].lname);
	    } else {
		sprintf(stmnt, "parameter (%s_len = %lu)",
			dims[idim].lname,
			(unsigned long) dims[idim].size);
	    }
	    fline(stmnt);
	}
	
    }

    maxdims = 0;		/* most dimensions of any variable */
    for (ivar = 0; ivar < nvars; ivar++)
      if (vars[ivar].ndims > maxdims)
	maxdims = vars[ivar].ndims;

    if (nvars > 0) {
	fline("* variable ids");
	for (ivar = 0; ivar < nvars; ivar++) {
	    sprintf(stmnt, "integer  %s_id", vars[ivar].lname);
	    fline(stmnt);
	}

	fline("* rank (number of dimensions) for each variable");
	for (ivar = 0; ivar < nvars; ivar++) {
	    sprintf(stmnt, "integer  %s_rank", vars[ivar].lname);
	    fline(stmnt);
	}
	for (ivar = 0; ivar < nvars; ivar++) {
	    sprintf(stmnt, "parameter (%s_rank = %d)", vars[ivar].lname,
		    vars[ivar].ndims);
	    fline(stmnt);
	}
	
	fline("* variable shapes");
	for (ivar = 0; ivar < nvars; ivar++) {
	    if (vars[ivar].ndims > 0) {
		sprintf(stmnt, "integer  %s_dims(%s_rank)",
			vars[ivar].lname, vars[ivar].lname);
		fline(stmnt);
	    }
	}
    }

    /* declarations for variables to be initialized */
    if (nvars > 0) {		/* we have variables */
	fline("* data variables");
	for (ivar = 0; ivar < nvars; ivar++) {
	    struct vars *v = &vars[ivar];
	    /* Generate declarations here for non-record data variables only.
	       Record variables are declared in separate subroutine later,
               when we know how big they are. */
	    if (v->ndims > 0 && v->dims[0] == rec_dim) {
		continue;
	    }
	    /* Make declarations for non-text variables only;
	       for text variables, just include string in nfmpi_put_var call */
	    if (v->type == NC_CHAR) {
                continue;
            }
	    if (v->ndims == 0) { /* scalar */
		sprintf(stmnt, "%s  %s", ncftype(v->type),
			v->lname);
	    } else {
		sprintf(stmnt, "%s  %s(", ncftype(v->type),
			v->lname);
		/* reverse dimensions for FORTRAN */
		for (idim = v->ndims-1; idim >= 0; idim--) {
		    sprintf(s2, "%s_len, ",
			    dims[v->dims[idim]].lname);
		    strcat(stmnt, s2);
		}
		sp = strrchr(stmnt, ',');
		if(sp != NULL) {
		    *sp = '\0';
		}
		strcat(stmnt, ")");
	    }
	    fline(stmnt);
	}
    }

    /* determine what attribute vectors needed */
    for (itype = 0; itype < ntypes; itype++)
        max_atts[(int)types[itype]] = 0;

    vector_atts = 0;
    for (iatt = 0; iatt < natts; iatt++) {
	if (atts[iatt].len > max_atts[(int) atts[iatt].type]) {
	    max_atts[(int)atts[iatt].type] = atts[iatt].len;
	    vector_atts = 1;
	}
    }
    if (vector_atts) {
	fline("* attribute vectors");
	for (itype = 0; itype < ntypes; itype++) {
	    if (types[itype] != NC_CHAR && max_atts[(int)types[itype]] > 0) {
		sprintf(stmnt, "%s  %sval(%lu)", ncftype(types[itype]),
			nfstype(types[itype]),
			(unsigned long) max_atts[(int)types[itype]]);
		fline(stmnt);
	    }
	}
    }

    /* create netCDF file, uses NC_CLOBBER mode */
    fline("* enter define mode");
    if (giantfile_flag) {
	    sprintf(stmnt, "iret = nfmpi_create(\'%s\', OR(NF_CLOBBER|NF_64BIT_OFFSET), ncid)", filename);
    } else if (giantvar_flag) {
	    sprintf(stmnt, "iret = nfmpi_create(\'%s\', OR(NF_CLOBBER|NF_64BIT_DATA), ncid)", filename); 
    } else {
	    sprintf(stmnt, "iret = nfmpi_create(\'%s\', NF_CLOBBER, ncid)", filename);
    }
    fline(stmnt);
    fline("call check_err(iret,\"nfmpi_create\", __func__, __LINE__,__FILE__)");
    
    /* define dimensions from info in dims array */
    if (ndims > 0)
        fline("* define dimensions");
    for (idim = 0; idim < ndims; idim++) {
	if (dims[idim].size == NC_UNLIMITED)
            sprintf(stmnt, "iret = nfmpi_def_dim(ncid, \'%s\', NFMPI_UNLIMITED, %s_dim)",
                    dims[idim].name, dims[idim].lname);
	else
            sprintf(stmnt, "iret = nfmpi_def_dim(ncid, \'%s\', %lu, %s_dim)",
                    dims[idim].name, (unsigned long) dims[idim].size,
			dims[idim].lname);
	fline(stmnt);
	fline("call check_err(iret,\"nfmpi_def_dim\", __func__, __LINE__,__FILE__)");
    }
	  
    /* define variables from info in vars array */
    if (nvars > 0) {
	fline("* define variables");
	for (ivar = 0; ivar < nvars; ivar++) {
	    for (idim = 0; idim < vars[ivar].ndims; idim++) {
		sprintf(stmnt, "%s_dims(%d) = %s_dim",
			vars[ivar].lname,
			vars[ivar].ndims - idim, /* reverse dimensions */
			dims[vars[ivar].dims[idim]].lname);
		fline(stmnt);
	    }
	    if (vars[ivar].ndims > 0) {	/* a dimensioned variable */
		sprintf(stmnt, 
			"iret = nfmpi_def_var(ncid, \'%s\', %s, %s_rank, %s_dims, %s_id)",
			vars[ivar].name,
			ftypename(vars[ivar].type),
			vars[ivar].lname,
			vars[ivar].lname,
			vars[ivar].lname);
	    } else {		/* a scalar */
		sprintf(stmnt, 
			"iret = nfmpi_def_var(ncid, \'%s\', %s, %s_rank, 0, %s_id)",
			vars[ivar].name,
			ftypename(vars[ivar].type),
			vars[ivar].lname,
			vars[ivar].lname);
	    }
	    fline(stmnt);
	    fline("call check_err(iret,\"nfmpi_def_var\", __func__, __LINE__,__FILE__)");
	}
    }

    /* define attributes from info in atts array */
    if (natts > 0) {
	fline("* assign attributes");
	for (iatt = 0; iatt < natts; iatt++) {
	    if (atts[iatt].type == NC_CHAR) { /* string */
		val_string = fstrstr((char *) atts[iatt].val, atts[iatt].len);
		sprintf(stmnt, 
			"iret = nfmpi_put_att_text(ncid, %s%s, \'%s\', %lu, %s)",
			atts[iatt].var == -1 ? "NF_GLOBAL" : vars[atts[iatt].var].lname,
			atts[iatt].var == -1 ? "" : "_id",
			atts[iatt].name,
			(unsigned long) atts[iatt].len,
			val_string);
		fline(stmnt);
		fline("call check_err(iret,\"nfmpi_put_att_text\", __func__, __LINE__,__FILE__)");
		free(val_string);
	    } else {
		for (jatt = 0; jatt < atts[iatt].len ; jatt++) {
		    val_string = fstring(atts[iatt].type,atts[iatt].val,jatt);
		    sprintf(stmnt, "%sval(%d) = %s",
			    nfstype(atts[iatt].type),
			    jatt+1, 
			    val_string);
		    fline(stmnt);
		    free (val_string);
		}
	    
		sprintf(stmnt,
			"iret = nfmpi_put_att_%s(ncid, %s%s, \'%s\', %s, %lu, %sval)",
			nfftype(atts[iatt].type),
			atts[iatt].var == -1 ? "NCGLOBAL" : vars[atts[iatt].var].lname,
			atts[iatt].var == -1 ? "" : "_id",
			atts[iatt].name,
			ftypename(atts[iatt].type),
			(unsigned long) atts[iatt].len,
			nfstype(atts[iatt].type));
		fline(stmnt);
		fline("call check_err(iret,\"nfmpi_put_att_xxx\", __func__, __LINE__,__FILE__)");
	    }
	}
    }
    /* skip the call to nfmpi_set_fill until we implement it */
    fline("* leave define mode");
    fline("iret = nfmpi_enddef(ncid)");
    fline("call check_err(iret,\"nfmpi_enddef\", __func__, __LINE__,__FILE__)");
}


/*
 * Output a C statement.
 */
void
cline(
     const char *stmnt)
{
    FILE *cout = stdout;
    
    fputs(stmnt, cout);
    fputs("\n", cout);
}

/*
 * From a long line FORTRAN statment, generates the necessary FORTRAN
 * lines with continuation characters in column 6.  If stmnt starts with "*",
 * it is treated as a one-line comment.  Statement labels are *not* handled,
 * but since we don't generate any labels, we don't care.
 */
void
fline(
     const char *stmnt)
{
    FILE *fout = stdout;
    int len = (int) strlen(stmnt);
    int line = 0;
    static char cont[] = {	/* continuation characters */
	' ', '1', '2', '3', '4', '5', '6', '7', '8', '9',
	'+', '1', '2', '3', '4', '5', '6', '7', '8', '9',
	'+', '1', '2', '3', '4', '5', '6', '7', '8', '9'};
    
    if(stmnt[0] == '*') {
	fputs(stmnt, fout);
	fputs("\n", fout);
	return;
    }

    while (len > 0) {
	if (line >= FORT_MAX_LINES)
	  derror("FORTRAN statement too long: %s",stmnt);
	(void) fprintf(fout, "     %c", cont[line++]);
	(void) fprintf(fout, "%.66s\n", stmnt);
	len -= 66;
	if (len > 0)
	  stmnt += 66;
    }
}


/* return C name for netCDF type, given type code */
const char *
nctype(
     nc_type type)			/* netCDF type code */
{
    switch (type) {
      case NC_BYTE:    return "NC_BYTE";
      case NC_CHAR:    return "NC_CHAR";
      case NC_SHORT:   return "NC_SHORT";
      case NC_INT:     return "NC_INT";
      case NC_FLOAT:   return "NC_FLOAT";
      case NC_DOUBLE:  return "NC_DOUBLE";
      case NC_UBYTE:   return "NC_UBYTE";
      case NC_USHORT:  return "NC_USHORT";
      case NC_UINT:    return "NC_UINT";
      case NC_INT64:   return "NC_INT64";
      case NC_UINT64:  return "NC_UINT64";
      default:
	derror("nctype: bad type code");
	return 0;
    }
}


/*
 * Return C type name for netCDF type, given type code.
 */
const char *
ncctype(
     nc_type type)			/* netCDF type code */
{
    switch (type) {
      case NC_BYTE:
	return "signed char";
      case NC_CHAR:
	return "char";
      case NC_SHORT:
	return "short";
      case NC_INT:
	return "int";
      case NC_FLOAT:
	return "float";
      case NC_DOUBLE:
	return "double";
      case NC_UBYTE:
	return "unsigned char";
      case NC_USHORT:
	return "unsigned short";
      case NC_UINT:
	return "unsigned int";
      case NC_INT64:
	return "long long";
      case NC_UINT64:
	return "unsigned long long";
      default:
	derror("ncctype: bad type code");
	return 0;
    }
}



/*
 * Return C type name for netCDF type suffix, given type code.
 */
const char *
ncstype(
     nc_type type)			/* netCDF type code */
{
    switch (type) {
      case NC_BYTE:
	return "schar";
      case NC_CHAR:
	return "text";
      case NC_SHORT:
	return "short";
      case NC_INT:
	return "int";
      case NC_FLOAT:
	return "float";
      case NC_DOUBLE:
	return "double";
      case NC_UBYTE:
	return "uchar";
      case NC_USHORT:
	return "ushort";
      case NC_UINT:
	return "uint";
      case NC_INT64:
	return "longlong";
      case NC_UINT64:
	return "ulonglong";
      default:
	derror("ncstype: bad type code");
	return 0;
    }
}


/*
 * Return C type name for netCDF attribute container type, given type code.
 */
const char *
ncatype(
     nc_type type)			/* netCDF type code */
{
    switch (type) {
      case NC_BYTE:
	return "int";		/* avoids choosing between uchar and schar */
      case NC_CHAR:
	return "text";
      case NC_SHORT:
	return "short";
      case NC_INT:
	return "int";
      case NC_FLOAT:
	return "float";
      case NC_DOUBLE:
	return "double";
      case NC_UBYTE: return "uchar";
      case NC_USHORT: return "ushort";
      case NC_UINT: return "uint";
      case NC_INT64: return "ulonglong";
      case NC_UINT64: return "longlong";
      default:
	derror("ncatype: bad type code");
	return 0;
    }
}


/* return internal size for values of specified netCDF type */
MPI_Offset
nctypesize(
     nc_type type)			/* netCDF type code */
{
    switch (type) {
      case NC_BYTE:
	return sizeof(char);
      case NC_CHAR:
	return sizeof(char);
      case NC_SHORT:
	return sizeof(short);
      case NC_INT:
	return sizeof(int);
      case NC_FLOAT:
	return sizeof(float);
      case NC_DOUBLE:
	return sizeof(double);
      case NC_UBYTE:
	return sizeof(unsigned char);
      case NC_USHORT:
	return sizeof(unsigned short);
      case NC_UINT:
	return sizeof(unsigned int);
      case NC_INT64:
	return sizeof(long long);
      case NC_UINT64:
	return sizeof(unsigned long long);
      default:
	derror("nctypesize: bad type code");
	return 0;
    }
}


/*
 * Given a netcdf numeric type, a pointer to a vector of values of that
 * type, and the index of the vector element desired, returns a pointer
 * to a malloced string representing the value in FORTRAN.  Since this
 * may be used in a DATA statement, it must not include non-constant
 * expressions, such as "char(26)".
 */
char *
fstring(
     nc_type type,		/* netCDF type code */
     void *valp,		/* pointer to vector of values */
     int num)			/* element of vector desired */
{
    static char *cp;
    signed char *schp;
    short *shortp;
    int *intp;
    float *floatp;
    double *doublep;

    switch (type) {
      case NC_BYTE:
	cp = (char *) emalloc (10);
	schp = (signed char *)valp;
        sprintf(cp,"%d", schp[num]);
	return cp;

      case NC_SHORT:
	cp = (char *) emalloc (10);
	shortp = (short *)valp;
	(void) sprintf(cp,"%d",* (shortp + num));
	return cp;

      case NC_INT:
	cp = (char *) emalloc (20);
	intp = (int *)valp;
	(void) sprintf(cp,"%d",* (intp + num));
	return cp;

      case NC_FLOAT:
	cp = (char *) emalloc (20);
	floatp = (float *)valp;
	(void) sprintf(cp,"%.8g",* (floatp + num));
	return cp;

      case NC_DOUBLE:
	cp = (char *) emalloc (25);
	doublep = (double *)valp;
	(void) sprintf(cp,"%.16g",* (doublep + num));
	expe2d(cp);	/* change 'e' to 'd' in exponent */
	return cp;

      default:
	derror("fstring: bad type code");
	return 0;
    }
}


/*
 * Given a pointer to a counted string, returns a pointer to a malloced string
 * representing the string as a C constant.
 */
char *
cstrstr(
     const char *valp,		/* pointer to vector of characters*/
     MPI_Offset len)		/* number of characters in valp */
{
    static char *sp;
    char *cp;
    char *istr, *istr0;		/* for null-terminated copy */
    int ii;
    
    if(4*len+3 != (unsigned)(4*len+3)) {
	derror("too much character data!");
	exit(9);
    }
    sp = cp = (char *) emalloc(4*len+3);

    if(len == 1 && *valp == 0) { /* empty string */
	strcpy(sp,"\"\"");
	return sp;
    }

    istr0 = istr = (char *) emalloc(len + 1);
    for(ii = 0; ii < len; ii++) {
	istr[ii] = valp[ii];
    }
    istr[len] = '\0';

    *cp++ = '"';
    for(ii = 0; ii < len; ii++) {
	switch (*istr) {
	case '\0': *cp++ = '\\'; *cp++ = '0'; *cp++ = '0'; *cp++ = '0'; break;
	case '\b': *cp++ = '\\'; *cp++ = 'b'; break;
	case '\f': *cp++ = '\\'; *cp++ = 'f'; break;
	case '\n': *cp++ = '\\'; *cp++ = 'n'; break;
	case '\r': *cp++ = '\\'; *cp++ = 'r'; break;
	case '\t': *cp++ = '\\'; *cp++ = 't'; break;
	case '\v': *cp++ = '\\'; *cp++ = 'v'; break;
	case '\\': *cp++ = '\\'; *cp++ = '\\'; break;
	case '\"': *cp++ = '\\'; *cp++ = '\"'; break;
	default:
	    if (!isprint(*istr)) {
		static char octs[] = "01234567";
		int rem = ((unsigned char)*istr)%64;
		*cp++ = '\\';
		*cp++ = octs[((unsigned char)*istr)/64]; /* to get, e.g. '\177' */
		*cp++ = octs[rem/8];
		*cp++ = octs[rem%8];
	    } else {
		*cp++ = *istr;
	    }
	    break;
	}
	istr++;
    }
    *cp++ = '"';
    *cp = '\0';
    free(istr0);
    return sp;
}


/* Given a pointer to a counted string (not necessarily
 * null-terminated), returns a pointer to a malloced string representing
 * the string as a FORTRAN string expression.  For example, the string
 * "don't" would yield the FORTRAN string "'don''t'", and the string
 * "ab\ncd" would yield "'ab'//char(10)//'cd'".  The common
 * interpretation of "\"-escaped characters is non-standard, so the
 * generated Fortran may require adjustment in compilers that don't
 * recognize "\" as anything special in a character context.  */
char *
fstrstr(
     const char *str,			/* pointer to vector of characters */
     MPI_Offset ilen)			/* number of characters in istr */
{
    static char *ostr;
    char *cp, tstr[12];
    int was_print = 0;		/* true if last character was printable */
    char *istr, *istr0;		/* for null-terminated copy */
    int ii;

    if(12*ilen != (MPI_Offset)(12*ilen)) {
	derror("too much character data!");
	exit(9);
    }
    istr0 = istr = (char *) emalloc(ilen + 1);
    for(ii = 0; ii < ilen; ii++) {
	istr[ii] = str[ii];
    }
    istr[ilen] = '\0';
    
    if (*istr == '\0') {	/* empty string input, not legal in FORTRAN */
	ostr = (char*) emalloc(strlen("char(0)") + 1);
	strcpy(ostr, "char(0)");
	free(istr0);
	return ostr;
    }
    ostr = cp = (char *) emalloc(12*ilen);
    *ostr = '\0';
    if (isprint(*istr)) {	/* handle first character in input */
	*cp++ = '\'';
	switch (*istr) {
	case '\'':
	    *cp++ = '\'';
	    *cp++ = '\'';
	    break;
	case '\\':
	    *cp++ = '\\';
	    *cp++ = '\\';
	    break;
	default:
	    *cp++ = *istr;
	    break;
	}
	*cp = '\0';
	was_print = 1;
    } else {
	sprintf(tstr, "char(%d)", (unsigned char)*istr);
	strcat(cp, tstr);
	cp += strlen(tstr);
	was_print = 0;
    }
    istr++;

    for(ii = 1; ii < ilen; ii++) { /* handle subsequent characters in input */
	if (isprint(*istr)) {
	    if (! was_print) {
		strcat(cp, "//'");
		cp += 3;
	    }
	    switch (*istr) {
	    case '\'':
		*cp++ = '\'';
		*cp++ = '\'';
		break;
	    case '\\':
		*cp++ = '\\';
		*cp++ = '\\';
		break;
	    default:
		*cp++ = *istr;
		break;
	    }
	    *cp = '\0';
	    was_print = 1;
	} else {
	    if (was_print) {
		*cp++ = '\'';
		*cp = '\0';
	    }
	    sprintf(tstr, "//char(%d)", (unsigned char)*istr);
	    strcat(cp, tstr);
	    cp += strlen(tstr);
	    was_print = 0;
	}
	istr++;
    }
    if (was_print)
      *cp++ = '\'';
    *cp = '\0';
    free(istr0);
    return ostr;
}


static void
cl_netcdf(void)
{
    int stat = ncmpi_close(ncid);
    check_err(stat, "ncmpi_close", __func__, __LINE__, __FILE__);
}


static void
cl_c(void)
{
    cline("   stat = ncmpi_close(ncid);");
    cline("   check_err(stat,\"ncmpi_close\", __func__, __LINE__,__FILE__);");
    cline("   MPI_Finalize();");
#ifndef vms
    cline("   return 0;");
#else
    cline("   return 1;");
#endif
    cline("}");
}

/* Returns true if dimension used in at least one record variable,
  otherwise false.  This is an inefficient algorithm, but we don't call
  it very often ... */
static int
used_in_rec_var(
    int idim			/* id of dimension */
    ) {
    int ivar;
    
    for (ivar = 0; ivar < nvars; ivar++) {
	if (vars[ivar].ndims > 0 && vars[ivar].dims[0] == rec_dim) {
	    int jdim;
	    for (jdim = 0; jdim < vars[ivar].ndims; jdim++) {
		if (vars[ivar].dims[jdim] == idim)
		    return 1;
	    }
	}
    }
    return 0;
}


/* Return name for Fortran fill constant of specified type */
static const char *
f_fill_name(
    nc_type type
    )
{
    switch(type) {
    case NC_BYTE:
	return "NF_FILL_BYTE";
    case NC_CHAR:
	return "NF_FILL_CHAR";
    case NC_SHORT:
	return "NF_FILL_SHORT";
    case NC_INT:
	return "NF_FILL_INT";
    case NC_FLOAT:
	return "NF_FILL_FLOAT";
    case NC_DOUBLE:
	return "NF_FILL_DOUBLE";
    default:
	derror("f_fill_name: bad type code");
    }
    return 0;
}


/* Generate Fortran for cleaning up and closing file */
static void
cl_fortran(void)
{
    int ivar;
	    int idim;
    char stmnt[FORT_MAX_STMNT];
    char s2[FORT_MAX_STMNT];
    char*sp;
    int have_rec_var = 0;
    
    /* do we have any record variables? */
    for (ivar = 0; ivar < nvars; ivar++) {
	struct vars *v = &vars[ivar];
        if (v->ndims > 0 && v->dims[0] == rec_dim) {
	    have_rec_var = 1;
            break;
        }
    }        

    if (have_rec_var) {
	fline(" ");
	fline("* Write record variables");
        sprintf(stmnt, "call writerecs(ncid,");
        /* generate parameter list for subroutine to write record vars */
        for (ivar = 0; ivar < nvars; ivar++) {
            struct vars *v = &vars[ivar];
            /* if a record variable, include id in parameter list */
            if (v->ndims > 0 && v->dims[0] == rec_dim) {
                sprintf(s2, "%s_id,", v->lname);
                strcat(stmnt, s2);
            }
        }        
        sp = strrchr(stmnt, ',');
        if(sp != NULL) {
            *sp = '\0';
        }
        strcat(stmnt, ")");
        fline(stmnt);
    }
    
    fline(" ");
    fline("iret = nfmpi_close(ncid)");
    fline("call check_err(iret,\"nfmpi_close\", __func__, __LINE__,__FILE__)");
    fline("end");

    fline(" ");

    if (have_rec_var) {
        sprintf(stmnt, "subroutine writerecs(ncid,");
        for (ivar = 0; ivar < nvars; ivar++) {
            struct vars *v = &vars[ivar];
            if (v->ndims > 0 && v->dims[0] == rec_dim) {
                sprintf(s2, "%s_id,", v->lname);
                strcat(stmnt, s2);
            }
        }        
        sp = strrchr(stmnt, ',');
        if(sp != NULL) {
            *sp = '\0';
        }
        strcat(stmnt, ")");
        fline(stmnt);
	fline(" ");
        fline("* netCDF id");
        fline("integer  ncid");

	fline("* variable ids");
	for (ivar = 0; ivar < nvars; ivar++) {
	    struct vars *v = &vars[ivar];
            if (v->ndims > 0 && v->dims[0] == rec_dim) {
                sprintf(stmnt, "integer  %s_id", v->lname);
                fline(stmnt);
            }
	}

	fline(" ");
        fline("include 'pnetcdf.inc'");

        /* create necessary declarations */
        fline("* error status return");
        fline("integer  iret");

        /* generate integer/parameter declarations for all dimensions
          used in record variables, except record dimension. */
        fline(" ");
        fline("* netCDF dimension sizes for dimensions used with record variables");
        for (idim = 0; idim < ndims; idim++) {
            /* if used in a record variable and not record dimension */
            if (used_in_rec_var(idim) && dims[idim].size != NC_UNLIMITED) {
                sprintf(stmnt, "integer  %s_len", dims[idim].lname);
                fline(stmnt);
                sprintf(stmnt, "parameter (%s_len = %lu)",
                        dims[idim].lname, (unsigned long) dims[idim].size);
                fline(stmnt);
            }
        }

	fline(" ");
	fline("* rank (number of dimensions) for each variable");
	for (ivar = 0; ivar < nvars; ivar++) {
	    struct vars *v = &vars[ivar];
            if (v->ndims > 0 && v->dims[0] == rec_dim) {
                sprintf(stmnt, "integer  %s_rank", v->lname);
                fline(stmnt);
            }
	}
	for (ivar = 0; ivar < nvars; ivar++) {
	    struct vars *v = &vars[ivar];
            if (v->ndims > 0 && v->dims[0] == rec_dim) {
                sprintf(stmnt, "parameter (%s_rank = %d)", v->lname,
                        v->ndims);
                fline(stmnt);
            }
	}

	fline("* starts and counts for array sections of record variables");
	for (ivar = 0; ivar < nvars; ivar++) {
	    struct vars *v = &vars[ivar];
	    if (v->ndims > 0 && v->dims[0] == rec_dim) {
		sprintf(stmnt,
			"integer  %s_start(%s_rank), %s_count(%s_rank)",
			v->lname, v->lname, v->lname, v->lname);
		fline(stmnt);
	    }
	}
        
	fline(" ");
	fline("* data variables");
        
        for (ivar = 0; ivar < nvars; ivar++) {
            struct vars *v = &vars[ivar];
            if (v->ndims > 0 && v->dims[0] == rec_dim) {
                fline(" ");
                sprintf(stmnt, "integer  %s_nr", v->lname);
                fline(stmnt);
                if (v->nrecs > 0) {
                    sprintf(stmnt, "parameter (%s_nr = %lu)",
                            v->lname, (unsigned long) v->nrecs);
                } else {
                    sprintf(stmnt, "parameter (%s_nr = 1)",
                            v->lname);
                }
                fline(stmnt);
		if (v->type != NC_CHAR) {
		    char *sp;
		    sprintf(stmnt, "%s  %s(", ncftype(v->type),
			    v->lname);
		    /* reverse dimensions for FORTRAN */
		    for (idim = v->ndims-1; idim >= 0; idim--) {
			if(v->dims[idim] == rec_dim) {
			    sprintf(s2, "%s_nr, ", v->lname);
			} else {
			    sprintf(s2, "%s_len, ",
				    dims[v->dims[idim]].lname);
			}
			strcat(stmnt, s2);
		    }
		    sp = strrchr(stmnt, ',');
		    if(sp != NULL) {
			*sp = '\0';
		    }
		    strcat(stmnt, ")");
		    fline(stmnt);
		}
            }
        }

        fline(" ");

        /* Emit DATA statements after declarations, because f2c on Linux can't
          handle interspersing them */
        for (ivar = 0; ivar < nvars; ivar++) {
            struct vars *v = &vars[ivar];

            if (v->ndims > 0 && v->dims[0] == rec_dim && v->type != NC_CHAR) {
                if (v->has_data) {
                    fline(v->data_stmnt);
                } else {		/* generate data statement for FILL record */
                    MPI_Offset rec_len = 1;
                    for (idim = 1; idim < v->ndims; idim++) {
                        rec_len *= dims[v->dims[idim]].size;
                    }
                    sprintf(stmnt,"data %s /%lu * %s/", v->lname,
			(unsigned long) rec_len,
                            f_fill_name(v->type));		
                    fline(stmnt);
                }
            }
        }
	fline(" ");
	for (ivar = 0; ivar < nvars; ivar++) {
	    struct vars *v = &vars[ivar];
	    /* if a record variable, declare starts and counts */
	    if (v->ndims > 0 && v->dims[0] == rec_dim) {
		if (!v->has_data)
		    continue;
		sprintf(stmnt, "* store %s", v->name);
		fline(stmnt);

		for (idim = 0; idim < v->ndims; idim++) {
		    sprintf(stmnt, "%s_start(%d) = 1", v->lname, idim+1);
		    fline(stmnt);
		}
		for (idim = v->ndims-1; idim > 0; idim--) {
		    sprintf(stmnt, "%s_count(%d) = %s_len", v->lname,
			    v->ndims - idim, dims[v->dims[idim]].lname);
		    fline(stmnt);
		}
                sprintf(stmnt, "%s_count(%d) = %s_nr", v->lname,
                        v->ndims, v->lname);
		fline(stmnt);
		
		if (v->type != NC_CHAR) {
		    sprintf(stmnt,
			    "iret = nfmpi_put_vara_%s_all(ncid, %s_id, %s_start, %s_count, %s)",
			    nfftype(v->type), v->lname, v->lname, v->lname, v->lname);
		} else {
		    sprintf(stmnt,
			    "iret = nfmpi_put_vara_%s_all(ncid, %s_id, %s_start, %s_count, %s)",
			    nfftype(v->type), v->lname, v->lname, v->lname,
			    v->data_stmnt);
		}
		
		fline(stmnt);
		fline("call check_err(iret,\"nfmpi_put_vara_xxx\", __func__, __LINE__,__FILE__)");
	    }
	}

        fline(" ");

        fline("end");

        fline(" ");
    }

    fline("subroutine check_err(iret, nfmpi_func, calling_func, lineno, calling_file)");
    fline("integer iret");
    fline("character*80 nfmpi_func");
    fline("character*80 calling_func");
    fline("character*80 calling_file");
    fline("integer lineno");
    fline("include 'pnetcdf.inc'");
    fline("if (iret .ne. NF_NOERR) then");
    fline("print *, \"ncmpigen error when calling \",trim(nfmpi_func),\" in \",trim(ncmpi_func),\"() at line \",lineno, \" of \",trim(calling_file),\": \", nfmpi_strerror(iret)");
    fline("stop");
    fline("endif");
    fline("end");
}


/* invoke netcdf calls (or generate C or Fortran code) to create netcdf
 * from in-memory structure. */
void
define_netcdf(
     char *netcdfname)
{
    char *filename;		/* output file name */
    
    if (netcdf_name) {		/* name given on command line */
	filename = netcdf_name;
    } else {			/* construct name from CDL name */
	filename = (char *) emalloc(strlen(netcdfname) + 5);
	(void) strcpy(filename,netcdfname);
	if (netcdf_flag == -1)
	  (void) strcat(filename,".cdf"); /* old, deprecated extension */
	else
	  (void) strcat(filename,".nc"); /* new, favored extension */
    }
    if (netcdf_flag)
      gen_netcdf(filename);	/* create netcdf */
    if (c_flag)			/* create C code to create netcdf */
      gen_c(filename);
    if (fortran_flag)		/* create Fortran code to create netcdf */
      gen_fortran(filename);
    free(filename);
}


void
close_netcdf(void)
{
    if (netcdf_flag)
      cl_netcdf();		/* close netcdf */
    if (c_flag)			/* create C code to close netcdf */
      cl_c();
    if (fortran_flag)		/* create Fortran code to close netcdf */
      cl_fortran();
}


void
check_err(int stat, const char *ncmpi_func, const char *calling_func, int lineno, const char *calling_file) {
    if (stat != NC_NOERR) {
	fprintf(stderr, "ncmpigen error when calling %s in %s() at line %d of %s: %s\n", ncmpi_func, calling_func, lineno, calling_file, ncmpi_strerror(stat));
	derror_count++;
    }
}

/*
 * For logging error conditions.
 */
#ifndef NO_STDARG
void
derror(const char *fmt, ...)
#else
/*VARARGS1*/
void
derror(fmt, va_alist)
     const char *fmt ;		/* error-message printf-style format */
     va_dcl			/* variable number of error args, if any */
#endif /* !NO_STDARG */
{
    va_list args ;


    if (lineno == 1)
      (void) fprintf(stderr,"%s: %s: ", progname, cdlname);
    else  
      (void) fprintf(stderr,"%s: %s line %d: ", progname, cdlname, lineno);

#ifndef NO_STDARG
    va_start(args ,fmt) ;
#else
    va_start(args) ;
#endif /* !NO_STDARG */

    (void) vfprintf(stderr,fmt,args) ;
    va_end(args) ;

    (void) fputc('\n',stderr) ;
    (void) fflush(stderr);	/* to ensure log files are current */
    derror_count++;
}


void *
emalloc (			/* check return from malloc */
	size_t size)
{
    void   *p;

    p = (void *) malloc (size);
    if (p == 0) {
	derror ("out of memory\n");
	exit(3);
    }
    return p;
}

void *
ecalloc (			/* check return from calloc */
	size_t size)
{
    void   *p;

    p = (void *) calloc (size, 1);
    if (p == 0) {
	derror ("out of memory\n");
	exit(3);
    }
    return p;
}

void *
erealloc (		/* check return from realloc */
     void *ptr,
     size_t size)			/* if 0, this is really a free */
{
    void *p;

    p = (void *) realloc (ptr, size);

    if (p == 0 && size != 0) {
 	derror ("out of memory");
	exit(3);
    }
    return p;
}


/*
 * For generated Fortran, change 'e' to 'd' in exponent of double precision
 * constants.
 */
void
expe2d(
    char *cp)			/* string containing double constant */
{
    char *expchar = strrchr(cp,'e');
    if (expchar) {
	*expchar = 'd';
    }
}



/* Returns non-zero if n is a power of 2, 0 otherwise */
static 
int
pow2(
     int n)
{
  int m = n;
  int p = 1;

  while (m > 0) {
    m /= 2;
    p *= 2;
  }
  return p == 2*n;
}


/*
 * Grow an integer array as necessary.
 *
 * Assumption: nar never incremented by more than 1 from last call.
 *
 * Makes sure an array is within a factor of 2 of the size needed.
 *
 * Make sure *arpp points to enough space to hold nar integers.  If not big
 * enough, malloc more space, copy over existing stuff, free old.  When
 * called for first time, *arpp assumed to be uninitialized.
 */
void
grow_iarray(
     int nar,			/* array must be at least this big */
     int **arpp)		/* address of start of int array */
{
  if (nar == 0) {
    *arpp = (int *) emalloc(1 * sizeof(int));
    return;
  }
  if (! pow2(nar))		/* return unless nar is a power of two */
    return;
  *arpp = (int *) erealloc(*arpp, 2 * nar * sizeof(int));
}


/*
 * Grow an array of variables as necessary.
 *
 * Assumption: nar never incremented by more than 1 from last call.
 *
 * Makes sure array is within a factor of 2 of the size needed.
 *
 * Make sure *arpp points to enough space to hold nar variables.  If not big
 * enough, malloc more space, copy over existing stuff, free old.  When
 * called for first time, *arpp assumed to be uninitialized.
 */
void
grow_varray(
     int nar,			/* array must be at least this big */
     struct vars **arpp)	/* address of start of var array */
{
  if (nar == 0) {
    *arpp = (struct vars *) emalloc(1 * sizeof(struct vars));
    return;
  }
  if (! pow2(nar))		/* return unless nar is a power of two */
    return;
  *arpp = (struct vars *) erealloc(*arpp, 2 * nar * sizeof(struct vars));
}


/*
 * Grow an array of dimensions as necessary.
 *
 * Assumption: nar never incremented by more than 1 from last call.
 *
 * Makes sure array is within a factor of 2 of the size needed.
 *
 * Make sure *arpp points to enough space to hold nar dimensions.  If not big
 * enough, malloc more space, copy over existing stuff, free old.  When
 * called for first time, *arpp assumed to be uninitialized.
 */
void
grow_darray(
     int nar,			/* array must be at least this big */
     struct dims **arpp)	/* address of start of var array */
{
  if (nar == 0) {
    *arpp = (struct dims *) emalloc(1 * sizeof(struct dims));
    return;
  }
  if (! pow2(nar))		/* return unless nar is a power of two */
    return;
  *arpp = (struct dims *) erealloc(*arpp, 2 * nar * sizeof(struct dims));
}


/*
 * Grow an array of attributes as necessary.
 *
 * Assumption: nar never incremented by more than 1 from last call.
 *
 * Makes sure array is within a factor of 2 of the size needed.
 *
 * Make sure *arpp points to enough space to hold nar attributes.  If not big
 * enough, malloc more space, copy over existing stuff, free old.  When
 * called for first time, *arpp assumed to be uninitialized.
 */
void
grow_aarray(
     int nar,			/* array must be at least this big */
     struct atts **arpp)	/* address of start of var array */
{
  if (nar == 0) {
    *arpp = (struct atts *) emalloc(1 * sizeof(struct atts));
    return;
  }
  if (! pow2(nar))		/* return unless nar is a power of two */
    return;
  *arpp = (struct atts *) erealloc(*arpp, 2 * nar * sizeof(struct atts));
}


/*
 * Replace dashes and dots in name so it can be used in C and
 * Fortran without causing syntax errors.  Here we just replace each "-"
 * in a name with "_dash_" and each "." with "_dot_", though any
 * similar replacement that doesn't clash with existing names would
 * work.
 */
extern char*
decodify (
    const char *name)
{
    int count=0;		/* number of minus signs in name */
    char *newname;
    const char *cp = name;
    char *sp;

    while(*cp != '\0') {
	switch (*cp) {
	case '-':
	    count += strlen("_dash_") - 1;
	    break;
	case '.':
	    count += strlen("_dot_") - 1;
	    break;
	case '@':
	    count += strlen("_at_") - 1;
	    break;
	case '#':
	    count += strlen("_hash_") - 1;
	    break;
	case '[':
	    count += strlen("_lbr_") - 1;
	    break;
	case ']':
	    count += strlen("_rbr_") - 1;
	    break;
	default:
	    break;
	}
	cp++;
    }
    newname = (char *) ecalloc(strlen(name) + count + 1);
    cp = name;
    sp = newname;
    while(*cp != '\0') {
	switch (*cp) {
	case '-':
	    strcat(sp, "_dash_");
	    sp += strlen("_dash_");
	    break;
	case '.':
	    strcat(sp, "_dot_");
	    sp += strlen("_dot_");
	    break;
	case '@':
	    strcat(sp, "_at_");
	    sp += strlen("_at_");
	    break;
	case '#':
	    strcat(sp, "_hash_");
	    sp += strlen("_hash_");
	    break;
	case '[':
	    strcat(sp, "_lbr_");
	    sp += strlen("_lbr_");
	    break;
	case ']':
	    strcat(sp, "_rbr_");
	    sp += strlen("_rbr_");
	    break;
	default:
	    *sp++ = *cp;
	    break;
	}
	cp++;
    }
    *sp = '\0';
    return newname;
}

/*
 * Replace escaped chars in CDL representation of name such as
 * 'abc\:def\ gh\\i' with unescaped version, such as 'abc:def gh\i'.
 */
void
deescapify (char *name)
{
    const char *cp = name;
    char *sp;
    size_t len = strlen(name);
    char *newname;

    if(strchr(name, '\\') == NULL)
        return;

    newname = (char *) emalloc(len + 1);
    cp = name;
    sp = newname;
    while(*cp != '\0') { /* delete '\' chars, except change '\\' to '\' */
        switch (*cp) {
        case '\\':
            if(*(cp+1) == '\\') {
                *sp++ = '\\';
                cp++;
            }
            break;
        default:
            *sp++ = *cp;
            break;
        }
        cp++;
    }
    *sp = '\0';
    /* assert(strlen(newname) <= strlen(name)); */
    strncpy(name, newname, len);
    free(newname);
    return;
}


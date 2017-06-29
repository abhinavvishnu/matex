/*
 *  Copyright (C) 2014, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id: t_nc.c 2219 2015-12-11 22:30:03Z wkliao $ */

/* This program is based on the test program t_nc.c of the netCDF package */

/* Copyright 1988-2010 University Corporation for Atmospheric Research
   See netcdf/COPYRIGHT file for copying and redistribution
   conditions.

   Program to create a cdf, exercise all cdf functions.  Creates cdf,
   stuff it full of numbers, closes it. Then reopens it, and checks
   for consistency.  Leaves the file around afterwards.
 
   Based on a program to test the nasa look-alike program, so not the
   most appropropriate test. See ../nctest for a complete spec test.
*/
 
#define REDEF
/* #define SYNCDEBUG */

#undef NDEBUG	/* always active assert() in this file */

#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>
#include <pnetcdf.h>

#include <testutils.h>

#define MAXSHORT	32767
#define MAXINT		2147483647
#define MAXBYTE		127


#define	NUM_DIMS 	3
#define DONT_CARE	-1
/* make these numbers big when you want to give this a real workout */
#define NUM_RECS	8
#define SIZE_1		7
#define SIZE_2		8

static struct {
	int num_dims;
	int num_vars;
	int num_attrs;
	int xtendim;
} cdesc[1];

static struct {
	char mnem[NC_MAX_NAME];
	nc_type type;
	int ndims;
	int dims[NC_MAX_VAR_DIMS];
	int num_attrs;
} vdesc[1];

static struct {
	char mnem[NC_MAX_NAME];
	nc_type type;
	MPI_Offset len;
} adesc[1];

union getret
{
    char            by[8];
    short           sh[4];
    int          in[2];
    float           fl[2];
    double          dbl;
};


#define ERR {if (err != NC_NOERR) {printf("Error at %s line %d: %s\n",__func__,__LINE__,ncmpi_strerror(err)); return 1;}}

static void
chkgot(nc_type type, union getret got, double check)
{
	switch(type){
	case NC_BYTE :
		assert( (char)check == got.by[0] );
		break;
	case NC_CHAR :	/* TODO */
		assert( (char)check == got.by[0] );
		break;
	case NC_SHORT :
		assert( (short)check == got.sh[0] );
		break;
	case NC_INT :
		assert( (int)check == got.in[0] );
		break;
	case NC_FLOAT :
		assert( (float)check == got.fl[0] );
		break;
	case NC_DOUBLE :
		assert( check == got.dbl );
		break;
	default:
		break;
	}
}


static size_t num_dims = NUM_DIMS;
static MPI_Offset sizes[] = { NC_UNLIMITED, SIZE_1 , SIZE_2 };
static const char * const dim_names[] = { "record", "ixx", "iyy"};

static int
createtestdims(int cdfid, size_t num_dims, const MPI_Offset *sizes, const char * const dim_names[])
{
	int dimid, err;
	while(num_dims-- != 0)
	{
		err = ncmpi_def_dim(cdfid, *dim_names++, *sizes, &dimid); ERR
		sizes++;
	}
	return 0;
}


static int
testdims(int cdfid, size_t num_dims, MPI_Offset *sizes, const char * const dim_names[])
{
	int ii, err;
	MPI_Offset size;
	char cp[NC_MAX_NAME];
	for(ii=0; (size_t) ii < num_dims; ii++, sizes++)
	{
		err = ncmpi_inq_dim(cdfid, ii, cp, &size); ERR
		if( size != *sizes)
			(void) fprintf(stderr, "%d: %lu != %lu\n",
				ii, (unsigned long)size, (unsigned long)*sizes);
		if ( size != *sizes) return 1;
		if ( strcmp(cp, *dim_names++) != 0) return 1;
	}
	return 0;
}



static const char * const reqattr[] = {
	"UNITS",
	"VALIDMIN",
	"VALIDMAX",
	"SCALEMIN",
	"SCALEMAX",
	"FIELDNAM",
	_FillValue
};
#define NUM_RATTRS	6

static struct tcdfvar {
	const char *mnem;
	nc_type type;
	const char *fieldnam;
	double validmin;
	double validmax;
	double scalemin;
	double scalemax;
	const char *units;
	int ndims;
	int dims[NUM_DIMS];
} const testvars[]  = {
#define Byte_id 0
	{ "Byte", NC_BYTE, "Byte sized integer variable",
		-MAXBYTE, MAXBYTE, -MAXBYTE, MAXBYTE , "ones",
			2, {0,1,DONT_CARE} },
#define Char_id 1
	{ "Char", NC_CHAR, "char (string) variable",
		DONT_CARE, DONT_CARE, DONT_CARE, DONT_CARE, "(unitless)",
			2, {0,2,DONT_CARE} },
#define Short_id 2
	{ "Short", NC_SHORT, "Short variable",
		-MAXSHORT, MAXSHORT, -MAXSHORT, MAXSHORT , "ones",
			2, {0, 2, DONT_CARE }},
#define Long_id 3
	{ "Long", NC_INT, "Long Integer variable", /* 2.x backward strings */
		-MAXINT, MAXINT, -MAXINT, MAXINT, "ones",
			2, {1, 2, DONT_CARE}},
#define Float_id 4
	{ "Float", NC_FLOAT, "Single Precision Floating Point variable",
		-MAXINT, MAXINT, -MAXINT, MAXINT, "flots",
			3, {0, 1, 2 }},
#define Double_id 5
	{ "Double", NC_DOUBLE, "Double Precision Floating Point variable",
		-MAXINT, MAXINT, -MAXINT, MAXINT, "dflots",
			3, {0, 1, 2 }},
};
#define	NUM_TESTVARS	6

static int
createtestvars(int id, const struct tcdfvar *testvars, size_t count)
{
	int ii, err;
	int varid;
	const struct tcdfvar *vp = testvars;

	for(ii = 0; (size_t) ii < count; ii++, vp++ )
	{
		err = ncmpi_def_var(id, vp->mnem, vp->type, vp->ndims, vp->dims, &varid); ERR

	 	err = ncmpi_put_att_text(id,ii,reqattr[0],strlen(vp->units), vp->units); ERR
	 	err = ncmpi_put_att_double(id,ii,reqattr[1],NC_DOUBLE,1, &vp->validmin); ERR
	 	err = ncmpi_put_att_double(id,ii,reqattr[2],NC_DOUBLE,1, &vp->validmax); ERR
	 	err = ncmpi_put_att_double(id,ii,reqattr[3],NC_DOUBLE,1, &vp->scalemin); ERR
	 	err = ncmpi_put_att_double(id,ii,reqattr[4],NC_DOUBLE,1, &vp->scalemax); ERR
	 	err = ncmpi_put_att_text(id,ii,reqattr[5],strlen(vp->fieldnam), vp->fieldnam); ERR
	}
	return 0;
}

/* static void */
/* parray(const char *label, size_t count, const size_t array[]) */
/* { */
/* 	(void) fprintf(stdout, "%s", label); */
/* 	(void) fputc('\t',stdout);	 */
/* 	for(; count != 0; count--, array++) */
/* 		(void) fprintf(stdout," %lu", (unsigned long) *array); */
/* } */


static int
fill_seq(int id)
{
        int err;
	float values[NUM_RECS * SIZE_1 * SIZE_2];
	MPI_Offset vindices[NUM_DIMS];

	{
		size_t ii = 0;
		for(; ii < sizeof(values)/sizeof(values[0]); ii++)
		{
			values[ii] = (float) ii;
		}
	}

	/* zero the vindices */
	{
		MPI_Offset *cc = vindices;
		while (cc < &vindices[num_dims])
			*cc++ = 0; 
	}

	sizes[0] = NUM_RECS;
	err = ncmpi_put_vara_float(id, Float_id, vindices, sizes, values); ERR
	return 0;
}

static int
check_fill_seq(int id)
{
	MPI_Offset vindices[NUM_DIMS];
	MPI_Offset *cc, *mm;
	union getret got;
	int ii = 0;
	/*float val;*/

	sizes[0] = NUM_RECS;
	cc = vindices;
	while (cc < &vindices[num_dims])
		*cc++ = 0; 

	/* ripple counter */
	cc = vindices;
	mm = sizes;
	while (*vindices < *sizes)
	{
	    while (*cc < *mm)
	    {
		if (mm == &sizes[num_dims - 1])
		{
	if(ncmpi_get_var1_float(id, Float_id, vindices, &got.fl[0]) == -1) 
		goto bad_ret;
	/* val = (float) ii;  */
	/* if(val != got.fl[0]) */
	/* { */
	/* 	parray("indices", NUM_DIMS, vindices); */
	/* 	(void) printf("\t%f != %f\n", val, got.fl[0]); */
	/* } */
		    (*cc)++; ii++;
		    continue;
		}
		cc++;
		mm++;
	    }
		if(cc == vindices)
			break;
	    *cc = 0;
	    cc--;
	    mm--;
	    (*cc)++;
	}
	return 0;
bad_ret :
	(void) printf("couldn't get a var in check_fill_seq() %d\n", ii);
	return 1;
}

static MPI_Offset	indices[][3] = {
	{0, 1, 3},
	{0, 3, 0},
	{1, 2, 3},
	{3, 2, 1},
	{2, 1, 3},
	{1, 0, 0},
	{0, 0, 0},
};

static const char chs[] = {'A','B', ((char)0xff) };
static const MPI_Offset s_start[] = {0,1};
static const MPI_Offset s_edges[] = {NUM_RECS, SIZE_1 - 1};
static char sentence[NUM_RECS* SIZE_1 -1] =
	"The red death had long devastated the country.";
static short shs[] = {97, 99};
static int birthday = 82555;
#define M_E	2.7182818284590452354
static float e = (float) M_E;
static double pinot = 3.25;
static double zed = 0.0;


/*ARGSUSED*/
static
int t_nc(char *filename, int cmode)
{
	int id, err;
	char buf[256];
#ifdef SYNCDEBUG
	char *str = "one";
#endif
	int ii;
	MPI_Offset ui;
	const struct tcdfvar *tvp = testvars;
	union getret got;
	MPI_Offset align = 8192/32;

	err = ncmpi_create(MPI_COMM_WORLD, filename,cmode, MPI_INFO_NULL, &id); ERR
	
	err = ncmpi_put_att_text(id, NC_GLOBAL, "TITLE", 12, "another name"); ERR
	err = ncmpi_get_att_text(id, NC_GLOBAL, "TITLE", buf); ERR
/*	(void) printf("title 1 \"%s\"\n", buf); */
	err = ncmpi_put_att_text(id, NC_GLOBAL, "TITLE", strlen(filename), filename); ERR
	err = ncmpi_get_att_text(id, NC_GLOBAL, "TITLE", buf); ERR
	buf[strlen(filename)] = 0;
/*	(void) printf("title 2 \"%s\"\n", buf); */
	assert(strcmp(filename, buf) == 0);

	err = createtestdims(id, NUM_DIMS, sizes, dim_names); ERR
	testdims(id, NUM_DIMS, sizes, dim_names);

	err = createtestvars(id, testvars, NUM_TESTVARS); ERR

 	{
 	int ifill = -1; double dfill = -9999;
 	err = ncmpi_put_att_int(id, Long_id, _FillValue, NC_INT, 1, &ifill); ERR
 	err = ncmpi_put_att_double(id, Double_id, _FillValue, NC_DOUBLE, 1, &dfill); ERR
 	}

#ifdef REDEF
	err = ncmpi__enddef(id, 0, align, 0, 2*align); ERR
	err = ncmpi_begin_indep_data(id); ERR
	err = ncmpi_put_var1_int(id, Long_id, indices[3], &birthday); ERR
	err = fill_seq(id); ERR
	err = ncmpi_end_indep_data(id); ERR

	err = ncmpi_redef(id); ERR
/*	err = ncmpi_rename_dim(id,2, "a long dim name"); ERR */
#endif

	err = ncmpi_rename_dim(id,1, "IXX"); ERR 
	err = ncmpi_inq_dim(id, 1, buf, &ui); ERR 
	/* (void) printf("dimrename: %s\n", buf); */
	err = ncmpi_rename_dim(id,1, dim_names[1]); ERR 

#ifdef ATTRX
	err = ncmpi_rename_att(id, 1, "UNITS", "units"); ERR 
	err = ncmpi_del_att(id, 4, "FIELDNAM"); ERR
	err = ncmpi_del_att(id, 2, "SCALEMIN"); ERR
	err = ncmpi_del_att(id, 2, "SCALEMAX"); ERR
#endif /* ATTRX */

	err = ncmpi__enddef(id, 0, align, 0, 2*align); ERR
	err = ncmpi_begin_indep_data(id); ERR

#ifndef REDEF
	err = fill_seq(id); ERR
	err = ncmpi_put_var1_int(id, Long_id, indices[3], &birthday); ERR
#endif

	err = ncmpi_put_vara_schar(id, Byte_id, s_start, s_edges, (signed char *)sentence); ERR
	err = ncmpi_put_var1_schar(id, Byte_id, indices[6], (signed char *)(chs+1)); ERR
	err = ncmpi_put_var1_schar(id, Byte_id, indices[5], (signed char *)chs); ERR

	err = ncmpi_put_vara_text(id, Char_id, s_start, s_edges, sentence); ERR
	err = ncmpi_put_var1_text(id, Char_id, indices[6], (chs+1)); ERR
	err = ncmpi_put_var1_text(id, Char_id, indices[5], chs); ERR

	err = ncmpi_put_var1_short(id, Short_id, indices[4], shs); ERR

	err = ncmpi_put_var1_float(id, Float_id, indices[2], &e); ERR

	err = ncmpi_put_var1_double(id, Double_id, indices[1], &zed); ERR
	err = ncmpi_put_var1_double(id, Double_id, indices[0], &pinot); ERR

#ifdef SYNCDEBUG
	(void) printf("Hit Return to sync\n");
	gets(str);
	ncmpi_sync(id,0);
	(void) printf("Sync done. Hit Return to continue\n");
	gets(str);
#endif /* SYNCDEBUG */

	err = ncmpi_close(id); ERR


/*
 *	read it
 */
	err = ncmpi_open(MPI_COMM_WORLD, filename,NC_NOWRITE, MPI_INFO_NULL, &id); ERR
	err = ncmpi_begin_indep_data(id); ERR

	/*	NC	*/ 
	/* (void) printf("NC "); */
	err = ncmpi_inq(id, &(cdesc->num_dims), &(cdesc->num_vars), &(cdesc->num_attrs), &(cdesc->xtendim) ); ERR
	assert((MPI_Offset) cdesc->num_dims == num_dims);
	assert(cdesc->num_attrs == 1);
	assert(cdesc->num_vars == NUM_TESTVARS);
	/* (void) printf("done\n"); */
	
	/*	GATTR	*/
	/* (void) printf("GATTR "); */

	err = ncmpi_inq_attname(id, NC_GLOBAL, 0, adesc->mnem); ERR
	assert(strcmp("TITLE",adesc->mnem) == 0);
	err = ncmpi_inq_att(id, NC_GLOBAL, adesc->mnem, &(adesc->type), &(adesc->len)); ERR
	assert( adesc->type == NC_CHAR );
	assert( adesc->len == strlen(filename) );
	err = ncmpi_get_att_text(id, NC_GLOBAL, "TITLE", buf); ERR
	buf[adesc->len] = 0;
	assert( strcmp(filename, buf) == 0);

	/*	VAR	*/
	/* (void) printf("VAR "); */
	assert( cdesc->num_vars == NUM_TESTVARS );

	for(ii = 0; ii < cdesc->num_vars; ii++, tvp++ ) 
	{
		int jj;
		err = ncmpi_inq_var(id, ii,
			vdesc->mnem,
			&(vdesc->type),
			&(vdesc->ndims),
			vdesc->dims,
			&(vdesc->num_attrs)); ERR
		if(strcmp(tvp->mnem , vdesc->mnem) != 0)
		{
			(void) printf("attr %d mnem mismatch %s, %s\n",
				ii, tvp->mnem, vdesc->mnem);
			continue;
		}
		if(tvp->type != vdesc->type)
		{
			(void) printf("attr %d type mismatch %d, %d\n",
				ii, (int)tvp->type, (int)vdesc->type);
			continue;
		}
		for(jj = 0; jj < vdesc->ndims; jj++ )
		{
			if(tvp->dims[jj] != vdesc->dims[jj] )
			{
		(void) printf(
		"inconsistent dim[%d] for variable %d: %d != %d\n",
		jj, ii, tvp->dims[jj], vdesc->dims[jj] );
			continue;
			}
		}

		/* VATTR */
		/* (void) printf("VATTR\n"); */
		for(jj=0; jj<vdesc->num_attrs; jj++ ) 
		{
			err = ncmpi_inq_attname(id, ii, jj, adesc->mnem); ERR
			if( strcmp(adesc->mnem, reqattr[jj]) != 0 )
			{
				(void) printf("var %d attr %d mismatch %s != %s\n",
					ii, jj, adesc->mnem, reqattr[jj] );
				break;
			}
		}

		if( ncmpi_inq_att(id, ii, reqattr[0], &(adesc->type), &(adesc->len))
			!= -1) {
		assert( adesc->type == NC_CHAR );
		assert( adesc->len == strlen(tvp->units) );
	 	err = ncmpi_get_att_text(id,ii,reqattr[0],buf); ERR
		buf[adesc->len] = 0;
		assert( strcmp(tvp->units, buf) == 0);
		}

		if(
			ncmpi_inq_att(id, ii, reqattr[1], &(adesc->type), &(adesc->len))
			!= -1)
		{
		assert( adesc->type == NC_DOUBLE );
		assert( adesc->len == 1 );
	 	err = ncmpi_get_att_double(id, ii, reqattr[1], &got.dbl); ERR
		chkgot(adesc->type, got, tvp->validmin);
		}

		if(
			ncmpi_inq_att(id, ii, reqattr[2], &(adesc->type), &(adesc->len))
			!= -1)
		{
		assert( adesc->type == NC_DOUBLE );
		assert( adesc->len == 1 );
	 	err = ncmpi_get_att_double(id, ii, reqattr[2], &got.dbl); ERR
		chkgot(adesc->type, got, tvp->validmax);
		}

		if(
			ncmpi_inq_att(id, ii, reqattr[3], &(adesc->type), &(adesc->len))
			!= -1)
		{
		assert( adesc->type == NC_DOUBLE );
		assert( adesc->len ==1 );
	 	err = ncmpi_get_att_double(id, ii, reqattr[3], &got.dbl); ERR
		chkgot(adesc->type, got, tvp->scalemin);
		}

		if(
			ncmpi_inq_att(id, ii, reqattr[4], &(adesc->type), &(adesc->len))
			!= -1)
		{
		assert( adesc->type == NC_DOUBLE );
		assert( adesc->len == 1 );
	 	err = ncmpi_get_att_double(id, ii, reqattr[4], &got.dbl); ERR
		chkgot(adesc->type, got, tvp->scalemax);
		}

		if( ncmpi_inq_att(id, ii, reqattr[5], &(adesc->type), &(adesc->len)) == NC_NOERR)
		{
		assert( adesc->type == NC_CHAR );
		assert( adesc->len == strlen(tvp->fieldnam) );
	 	err = ncmpi_get_att_text(id,ii,reqattr[5],buf); ERR
		buf[adesc->len] = 0;
		assert( strcmp(tvp->fieldnam, buf) == 0);
		}
	}

	/* (void) printf("fill_seq "); */
	err = check_fill_seq(id); ERR
	/* (void) printf("Done\n"); */

	err = ncmpi_get_var1_double(id, Double_id, indices[0], &got.dbl); ERR
	/* (void) printf("got val = %f\n", got.dbl ); */

	err = ncmpi_get_var1_double(id, Double_id, indices[1], &got.dbl); ERR
	/* (void) printf("got val = %f\n", got.dbl ); */

	err = ncmpi_get_var1_float(id, Float_id, indices[2], &got.fl[0]); ERR
	/* (void) printf("got val = %f\n", got.fl[0] ); */

	err = ncmpi_get_var1_int(id, Long_id, indices[3], &got.in[0]); ERR
	/* (void) printf("got val = %d\n", got.in[0] ); */

	err = ncmpi_get_var1_short(id, Short_id, indices[4], &got.sh[0]); ERR
	/* (void) printf("got val = %d\n", got.sh[0] ); */

	err = ncmpi_get_var1_text(id, Char_id, indices[5], &got.by[0]); ERR
	/* (void) printf("got NC_CHAR val = %c (0x%02x) \n", */
		 /* got.by[0] , got.by[0]); */

	err = ncmpi_get_var1_text(id, Char_id, indices[6], &got.by[0]); ERR
	/* (void) printf("got NC_CHAR val = %c (0x%02x) \n", */
	/* 	 got.by[0], got.by[0] ); */

	(void) memset(buf,0,sizeof(buf));
	err = ncmpi_get_vara_text(id, Char_id, s_start, s_edges, buf); ERR
	/* (void) printf("got NC_CHAR val = \"%s\"\n", buf); */

	err = ncmpi_get_var1_schar(id, Byte_id, indices[5], (signed char *)&got.by[0]); ERR
	/* (void) printf("got val = %c (0x%02x) \n", got.by[0] , got.by[0]); */

	err = ncmpi_get_var1_schar(id, Byte_id, indices[6], (signed char *)&got.by[0]); ERR
	/* (void) printf("got val = %c (0x%02x) \n", got.by[0], got.by[0] ); */

	(void) memset(buf,0,sizeof(buf));
	err = ncmpi_get_vara_schar(id, Byte_id, s_start, s_edges, (signed char *)buf); ERR
	/* (void) printf("got val = \"%s\"\n", buf); */

	{
		double dbuf[NUM_RECS * SIZE_1 * SIZE_2];
                err = ncmpi_get_var_double(id, Float_id, dbuf); ERR

		/* (void) printf("got vals = %f ... %f\n", dbuf[0], */
		/* 	 dbuf[NUM_RECS * SIZE_1 * SIZE_2 -1] ); */
	}

	err = ncmpi_close(id); ERR

	return 0;
}

int main(int argc, char *argv[])
{
    char filename[256];
    int rank, nprocs, cmode, err, nerrs=0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc > 2) {
        if (!rank) printf("Usage: %s [filename]\n",argv[0]);
        MPI_Finalize();
        return 0;
    }
    strcpy(filename, "testfile.nc");
    if (argc == 2) strcpy(filename, argv[1]);
    MPI_Bcast(filename, 256, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        char cmd_str[256];
        sprintf(cmd_str, "*** TESTING C   %s for emulating netCDF t_nc ", argv[0]);
        printf("%-66s ------ ", cmd_str);
    }

    /* test CDF-1 format */
    cmode = NC_CLOBBER;
    nerrs += t_nc(filename, cmode);

    /* test CDF-2 format */
    cmode = NC_CLOBBER | NC_64BIT_OFFSET;
    nerrs += t_nc(filename, cmode);

    /* test CDF-5 format */
    cmode = NC_CLOBBER | NC_64BIT_DATA;
    nerrs += t_nc(filename, cmode);

    /* check if PnetCDF freed all internal malloc */
    MPI_Offset malloc_size, sum_size;
    err = ncmpi_inq_malloc_size(&malloc_size);
    if (err == NC_NOERR) {
        MPI_Reduce(&malloc_size, &sum_size, 1, MPI_OFFSET, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0 && sum_size > 0)
            printf("heap memory allocated by PnetCDF internally has %lld bytes yet to be freed\n",
                   sum_size);
    }

    MPI_Allreduce(MPI_IN_PLACE, &nerrs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        if (nerrs) printf(FAIL_STR,nerrs);
        else       printf(PASS_STR);
    }

    MPI_Finalize();
    return 0;
}


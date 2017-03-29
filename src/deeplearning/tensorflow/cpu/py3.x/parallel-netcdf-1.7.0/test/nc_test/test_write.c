/*
 *  Copyright (C) 2003, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *  $Id: test_write.c 2289 2016-01-02 08:26:40Z wkliao $
 */

#include "tests.h"
#include "math.h"


/*
 * Test ncmpi_create
 *    For mode in NC_NOCLOBBER|extra_flags, NC_CLOBBER, info do:
 *       create netcdf file 'scratch.nc' with no data, close it
 *       test that it can be opened, do ncmpi_inq to check nvars = 0, etc.
 *    Try again in NC_NOCLOBBER|extra_flags mode, check error return
 * On exit, delete this file
 */
int
test_ncmpi_create(void)
{
    int clobber; /* 0 for NC_NOCLOBBER|extra_flags, 1 for NC_CLOBBER, info */
    int err;
    int ncid;
    int ndims;   /* number of dimensions */
    int nvars;   /* number of variables */
    int ngatts;  /* number of global attributes */
    int recdim;  /* id of unlimited dimension */
    int nok=0;

    for (clobber = 0; clobber < 2; clobber++) {
	err = ncmpi_create(comm, scratch, clobber ? NC_CLOBBER|extra_flags : NC_NOCLOBBER, info, &ncid);
	IF (err != NC_NOERR)
	    error("ncmpi_create: %s", ncmpi_strerror(err));
        ELSE_NOK
	err = ncmpi_close(ncid);
	IF (err != NC_NOERR)
	    error("ncmpi_close: %s", ncmpi_strerror(err));
	err = ncmpi_open(comm, scratch, NC_NOWRITE, info, &ncid);
	IF (err != NC_NOERR)
	    error("ncmpi_open: %s", ncmpi_strerror(err));
	err = ncmpi_inq(ncid, &ndims, &nvars, &ngatts, &recdim);
	IF (err != NC_NOERR)
	    error("ncmpi_inq: %s", ncmpi_strerror(err));
	else IF (ndims != 0)
	    error("ncmpi_inq: wrong number of dimensions returned, %d", ndims);
	else IF (nvars != 0)
	    error("ncmpi_inq: wrong number of variables returned, %d", nvars);
	else IF (ngatts != 0)
	    error("ncmpi_inq: wrong number of global atts returned, %d", ngatts);
	else IF (recdim != -1)
	    error("ncmpi_inq: wrong record dimension ID returned, %d", recdim);
        ELSE_NOK
	err = ncmpi_close(ncid);
	IF (err != NC_NOERR)
	    error("ncmpi_close: %s", ncmpi_strerror(err));
    }

    err = ncmpi_create(comm, scratch, NC_NOCLOBBER|extra_flags, info, &ncid);
    IF (err != NC_EEXIST)
	error("attempt to overwrite file: status = %d", err);
    ELSE_NOK

    err = ncmpi_delete(scratch, info);
    IF (err != NC_NOERR)
	error("remove of %s failed", scratch);
    return nok;
}


/*
 * Test ncmpi_redef 
 * (In fact also tests ncmpi_enddef - called from test_ncmpi_enddef)
 *    BAD_ID
 *    attempt redef (error) & enddef on read-only file
 *    create file, define dims & vars. 
 *    attempt put var (error)
 *    attempt redef (error) & enddef.
 *    put vars
 *    attempt def new dims (error)
 *    redef
 *    def new dims, vars.
 *    put atts
 *    enddef
 *    put vars
 *    close
 *    check file: vars & atts
 */
int
test_ncmpi_redef(void)
{
    int ncid;                   /* netcdf id */
    /* used to force effective test of ncio->move() in redef */
    size_t sizehint = 8192;
    int dimid;         /* dimension id */
    int varid;         /* variable id */
    int varid1;        /* variable id */
    int nok=0, err;
    const char * title = "Not funny";
    double var;
    char name[NC_MAX_NAME];
    MPI_Offset length;

    /* BAD_ID tests */
    err = ncmpi_redef(BAD_ID);
    IF (err != NC_EBADID)
	error("bad ncid: status = %d", err);
    ELSE_NOK
    err = ncmpi_enddef(BAD_ID);
    IF (err != NC_EBADID)
	error("bad ncid: status = %d", err);
    ELSE_NOK

    /* read-only tests */
    err = ncmpi_open(comm, testfile, NC_NOWRITE, info, &ncid);
    IF (err != NC_NOERR)
        error("ncmpi_open: %s", ncmpi_strerror(err));
    err = ncmpi_redef(ncid);
    IF (err != NC_EPERM)
	error("ncmpi_redef in NC_NOWRITE, info mode: status = %d", err);
    ELSE_NOK
    err = ncmpi_enddef(ncid);
    IF (err != NC_ENOTINDEFINE)
	error("ncmpi_endfef in NC_NOWRITE, info mode: status = %d", err);
    ELSE_NOK
    err = ncmpi_close(ncid);
    IF (err != NC_NOERR) 
	error("ncmpi_close: %s", ncmpi_strerror(err));

    /* tests using scratch file */
    err = ncmpi_create(comm, scratch, NC_NOCLOBBER|extra_flags, info, &ncid);
    /* err = ncmpi__create(scratch, NC_NOCLOBBER|extra_flags, 0, &sizehint, &ncid); */
    IF (err != NC_NOERR) {
        error("ncmpi_create: %s", ncmpi_strerror(err));
        return nok;
    }
    /* limit for ncio implementations which which have infinite chunksize */
    if(sizehint > 32768)
	sizehint = 16384;
    def_dims(ncid);
    def_vars(ncid);
    put_atts(ncid);
    err = ncmpi_inq_varid(ncid, "d", &varid);
    IF (err != NC_NOERR) 
	error("ncmpi_inq_varid: %s", ncmpi_strerror(err));
    var = 1.0;
    err = ncmpi_begin_indep_data(ncid);
    IF (err != NC_EINDEFINE)
        error("ncmpi_begin_indep_data... in define mode: status = %d", err);
    err = ncmpi_put_var1_double(ncid, varid, NULL, &var);
    IF (err != NC_EINDEFINE)
        error("ncmpi_put_var... in define mode: status = %d", err);
    err = ncmpi_end_indep_data(ncid);
    IF (err != NC_ENOTINDEP)
        error("ncmpi_end_indep_data... not in indep mode: status = %d", err);
    err = ncmpi_redef(ncid);
    IF (err != NC_EINDEFINE)
        error("ncmpi_redef in define mode: status = %d", err);
    ELSE_NOK
    err = ncmpi_enddef(ncid);
    IF (err != NC_NOERR)
        error("ncmpi_enddef: %s", ncmpi_strerror(err));
    ELSE_NOK
    put_vars(ncid);
    err = ncmpi_def_dim(ncid, "abc", sizehint, &dimid);
    IF (err != NC_ENOTINDEFINE)
        error("ncmpi_def_dim in define mode: status = %d", err);
    err = ncmpi_redef(ncid);
    IF (err != NC_NOERR)
        error("ncmpi_redef: %s", ncmpi_strerror(err));
    ELSE_NOK
    err = ncmpi_set_fill(ncid, NC_NOFILL, NULL);
    IF (err != NC_NOERR)
        error("ncmpi_set_fill: %s", ncmpi_strerror(err));
    err = ncmpi_def_dim(ncid, "abc", sizehint, &dimid);
    IF (err != NC_NOERR)
        error("ncmpi_def_dim: %s", ncmpi_strerror(err));
    err = ncmpi_def_var(ncid, "abcScalar", NC_INT, 0, NULL, &varid);
    IF (err != NC_NOERR)
        error("ncmpi_def_var: %s", ncmpi_strerror(err));
    err = ncmpi_def_var(ncid, "abc", NC_INT, 1, &dimid, &varid1);
    IF (err != NC_NOERR)
        error("ncmpi_def_var: %s", ncmpi_strerror(err));
    {
	int dimids[NDIMS +1];
	int ii = 0;
	for(ii = 0; ii < NDIMS; ii++)
		dimids[ii] = ii;
	dimids[NDIMS] = dimid;
    	err = ncmpi_def_var(ncid, "abcRec", NC_INT, NDIMS, dimids, &varid1);
    	IF (err != NC_NOERR)
        	error("ncmpi_def_var: %s", ncmpi_strerror(err));
    }
    err = ncmpi_put_att_text(ncid, NC_GLOBAL, "title", 1+strlen(title), title);
    IF (err != NC_NOERR)
        error("ncmpi_put_att_text: %s", ncmpi_strerror(err));
    err = ncmpi_enddef(ncid);
    IF (err != NC_NOERR)
        error("ncmpi_enddef: %s", ncmpi_strerror(err));
    ELSE_NOK
    var = 1.0;
    err = ncmpi_end_indep_data(ncid);
    IF (err != NC_ENOTINDEP)
        error("ncmpi_end_indep_data: in collective mode status = %s", ncmpi_strerror(err));
    err = ncmpi_begin_indep_data(ncid);
    IF (err != NC_NOERR) 
	error("ncmpi_begin_indep_data: %s", ncmpi_strerror(err));
    err = ncmpi_put_var1_double(ncid, varid, NULL, &var);
    IF (err != NC_NOERR)
        error("ncmpi_put_var1_double: %s", ncmpi_strerror(err));
    err = ncmpi_end_indep_data(ncid);
    IF (err != NC_NOERR) 
	error("ncmpi_begin_indep_data: %s", ncmpi_strerror(err));
    err = ncmpi_close(ncid);
    IF (err != NC_NOERR) 
	error("ncmpi_close: %s", ncmpi_strerror(err));

	/* check scratch file written as expected */
    check_file(scratch);
    err = ncmpi_open(comm, scratch, NC_NOWRITE, info, &ncid);
    IF (err != NC_NOERR)
        error("ncmpi_open: %s", ncmpi_strerror(err));
    err = ncmpi_inq_dim(ncid, dimid, name, &length);
    IF (err != NC_NOERR) 
	error("ncmpi_inq_dim: %s", ncmpi_strerror(err));
    IF (strcmp(name, "abc") != 0) 
	error("Unexpected dim name");
    IF (length != sizehint) 
	error("Unexpected dim length");
    ncmpi_begin_indep_data(ncid);
    err = ncmpi_get_var1_double(ncid, varid, NULL, &var);
    IF (err != NC_NOERR)
        error("ncmpi_get_var1_double: %s", ncmpi_strerror(err));
    ncmpi_end_indep_data(ncid);
    IF (var != 1.0)
        error("ncmpi_get_var1_double: unexpected value");
    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
        error("ncmpi_close: %s", ncmpi_strerror(err));

    err = ncmpi_delete(scratch, info);
    IF (err != NC_NOERR)
        error("remove of %s failed", scratch);
    return nok;
}


/*
 * Test ncmpi_enddef 
 * Simply calls test_ncmpi_redef which tests both ncmpi_redef & ncmpi_enddef
 */
int
test_ncmpi_enddef(void)
{
    return test_ncmpi_redef();
}


/*
 * Test ncmpi_sync
 *    try with bad handle, check error
 *    try in define mode, check error
 *    try writing with one handle, reading with another on same netCDF
 */
int
test_ncmpi_sync(void)
{
    int ncidw;         /* netcdf id for writing */
    int ncidr;         /* netcdf id for reading */
    int nok=0, err;

    /* BAD_ID test */
    err = ncmpi_sync(BAD_ID);
    IF (err != NC_EBADID)
        error("bad ncid: status = %d", err);
    ELSE_NOK

        /* create scratch file & try ncmpi_sync in define mode */
    err = ncmpi_create(comm, scratch, NC_NOCLOBBER|extra_flags, info, &ncidw);
    IF (err != NC_NOERR) {
        error("ncmpi_create: %s", ncmpi_strerror(err));
	return nok;
    }
    err = ncmpi_sync(ncidw);
    IF (err != NC_EINDEFINE)
        error("ncmpi_sync called in define mode: status = %d", err);
    ELSE_NOK

    /* write using same handle */
    def_dims(ncidw);
    def_vars(ncidw);
    put_atts(ncidw);
    err = ncmpi_enddef(ncidw);
    IF (err != NC_NOERR)
        error("ncmpi_enddef: %s", ncmpi_strerror(err));
    put_vars(ncidw);
    err = ncmpi_sync(ncidw);
    IF (err != NC_NOERR)
        error("ncmpi_sync of ncidw failed: %s", ncmpi_strerror(err));
    ELSE_NOK

    /* open another handle, ncmpi_sync, read (check) */
    err = ncmpi_open(comm, scratch, NC_NOWRITE, info, &ncidr);
    IF (err != NC_NOERR)
        error("ncmpi_open: %s", ncmpi_strerror(err));
    err = ncmpi_sync(ncidr);
    IF (err != NC_NOERR)
        error("ncmpi_sync of ncidr failed: %s", ncmpi_strerror(err));
    ELSE_NOK
    check_dims(ncidr);
    check_atts(ncidr);
    check_vars(ncidr);

    /* close both handles */
    err = ncmpi_close(ncidr);
    IF (err != NC_NOERR)
        error("ncmpi_close: %s", ncmpi_strerror(err));
    err = ncmpi_close(ncidw);
    IF (err != NC_NOERR)
        error("ncmpi_close: %s", ncmpi_strerror(err));

    err = ncmpi_delete(scratch, info);
    IF (err != NC_NOERR)
        error("remove of %s failed", scratch);
    return nok;
}


/*
 * Test ncmpi_abort
 *    try with bad handle, check error
 *    try in define mode before anything written, check that file was deleted
 *    try after ncmpi_enddef, ncmpi_redef, define new dims, vars, atts
 *    try after writing variable
 */
int
test_ncmpi_abort(void)
{
    int ncid;          /* netcdf id */
    int err;
    int ndims;
    int nvars;
    int ngatts;
    int nok=0;

    /* BAD_ID test */
    err = ncmpi_abort(BAD_ID);
    IF (err != NC_EBADID)
        error("bad ncid: status = %d", err);
    ELSE_NOK

        /* create scratch file & try ncmpi_abort in define mode */
    err = ncmpi_create(comm, scratch, NC_NOCLOBBER|extra_flags, info, &ncid);
    IF (err != NC_NOERR) {
        error("ncmpi_create: %s", ncmpi_strerror(err));
        return nok;
    }
    def_dims(ncid);
    def_vars(ncid);
    put_atts(ncid);
    err = ncmpi_abort(ncid);
    IF (err != NC_NOERR)
        error("ncmpi_abort of ncid failed: %s", ncmpi_strerror(err));
    ELSE_NOK
    err = ncmpi_close(ncid);	/* should already be closed */
    IF (err != NC_EBADID)
        error("bad ncid: status = %d", err);
    err = ncmpi_delete(scratch, info);	/* should already be deleted */
    IF (!err) /* err is expected to be NC_ENOENT */
        error("file %s should not exist", scratch);

        /* 
         * create scratch file
	 * do ncmpi_enddef & ncmpi_redef
	 * define new dims, vars, atts
	 * try ncmpi_abort: should restore previous state (no dims, vars, atts)
	 */ 
    err = ncmpi_create(comm, scratch, NC_NOCLOBBER|extra_flags, info, &ncid);
    IF (err != NC_NOERR) {
        error("ncmpi_create: %s", ncmpi_strerror(err));
        return nok;
    }
    err = ncmpi_enddef(ncid);
    IF (err != NC_NOERR)
        error("ncmpi_enddef: %s", ncmpi_strerror(err));
    err = ncmpi_redef(ncid);
    IF (err != NC_NOERR)
        error("ncmpi_redef: %s", ncmpi_strerror(err));
    def_dims(ncid);
    def_vars(ncid);
    put_atts(ncid);
    err = ncmpi_abort(ncid);
    IF (err != NC_NOERR)
        error("ncmpi_abort of ncid failed: %s", ncmpi_strerror(err));
    ELSE_NOK
    err = ncmpi_close(ncid);	/* should already be closed */
    IF (err != NC_EBADID)
        error("bad ncid: status = %d", err);
    err = ncmpi_open(comm, scratch, NC_NOWRITE, info, &ncid);
    IF (err != NC_NOERR)
        error("ncmpi_open: %s", ncmpi_strerror(err));
    err = ncmpi_inq (ncid, &ndims, &nvars, &ngatts, NULL);
    IF (err != NC_NOERR)
        error("ncmpi_inq: %s", ncmpi_strerror(err));
    IF (ndims != 0)
        error("ndims should be 0");
    IF (nvars != 0)
        error("nvars should be 0");
    IF (ngatts != 0)
        error("ngatts should be 0");
    err = ncmpi_close (ncid);
    IF (err != NC_NOERR)
        error("ncmpi_close: %s", ncmpi_strerror(err));

    /* try ncmpi_abort in data mode - should just close */
    err = ncmpi_create(comm, scratch, NC_CLOBBER|extra_flags, info, &ncid);
    IF (err != NC_NOERR) {
        error("ncmpi_create: %s", ncmpi_strerror(err));
        return nok;
    }
    def_dims(ncid);
    def_vars(ncid);
    put_atts(ncid);
    err = ncmpi_enddef(ncid);
    IF (err != NC_NOERR)
        error("ncmpi_enddef: %s", ncmpi_strerror(err));
    put_vars(ncid);
    err = ncmpi_abort(ncid);
    IF (err != NC_NOERR)
        error("ncmpi_abort of ncid failed: %s", ncmpi_strerror(err));
    ELSE_NOK
    err = ncmpi_close(ncid);       /* should already be closed */
    IF (err != NC_EBADID)
        error("bad ncid: status = %d", err);
    check_file(scratch);
    err = ncmpi_delete(scratch, info);
    IF (err != NC_NOERR)
        error("remove of %s failed", scratch);
    return nok;
}


/*
 * Test ncmpi_def_dim
 *    try with bad netCDF handle, check error
 *    try in data mode, check error
 *    check that returned id is one more than previous id
 *    try adding same dimension twice, check error
 *    try with illegal sizes, check error
 *    make sure unlimited size works, shows up in ncmpi_inq_unlimdim
 *    try to define a second unlimited dimension, check error
 */
int
test_ncmpi_def_dim(void)
{
    int ncid;
    int  err;             /* status */
    int  i, nok=0;
    int  dimid;         /* dimension id */
    MPI_Offset length;

        /* BAD_ID test */
    err = ncmpi_def_dim(BAD_ID, "abc", 8, &dimid);
    IF (err != NC_EBADID)
        error("bad ncid: status = %d", err);
    ELSE_NOK

        /* data mode test */
    err = ncmpi_create(comm, scratch, NC_CLOBBER|extra_flags, info, &ncid);
    IF (err != NC_NOERR) {
        error("ncmpi_create: %s", ncmpi_strerror(err));
        return nok;
    }
    err = ncmpi_enddef(ncid);
    IF (err != NC_NOERR)
        error("ncmpi_enddef: %s", ncmpi_strerror(err));
    err = ncmpi_def_dim(ncid, "abc", 8, &dimid);
    IF (err != NC_ENOTINDEFINE)
        error("bad ncid: status = %d", err);
    ELSE_NOK

    /* define-mode tests: unlimited dim */
    err = ncmpi_redef(ncid);
    IF (err != NC_NOERR)
        error("ncmpi_redef: %s", ncmpi_strerror(err));
    err = ncmpi_def_dim(ncid, dim_name[0], NC_UNLIMITED, &dimid);
    IF (err != NC_NOERR) 
	error("ncmpi_def_dim: %s", ncmpi_strerror(err));
    ELSE_NOK
    IF (dimid != 0) 
	error("Unexpected dimid");
    ELSE_NOK
    err = ncmpi_inq_unlimdim(ncid, &dimid);
    IF (err != NC_NOERR) 
	error("ncmpi_inq_unlimdim: %s", ncmpi_strerror(err));
    IF (dimid != 0) 
	error("Unexpected recdim");
    err = ncmpi_inq_dimlen(ncid, dimid, &length);
    IF (length != 0) 
	error("Unexpected length");
    err = ncmpi_def_dim(ncid, "abc", NC_UNLIMITED, &dimid);
    IF (err != NC_EUNLIMIT)
        error("2nd unlimited dimension: status = %d", err);
    ELSE_NOK

    /* define-mode tests: remaining dims */
    for (i = 1; i < NDIMS; i++) {
        err = ncmpi_def_dim(ncid, dim_name[i-1], dim_len[i], &dimid);
	IF (err != NC_ENAMEINUSE)
	    error("duplicate name: status = %d", err);
        ELSE_NOK
	err = ncmpi_def_dim(ncid, BAD_NAME, dim_len[i], &dimid);
	IF (err != NC_EBADNAME)
	    error("bad name: status = %d", err);
        ELSE_NOK
        err = ncmpi_def_dim(ncid, dim_name[i], NC_UNLIMITED-1, &dimid);
	IF (err != NC_EDIMSIZE)
	    error("bad size: status = %d", err);
        ELSE_NOK
        err = ncmpi_def_dim(ncid, dim_name[i], dim_len[i], &dimid);
        IF (err != NC_NOERR) 
	    error("ncmpi_def_dim: %s", ncmpi_strerror(err));
        ELSE_NOK
	IF (dimid != i) 
	    error("Unexpected dimid");
    }

        /* Following just to expand unlimited dim */
    def_vars(ncid);
    err = ncmpi_enddef(ncid);
    IF (err != NC_NOERR)
        error("ncmpi_enddef: %s", ncmpi_strerror(err));
    put_vars(ncid);

        /* Check all dims */
    check_dims(ncid);

    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
	error("ncmpi_close: %s", ncmpi_strerror(err));
    err = ncmpi_delete(scratch, info);
    IF (err != NC_NOERR)
        error("remove of %s failed", scratch);
    return nok;
}


/*
 * Test ncmpi_rename_dim
 *    try with bad netCDF handle, check error
 *    check that proper rename worked with ncmpi_inq_dim
 *    try renaming to existing dimension name, check error
 *    try with bad dimension handle, check error
 */
int
test_ncmpi_rename_dim(void)
{
    int ncid;
    int  err, nok=0;             /* status */
    char name[NC_MAX_NAME];

        /* BAD_ID test */
    err = ncmpi_rename_dim(BAD_ID, 0, "abc");
    IF (err != NC_EBADID)
        error("bad ncid: status = %d", err);
    ELSE_NOK

        /* main tests */
    err = ncmpi_create(comm, scratch, NC_NOCLOBBER|extra_flags, info, &ncid);
    IF (err != NC_NOERR) {
        error("ncmpi_create: %s", ncmpi_strerror(err));
        return nok;
    }
    def_dims(ncid);
    err = ncmpi_rename_dim(ncid, BAD_DIMID, "abc");
    IF (err != NC_EBADDIM)
        error("bad dimid: status = %d", err);
    ELSE_NOK
    err = ncmpi_rename_dim(ncid, 2, "abc");
    IF (err != NC_NOERR)
        error("ncmpi_rename_dim: %s", ncmpi_strerror(err));
    ELSE_NOK
    err = ncmpi_inq_dimname(ncid, 2, name);
    IF (strcmp(name, "abc") != 0)
        error("Unexpected name: %s", name);
    err = ncmpi_rename_dim(ncid, 0, "abc");
    IF (err != NC_ENAMEINUSE)
        error("duplicate name: status = %d", err);
    ELSE_NOK

    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
        error("ncmpi_close: %s", ncmpi_strerror(err));
    err = ncmpi_delete(scratch, info);
    IF (err != NC_NOERR)
        error("remove of %s failed", scratch);
    return nok;
}


/*
 * Test ncmpi_def_var
 *    try with bad netCDF handle, check error
 *    try with bad name, check error
 *    scalar tests:
 *      check that proper define worked with ncmpi_inq_var
 *      try redefining an existing variable, check error
 *      try with bad datatype, check error
 *      try with bad number of dimensions, check error
 *      try in data mode, check error
 *    check that returned id is one more than previous id
 *    try with bad dimension ids, check error
 */
int
test_ncmpi_def_var(void)
{
    int  ncid;
    int  varid;
    int  err, nok=0;             /* status */
    int  i;
    int  ndims;
    int  natts;
    char name[NC_MAX_NAME];
    int dimids[MAX_RANK];
    nc_type datatype;

        /* BAD_ID test */
    err = ncmpi_def_var(BAD_ID, "abc", NC_SHORT, 0, NULL, &varid);
    IF (err != NC_EBADID)
        error("bad ncid: status = %d", err);
    ELSE_NOK

        /* scalar tests */
    err = ncmpi_create(comm, scratch, NC_NOCLOBBER|extra_flags, info, &ncid);
    IF (err != NC_NOERR) {
        error("ncmpi_create: %s", ncmpi_strerror(err));
        return nok;
    }
    err = ncmpi_def_var(ncid, "abc", NC_SHORT, 0, NULL, &varid);
    IF (err != NC_NOERR)
        error("ncmpi_def_var: %s", ncmpi_strerror(err));
    ELSE_NOK
    err = ncmpi_inq_var(ncid, varid, name, &datatype, &ndims, dimids, &natts);
    IF (err != NC_NOERR)
        error("ncmpi_inq_var: %s", ncmpi_strerror(err));
    IF (strcmp(name, "abc") != 0)
        error("Unexpected name: %s", name);
    IF (datatype != NC_SHORT)
        error("Unexpected datatype");
    IF (ndims != 0)
        error("Unexpected rank");
    err = ncmpi_def_var(ncid, BAD_NAME, NC_SHORT, 0, NULL, &varid);
    IF (err != NC_EBADNAME)
        error("bad name: status = %d", err);
    ELSE_NOK
    err = ncmpi_def_var(ncid, "abc", NC_SHORT, 0, NULL, &varid);
    IF (err != NC_ENAMEINUSE)
        error("duplicate name: status = %d", err);
    ELSE_NOK
    err = ncmpi_def_var(ncid, "ABC", BAD_TYPE, -1, dimids, &varid);
    IF (err != NC_EBADTYPE)
        error("bad type: status = %d", err);
    ELSE_NOK
    err = ncmpi_def_var(ncid, "ABC", NC_SHORT, -1, dimids, &varid);
    IF (err != NC_EINVAL)
        error("bad rank: status = %d", err);
    ELSE_NOK
    err = ncmpi_enddef(ncid);
    IF (err != NC_NOERR)
	error("ncmpi_enddef: %s", ncmpi_strerror(err));
    err = ncmpi_def_var(ncid, "ABC", NC_SHORT, 0, dimids, &varid);
    IF (err != NC_ENOTINDEFINE)
        error("ncmpi_def_var called in data mode: status = %d", err);
    ELSE_NOK
    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
        error("ncmpi_close: %s", ncmpi_strerror(err));
    err = ncmpi_delete(scratch, info);
    IF (err != NC_NOERR)
        error("remove of %s failed", scratch);

    /* general tests using global vars */
    err = ncmpi_create(comm, scratch, NC_CLOBBER|extra_flags, info, &ncid);
    IF (err != NC_NOERR) {
        error("ncmpi_create: %s", ncmpi_strerror(err));
        return nok;
    }
    def_dims(ncid);
    for (i = 0; i < numVars; i++) {
        err = ncmpi_def_var(ncid, var_name[i], var_type[i], var_rank[i],
            var_dimid[i], &varid);
        IF (err != NC_NOERR) 
	    error("ncmpi_def_var: %s", ncmpi_strerror(err));
        ELSE_NOK
	IF (varid != i)
	    error("Unexpected varid");
        ELSE_NOK
    }

    /* try bad dim ids */
    dimids[0] = BAD_DIMID;
    err = ncmpi_def_var(ncid, "abc", NC_SHORT, 1, dimids, &varid);
    IF (err != NC_EBADDIM)
        error("bad dim ids: status = %d", err);
    ELSE_NOK
    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
        error("ncmpi_close: %s", ncmpi_strerror(err));

    err = ncmpi_delete(scratch, info);
    IF (err != NC_NOERR)
        error("remove of %s failed", scratch);
    return nok;
}


/*
 * Test ncmpi_put_var1
 */
int
test_ncmpi_put_var1(void)
{
    int i, j, err, ncid, nok=0;
    MPI_Offset index[MAX_RANK];
    MPI_Datatype buftype;
    double value;
    double buf[1];		/* (void *) buffer */

    err = ncmpi_create(comm, scratch, NC_NOCLOBBER|extra_flags, info, &ncid);
    IF (err != NC_NOERR) {
        error("ncmpi_create: %s", ncmpi_strerror(err));
        return nok;
    }
    def_dims(ncid);
    def_vars(ncid);
    err = ncmpi_enddef(ncid);
    IF (err != NC_NOERR)
        error("ncmpi_enddef: %s", ncmpi_strerror(err));

    /* var1 is only available for indep mode */
    err = ncmpi_begin_indep_data(ncid);
    IF (err != NC_NOERR)
        error("ncmpi_begin_indep_data: %s", ncmpi_strerror(err));

    for (i = 0; i < numVars; i++) {
        buftype = nc_mpi_type(var_type[i]);
        for (j = 0; j < var_rank[i]; j++)
            index[j] = 0;
        err = ncmpi_put_var1(BAD_ID, i, index, buf, 1, buftype);
        IF (err != NC_EBADID)
            error("bad ncid: status = %d", err);
        ELSE_NOK
        err = ncmpi_put_var1(ncid, BAD_VARID, index, buf, 1, buftype);
        IF (err != NC_ENOTVAR)
            error("bad var id: status = %d", err);
        ELSE_NOK
        for (j = 0; j < var_rank[i]; j++) {
            if (var_dimid[i][j] > 0) {          /* skip record dim */
                index[j] = var_shape[i][j];
                err = ncmpi_put_var1(ncid, i, index, buf, 1, buftype);
                IF (err != NC_EINVALCOORDS)
                    error("bad index: status = %d", err);
                ELSE_NOK
                index[j] = 0;
            }
        }
        for (j = 0; j < var_nels[i]; j++) {
            err = toMixedBase(j, var_rank[i], var_shape[i], index);
            IF (err != NC_NOERR)
                error("error in toMixedBase");
            value = hash(var_type[i], var_rank[i], index);
	    if (inRange(value, var_type[i])) {
		err = dbl2nc(value, var_type[i], buf);
		IF (err != NC_NOERR)
		    error("error in dbl2nc");
		if (var_rank[i] == 0 && i%2 == 0)
		    err = ncmpi_put_var1(ncid, i, NULL, buf, 1, buftype);
		else
		    err = ncmpi_put_var1(ncid, i, index, buf, 1, buftype);
		IF (err != NC_NOERR)
		    error("%s", ncmpi_strerror(err));
                ELSE_NOK
	    }
        }
    }
    err = ncmpi_end_indep_data(ncid);
    IF (err != NC_NOERR)
        error("ncmpi_end_indep_data: %s", ncmpi_strerror(err));

    check_vars(ncid);
    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
        error("ncmpi_close: %s", ncmpi_strerror(err));

    err = ncmpi_delete(scratch, info);
    IF (err != NC_NOERR)
        error("remove of %s failed", scratch);
    return nok;
}


/*
 * Test ncmpi_put_vara
 * Choose a random point dividing each dim into 2 parts
 * Put 2^rank (nslabs) slabs so defined
 * Redefine buffer for each put.
 * At end check all variables using check_vars
 */
int
test_ncmpi_put_vara(void)
{
    int d, i, j, k, err, nels,nslabs, ncid, nok=0;
    MPI_Offset start[MAX_RANK];
    MPI_Offset edge[MAX_RANK];
    MPI_Offset index[MAX_RANK];
    MPI_Offset mid[MAX_RANK];
    MPI_Offset bufcount;
    MPI_Datatype buftype;
    double buf[MAX_NELS]; 	/* (void *) buffer */
    char *p;			/* (void *) pointer */
    double value;

    err = ncmpi_create(comm, scratch, NC_NOCLOBBER|extra_flags, info, &ncid);
    IF (err != NC_NOERR) {
        error("ncmpi_create: %s", ncmpi_strerror(err));
        return nok;
    }
    def_dims(ncid);
    def_vars(ncid);
    err = ncmpi_enddef(ncid);
    IF (err != NC_NOERR)
        error("ncmpi_enddef: %s", ncmpi_strerror(err));

    for (i = 0; i < numVars; i++) {
        assert(var_rank[i] <= MAX_RANK);
        assert(var_nels[i] <= MAX_NELS);
        for (j = 0; j < var_rank[i]; j++) {
            start[j] = 0;
            edge[j] = 1;
        }
        buftype = nc_mpi_type(var_type[i]);
        for (bufcount=1,j=0; j<var_rank[i]; j++) bufcount *= edge[j];
        err = ncmpi_put_vara_all(BAD_ID, i, start, edge, buf, bufcount, buftype);
        IF (err != NC_EBADID)
            error("expecting bad ncid: status = %d", err);
        ELSE_NOK
        err = ncmpi_put_vara_all(ncid, BAD_VARID, start, edge, buf, bufcount, buftype);
        IF (err != NC_ENOTVAR)
            error("expecting bad var id: status = %d", err);
        ELSE_NOK
        for (j = 0; j < var_rank[i]; j++) {
            if (var_dimid[i][j] > 0) {          /* skip record dim */
		start[j] = var_shape[i][j];
                for (bufcount=1,k=0; k<var_rank[i]; k++) bufcount *= edge[k];
		err = ncmpi_put_vara_all(ncid, i, start, edge, buf, bufcount, buftype);
                IF (err != NC_EINVALCOORDS)
                    error("expecting bad start, but err = %d", err);
                ELSE_NOK
		start[j] = 0;
		edge[j] = var_shape[i][j] + 1;
                for (bufcount=1,k=0; k<var_rank[i]; k++) bufcount *= edge[k];
		err = ncmpi_put_vara_all(ncid, i, start, edge, buf, bufcount, buftype);
                IF (err != NC_EEDGE)
                    error("expecting bad edge, but err = %d", err);
                ELSE_NOK
		edge[j] = 1;
	    }
        }
            /* Choose a random point dividing each dim into 2 parts */
            /* put 2^rank (nslabs) slabs so defined */
        nslabs = 1;
        for (j = 0; j < var_rank[i]; j++) {
            mid[j] = roll( var_shape[i][j] );
            nslabs *= 2;
        }
            /* bits of k determine whether to put lower or upper part of dim */
        for (k = 0; k < nslabs; k++) {
            nels = 1;
            for (j = 0; j < var_rank[i]; j++) {
                if ((k >> j) & 1) {
                    start[j] = 0;
                    edge[j] = mid[j];
                }else{
                    start[j] = mid[j];
                    edge[j] = var_shape[i][j] - mid[j];
                }
                nels *= edge[j];
            }
	    p = (char *) buf;
	    for (j = 0; j < nels; j++) {
		err = toMixedBase(j, var_rank[i], edge, index);
		IF (err != NC_NOERR)
		    error("error in toMixedBase");
		for (d = 0; d < var_rank[i]; d++)
		    index[d] += start[d];
		value = hash(var_type[i], var_rank[i], index);
		if (!inRange(value, var_type[i]))
		    value = 0;
		err = dbl2nc(value, var_type[i], p);
		IF (err != NC_NOERR)
		    error("error in dbl2nc");
		p += nctypelen(var_type[i]);
            }
            for (bufcount=1,j=0; j<var_rank[i]; j++) bufcount *= edge[j];
            if (var_rank[i] == 0 && i%2 == 0)
		err = ncmpi_put_vara_all(ncid, i, NULL, NULL, buf, bufcount, buftype);
            else
		err = ncmpi_put_vara_all(ncid, i, start, edge, buf, bufcount, buftype);
            IF (err != NC_NOERR)
                error("%s", ncmpi_strerror(err));
            ELSE_NOK
        }
    }

    check_vars(ncid);
    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
        error("ncmpi_close: %s", ncmpi_strerror(err));

    err = ncmpi_delete(scratch, info);
    IF (err != NC_NOERR)
        error("remove of %s failed", scratch);
    return nok;
}


/*
 * Test ncmpi_put_vars
 * Choose a random point dividing each dim into 2 parts
 * Put 2^rank (nslabs) slabs so defined
 * Choose random stride from 1 to edge
 * Redefine buffer for each put.
 * At end check all variables using check_vars
 */
int
test_ncmpi_put_vars(void)
{
    int ncid, d, i, j, k, m, err, nels, nslabs, nok=0;
    int nstarts;        /* number of different starts */
    MPI_Offset start[MAX_RANK];
    MPI_Offset edge[MAX_RANK];
    MPI_Offset index[MAX_RANK];
    MPI_Offset index2[MAX_RANK];
    MPI_Offset mid[MAX_RANK];
    MPI_Offset count[MAX_RANK];
    MPI_Offset sstride[MAX_RANK];
    MPI_Offset stride[MAX_RANK];
    MPI_Offset bufcount;
    MPI_Datatype buftype;
    double buf[MAX_NELS]; /* (void *) buffer */
    char *p;			/* (void *) pointer */
    double value;

    err = ncmpi_create(comm, scratch, NC_NOCLOBBER|extra_flags, info, &ncid);
    IF (err != NC_NOERR) {
        error("ncmpi_create: %s", ncmpi_strerror(err));
        return nok;
    }
    def_dims(ncid);
    def_vars(ncid);
    err = ncmpi_enddef(ncid);
    IF (err != NC_NOERR)
        error("ncmpi_enddef: %s", ncmpi_strerror(err));

    for (i = 0; i < numVars; i++) {
        assert(var_rank[i] <= MAX_RANK);
        assert(var_nels[i] <= MAX_NELS);
        for (j = 0; j < var_rank[i]; j++) {
            start[j] = 0;
            edge[j] = 1;
            stride[j] = 1;
        }
        buftype = nc_mpi_type(var_type[i]);
        for (bufcount=1,j=0; j<var_rank[i]; j++) bufcount *= edge[j];
        err = ncmpi_put_vars_all(BAD_ID, i, start, edge, stride, buf, bufcount, buftype);
        IF (err != NC_EBADID)
            error("bad ncid: status = %d", err);
        ELSE_NOK
        err = ncmpi_put_vars_all(ncid, BAD_VARID, start, edge, stride, buf, bufcount, buftype);
        IF (err != NC_ENOTVAR)
            error("bad var id: status = %d", err);
        ELSE_NOK
        for (j = 0; j < var_rank[i]; j++) {
            if (var_dimid[i][j] > 0) {          /* skip record dim */
		start[j] = var_shape[i][j];
                for (bufcount=1,k=0; k<var_rank[i]; k++) bufcount *= edge[k];
		err = ncmpi_put_vars_all(ncid, i, start, edge, stride, buf, bufcount, buftype);
		IF (err != NC_EINVALCOORDS)
		    error("bad index: status = %d", err);
                ELSE_NOK
		start[j] = 0;
		edge[j] = var_shape[i][j] + 1;
                for (bufcount=1,k=0; k<var_rank[i]; k++) bufcount *= edge[k];
		err = ncmpi_put_vars_all(ncid, i, start, edge, stride, buf, bufcount, buftype);
		IF (err != NC_EEDGE)
		    error("bad edge: status = %d", err);
                ELSE_NOK
		edge[j] = 1;
		stride[j] = 0;
                for (bufcount=1,k=0; k<var_rank[i]; k++) bufcount *= edge[k];
		err = ncmpi_put_vars_all(ncid, i, start, edge, stride, buf, bufcount, buftype);
		IF (err != NC_ESTRIDE)
		    error("bad stride: status = %d", err);
                ELSE_NOK
		stride[j] = 1;
	    }
        }
            /* Choose a random point dividing each dim into 2 parts */
            /* put 2^rank (nslabs) slabs so defined */
        nslabs = 1;
        for (j = 0; j < var_rank[i]; j++) {
            mid[j] = roll( var_shape[i][j] );
            nslabs *= 2;
        }
            /* bits of k determine whether to put lower or upper part of dim */
            /* choose random stride from 1 to edge */
        for (k = 0; k < nslabs; k++) {
            nstarts = 1;
            for (j = 0; j < var_rank[i]; j++) {
                if ((k >> j) & 1) {
                    start[j] = 0;
                    edge[j] = mid[j];
                }else{
                    start[j] = mid[j];
                    edge[j] = var_shape[i][j] - mid[j];
                }
                sstride[j] = stride[j] = edge[j] > 0 ? 1+roll(edge[j]) : 1;
                nstarts *= stride[j];
            }
            for (m = 0; m < nstarts; m++) {
                err = toMixedBase(m, var_rank[i], sstride, index);
                IF (err != NC_NOERR)
                    error("error in toMixedBase");
                nels = 1;
                for (j = 0; j < var_rank[i]; j++) {
                    count[j] = 1 + (edge[j] - index[j] - 1) / stride[j];
                    nels *= count[j];
                    index[j] += start[j];
                }
                        /* Random choice of forward or backward */
/* TODO
                if ( roll(2) ) {
                    for (j = 0; j < var_rank[i]; j++) {
                        index[j] += (count[j] - 1) * stride[j];
                        stride[j] = -stride[j];
                    }
                }
 */
		p = (char *) buf;
		for (j = 0; j < nels; j++) {
		    err = toMixedBase(j, var_rank[i], count, index2);
		    IF (err != NC_NOERR)
			error("error in toMixedBase");
		    for (d = 0; d < var_rank[i]; d++)
			index2[d] = index[d] + index2[d] * stride[d];
		    value = hash(var_type[i], var_rank[i], index2);
		    if (!inRange(value, var_type[i]))
			value = 0;
		    err = dbl2nc(value, var_type[i], p);
		    IF (err != NC_NOERR)
			error("error in dbl2nc");
		    p += nctypelen(var_type[i]);
		}
                for (bufcount=1,j=0; j<var_rank[i]; j++) bufcount *= count[j];
		if (var_rank[i] == 0 && i%2 == 0)
		    err = ncmpi_put_vars_all(ncid, i, NULL, NULL, NULL, buf, bufcount, buftype);
		else
		    err = ncmpi_put_vars_all(ncid, i, index, count, stride, buf, bufcount, buftype);
		IF (err != NC_NOERR)
		    error("%s", ncmpi_strerror(err));
                ELSE_NOK
            }
        }
    }

    check_vars(ncid);
    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
        error("ncmpi_close: %s", ncmpi_strerror(err));

    err = ncmpi_delete(scratch, info);
    IF (err != NC_NOERR)
        error("remove of %s failed", scratch);
    return nok;
}


/*
 * Test ncmpi_put_varm
 * Choose a random point dividing each dim into 2 parts
 * Put 2^rank (nslabs) slabs so defined
 * Choose random stride from 1 to edge
 * Buffer is bit image of whole external variable.
 * So all puts for a variable put different elements of buffer
 * At end check all variables using check_vars
 */
int
test_ncmpi_put_varm(void)
{
    int ncid, nok=0;
    int i;
    int j;
    int k;
    int m;
    int err;
    int nslabs;
    int nstarts;        /* number of different starts */
    MPI_Offset start[MAX_RANK];
    MPI_Offset edge[MAX_RANK];
    MPI_Offset index[MAX_RANK];
    MPI_Offset mid[MAX_RANK];
    MPI_Offset count[MAX_RANK];
    MPI_Offset sstride[MAX_RANK];
    MPI_Offset stride[MAX_RANK];
    MPI_Offset imap[MAX_RANK];
    MPI_Offset imap2[MAX_RANK];
    MPI_Offset bufcount;
    MPI_Datatype buftype;
    double buf[MAX_NELS];       /* (void *) buffer */
    char *p;			/* (void *) pointer */
    double value;

    err = ncmpi_create(comm, scratch, NC_NOCLOBBER|extra_flags, info, &ncid);
    IF (err != NC_NOERR) {
        error("ncmpi_create: %s", ncmpi_strerror(err));
        return nok;
    }
    def_dims(ncid);
    def_vars(ncid);
    err = ncmpi_enddef(ncid);
    IF (err != NC_NOERR)
        error("ncmpi_enddef: %s", ncmpi_strerror(err));

    for (i = 0; i < numVars; i++) {
        assert(var_rank[i] <= MAX_RANK);
        assert(var_nels[i] <= MAX_NELS);
        for (j = 0; j < var_rank[i]; j++) {
            start[j] = 0;
            edge[j] = 1;
            stride[j] = 1;
        }
	if (var_rank[i] > 0) {
	    j = var_rank[i] - 1; 
	    imap[j] = nctypelen(var_type[i]); /*  netCDF considers imap in bytes */
	    imap[j] = 1;                      /* PnetCDF considers imap in elements */
	    for (; j > 0; j--)
		imap[j-1] = imap[j] * var_shape[i][j];
	}
	p = (char *) buf;
	for (j = 0; j < var_nels[i]; j++) {
	    err = toMixedBase(j, var_rank[i], var_shape[i], index);
	    IF (err != NC_NOERR)
		error("error in toMixedBase");
	    value = hash(var_type[i], var_rank[i], index);
	    if (!inRange(value, var_type[i]))
		value = 0;
	    err = dbl2nc(value, var_type[i], p);
	    IF (err != NC_NOERR)
		error("error in dbl2nc");
	    p += nctypelen(var_type[i]);
	}
        buftype = nc_mpi_type(var_type[i]);
        for (bufcount=1,j=0; j<var_rank[i]; j++) bufcount *= edge[j];
        err = ncmpi_put_varm_all(BAD_ID, i, start, edge, stride, imap, buf, bufcount, buftype);
        IF (err != NC_EBADID)
            error("bad ncid: status = %d", err);
        ELSE_NOK
        err = ncmpi_put_varm_all(ncid, BAD_VARID, start, edge, stride, imap, buf, bufcount, buftype);
        IF (err != NC_ENOTVAR)
            error("bad var id: status = %d", err);
        ELSE_NOK
        for (j = 0; j < var_rank[i]; j++) {
            if (var_dimid[i][j] > 0) {          /* skip record dim */
		start[j] = var_shape[i][j];
                for (bufcount=1,k=0; k<var_rank[i]; k++) bufcount *= edge[k];
		err = ncmpi_put_varm_all(ncid, i, start, edge, stride, imap, buf, bufcount, buftype);
		IF (err != NC_EINVALCOORDS)
		    error("bad index: status = %d", err);
                ELSE_NOK
		start[j] = 0;
		edge[j] = var_shape[i][j] + 1;
                for (bufcount=1,k=0; k<var_rank[i]; k++) bufcount *= edge[k];
		err = ncmpi_put_varm_all(ncid, i, start, edge, stride, imap, buf, bufcount, buftype);
		IF (err != NC_EEDGE)
		    error("bad edge: status = %d", err);
                ELSE_NOK
		edge[j] = 1;
		stride[j] = 0;
                for (bufcount=1,k=0; k<var_rank[i]; k++) bufcount *= edge[k];
		err = ncmpi_put_varm_all(ncid, i, start, edge, stride, imap, buf, bufcount, buftype);
		IF (err != NC_ESTRIDE)
		    error("bad stride: status = %d", err);
                ELSE_NOK
		stride[j] = 1;
	    }
        }
            /* Choose a random point dividing each dim into 2 parts */
            /* put 2^rank (nslabs) slabs so defined */
        nslabs = 1;
        for (j = 0; j < var_rank[i]; j++) {
            mid[j] = roll( var_shape[i][j] );
            nslabs *= 2;
        }
            /* bits of k determine whether to put lower or upper part of dim */
            /* choose random stride from 1 to edge */
        for (k = 0; k < nslabs; k++) {
            nstarts = 1;
            for (j = 0; j < var_rank[i]; j++) {
                if ((k >> j) & 1) {
                    start[j] = 0;
                    edge[j] = mid[j];
                }else{
                    start[j] = mid[j];
                    edge[j] = var_shape[i][j] - mid[j];
                }
                sstride[j] = stride[j] = edge[j] > 0 ? 1+roll(edge[j]) : 1;
                imap2[j] = imap[j] * sstride[j];
                nstarts *= stride[j];
            }
            for (m = 0; m < nstarts; m++) {
		if (var_rank[i] == 0 && i%2 == 0) {
		    err = ncmpi_put_varm_all(ncid, i, NULL, NULL, NULL, NULL, buf, 1, buftype);
		} else {
		    err = toMixedBase(m, var_rank[i], sstride, index);
		    IF (err != NC_NOERR)
			error("error in toMixedBase");
		    for (j = 0; j < var_rank[i]; j++) {
			count[j] = 1 + (edge[j] - index[j] - 1) / stride[j];
			index[j] += start[j];
		    }
			    /* Random choice of forward or backward */
/* TODO
		    if ( roll(2) ) {
			for (j = 0; j < var_rank[i]; j++) {
			    index[j] += (count[j] - 1) * stride[j];
			    stride[j] = -stride[j];
			}
		    }
 */
		    j = fromMixedBase(var_rank[i], index, var_shape[i]);
                    p = (char *) buf + j * nctypelen(var_type[i]);
                    for (bufcount=1,j=0; j<var_rank[i]; j++) bufcount *= count[j];
		    err = ncmpi_put_varm_all(ncid, i, index, count, stride, imap2, p, bufcount, buftype);
		}
		IF (err != NC_NOERR)
		    error("i=%d var_rank[i]=%d %s", i,var_rank[i],ncmpi_strerror(err));
                ELSE_NOK
            }
        }
    }

    check_vars(ncid);
    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
        error("ncmpi_close: %s", ncmpi_strerror(err));

    err = ncmpi_delete(scratch, info);
    IF (err != NC_NOERR)
        error("remove of %s failed", scratch);
    return nok;
}


/*
 * Test ncmpi_rename_var
 *    try with bad netCDF handle, check error
 *    try with bad variable handle, check error
 *    try renaming to existing variable name, check error
 *    check that proper rename worked with ncmpi_inq_varid
 *    try in data mode, check error
 */
int
test_ncmpi_rename_var(void)
{
    int ncid;
    int varid;
    int err, nok=0;
    int i;
    char name[NC_MAX_NAME];

    err = ncmpi_create(comm, scratch, NC_NOCLOBBER|extra_flags, info, &ncid);
    IF (err != NC_NOERR) {
        error("ncmpi_create: %s", ncmpi_strerror(err));
        return nok;
    }
    err = ncmpi_rename_var(ncid, BAD_VARID, "newName");
    IF (err != NC_ENOTVAR)
	error("bad var id: status = %d", err);
    ELSE_NOK
    def_dims(ncid);
    def_vars(ncid);

	/* Prefix "new_" to each name */
    for (i = 0; i < numVars; i++) {
        err = ncmpi_rename_var(BAD_ID, i, "newName");
        IF (err != NC_EBADID)
            error("bad ncid: status = %d", err);
        ELSE_NOK
        err = ncmpi_rename_var(ncid, i, var_name[numVars-1]);
        IF (err != NC_ENAMEINUSE)
            error("duplicate name: status = %d", err);
        ELSE_NOK
	strcpy(name, "new_");
	strcat(name, var_name[i]);
        err = ncmpi_rename_var(ncid, i, name);
        IF (err != NC_NOERR)
	    error("ncmpi_rename_var: %s", ncmpi_strerror(err));
        ELSE_NOK
        err = ncmpi_inq_varid(ncid, name, &varid);
        IF (err != NC_NOERR)
	    error("ncmpi_inq_varid: %s", ncmpi_strerror(err));
        IF (varid != i)
	    error("Unexpected varid");
    }

	/* Change to data mode */
	/* Try making names even longer. Then restore original names */
    err = ncmpi_enddef(ncid);
    IF (err != NC_NOERR)
        error("ncmpi_enddef: %s", ncmpi_strerror(err));
    for (i = 0; i < numVars; i++) {
	strcpy(name, "even_longer_");
	strcat(name, var_name[i]);
        err = ncmpi_rename_var(ncid, i, name);
        IF (err != NC_ENOTINDEFINE)
            error("longer name in data mode: status = %d", err);
        ELSE_NOK
        err = ncmpi_rename_var(ncid, i, var_name[i]);
        IF (err != NC_NOERR)
	    error("ncmpi_rename_var: %s", ncmpi_strerror(err));
        ELSE_NOK
        err = ncmpi_inq_varid(ncid, var_name[i], &varid);
        IF (err != NC_NOERR)
	    error("ncmpi_inq_varid: %s", ncmpi_strerror(err));
        IF (varid != i)
	    error("Unexpected varid");
    }

    put_vars(ncid);
    check_vars(ncid);

    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
        error("ncmpi_close: %s", ncmpi_strerror(err));

    err = ncmpi_delete(scratch, info);
    IF (err != NC_NOERR)
        error("remove of %s failed", scratch);
    return nok;
}


int
test_ncmpi_put_att(void)
{
    int ncid, nok=0;
    int varid;
    int i;
    int j;
    MPI_Offset k;
    int err;
    double buf[MAX_NELS];       /* (void *) buffer */
    char *p;                    /* (void *) pointer */
    char *name;			/* of att */
    nc_type datatype;		/* of att */
    MPI_Offset length;		/* of att */
    double value;

    err = ncmpi_create(comm, scratch, NC_NOCLOBBER|extra_flags, info, &ncid);
    IF (err != NC_NOERR) {
        error("ncmpi_create: %s", ncmpi_strerror(err));
        return nok;
    }
    def_dims(ncid);
    def_vars(ncid);

    for (i = -1; i < numVars; i++) {
	varid = VARID(i);
        for (j = 0; j < NATTS(i); j++) {
	    name = ATT_NAME(i,j);
	    datatype = ATT_TYPE(i,j);
	    length = ATT_LEN(i,j);
            err = ncmpi_put_att(BAD_ID, varid, name, datatype, length, buf);
            IF (err != NC_EBADID)
                error("bad ncid: status = %d", err);
            ELSE_NOK
            err = ncmpi_put_att(ncid, varid, BAD_NAME, datatype, length, buf);
	    IF (err != NC_EBADNAME)
		error("bad name: status = %d", err);
            ELSE_NOK
            err = ncmpi_put_att(ncid, BAD_VARID, name, datatype, length, buf);
            IF (err != NC_ENOTVAR)
                error("bad var id: status = %d", err);
            ELSE_NOK
	    err = ncmpi_put_att(ncid, varid, name, BAD_TYPE, length, buf);
	    IF (err != NC_EBADTYPE)
		error("bad type: status = %d", err);
            ELSE_NOK
	    p = (char *) buf;
	    for (k=0; k<length; k++) {
		value = hash(datatype, -1, &k);
		if (!inRange(value, datatype))
		    value = 0;
		err = dbl2nc(value, datatype, p);
		IF (err != NC_NOERR)
		    error("error in dbl2nc");
		p += nctypelen(datatype);
	    }
            err = ncmpi_put_att(ncid, varid, name, datatype, length, buf);
            IF (err != NC_NOERR)
                error("%s", ncmpi_strerror(err));
            ELSE_NOK
        }
    }

    check_atts(ncid);
    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
        error("ncmpi_close: %s", ncmpi_strerror(err));

    err = ncmpi_delete(scratch, info);
    IF (err != NC_NOERR)
        error("remove of %s failed", scratch);
    return nok;
}


/*
 * Test ncmpi_copy_att
 *    try with bad source or target netCDF handles, check error
 *    try with bad source or target variable handle, check error
 *    try with nonexisting attribute, check error
 *    check that NC_GLOBAL variable for source or target works
 *    check that new attribute put works with target in define mode
 *    check that old attribute put works with target in data mode
 *    check that changing type and length of an attribute work OK
 *    try with same ncid for source and target, different variables
 *    try with same ncid for source and target, same variable
 */
int
test_ncmpi_copy_att(void)
{
    int ncid_in;
    int ncid_out;
    int varid;
    int err, nok=0;
    int i;
    int j=0;
    char *name;                 /* of att */
    nc_type datatype;           /* of att */
    MPI_Offset length;          /* of att */
    char  value;

    err = ncmpi_open(comm, testfile, NC_NOWRITE, info, &ncid_in);
    IF (err != NC_NOERR)
        error("ncmpi_open: %s", ncmpi_strerror(err));
    err = ncmpi_create(comm, scratch, NC_NOCLOBBER|extra_flags, info, &ncid_out);
    IF (err != NC_NOERR) {
        error("ncmpi_create: %s", ncmpi_strerror(err));
        return nok;
    }
    def_dims(ncid_out);
    def_vars(ncid_out);

    for (i = -1; i < numVars; i++) {
        varid = VARID(i);
        for (j = 0; j < NATTS(i); j++) {
            name = ATT_NAME(i,j);
	    err = ncmpi_copy_att(ncid_in, BAD_VARID, name, ncid_out, varid);
	    IF (err != NC_ENOTVAR)
		error("bad var id: status = %d", err);
            ELSE_NOK
	    err = ncmpi_copy_att(ncid_in, varid, name, ncid_out, BAD_VARID);
	    IF (err != NC_ENOTVAR)
		error("bad var id: status = %d", err);
            ELSE_NOK
	    err = ncmpi_copy_att(BAD_ID, varid, name, ncid_out, varid);
	    IF (err != NC_EBADID)
		error("bad ncid: status = %d", err);
            ELSE_NOK
	    err = ncmpi_copy_att(ncid_in, varid, name, BAD_ID, varid);
	    IF (err != NC_EBADID)
		error("bad ncid: status = %d", err);
            ELSE_NOK
	    err = ncmpi_copy_att(ncid_in, varid, "noSuch", ncid_out, varid);
	    IF (err != NC_ENOTATT)
		error("bad attname: status = %d", err);
            ELSE_NOK
	    err = ncmpi_copy_att(ncid_in, varid, name, ncid_out, varid);
	    IF (err != NC_NOERR)
		error("ncmpi_copy_att: %s", ncmpi_strerror(err));
            ELSE_NOK
	    err = ncmpi_copy_att(ncid_out, varid, name, ncid_out, varid);
	    IF (err != NC_NOERR)
		error("source = target: %s", ncmpi_strerror(err));
            ELSE_NOK
	}
    }

    err = ncmpi_close(ncid_in);
    IF (err != NC_NOERR)
        error("ncmpi_close: %s", ncmpi_strerror(err));

        /* Close scratch. Reopen & check attributes */
    err = ncmpi_close(ncid_out);
    IF (err != NC_NOERR)
        error("ncmpi_close: %s", ncmpi_strerror(err));
    err = ncmpi_open(comm, scratch, NC_WRITE, info, &ncid_out);
    IF (err != NC_NOERR)
        error("ncmpi_open: %s", ncmpi_strerror(err));
    check_atts(ncid_out);

       /* 
	* change to define mode
	* define single char. global att. ':a' with value 'A'
	* This will be used as source for following copies
	*/
    err = ncmpi_redef(ncid_out);
    IF (err != NC_NOERR)
        error("ncmpi_redef: %s", ncmpi_strerror(err));
    err = ncmpi_put_att_text(ncid_out, NC_GLOBAL, "a", 1, "A");
    IF (err != NC_NOERR)
	error("ncmpi_put_att_text: %s", ncmpi_strerror(err));

       /* 
	* change to data mode
	* Use scratch as both source & dest.
	* try copy to existing att. change type & decrease length
	* rename 1st existing att of each var (if any) 'a'
	* if this att. exists them copy ':a' to it
	*/
    err = ncmpi_enddef(ncid_out);
    IF (err != NC_NOERR)
        error("ncmpi_enddef: %s", ncmpi_strerror(err));
    for (i = 0; i < numVars; i++) {
	if (NATTS(i) > 0 && ATT_LEN(i,j) > 0) {
	    err = ncmpi_rename_att(ncid_out, i, att_name[i][0], "a");
	    IF (err != NC_NOERR)
		error("ncmpi_rename_att: %s", ncmpi_strerror(err));
	    err = ncmpi_copy_att(ncid_out, NC_GLOBAL, "a", ncid_out, i);
	    IF (err != NC_NOERR)
		error("ncmpi_copy_att: %s", ncmpi_strerror(err));
            ELSE_NOK
	}
    }
    err = ncmpi_close(ncid_out);
    IF (err != NC_NOERR)
        error("ncmpi_close: %s", ncmpi_strerror(err));

	/* Reopen & check */
    err = ncmpi_open(comm, scratch, NC_WRITE, info, &ncid_out);
    IF (err != NC_NOERR)
        error("ncmpi_open: %s", ncmpi_strerror(err));
    for (i = 0; i < numVars; i++) {
	if (NATTS(i) > 0 && ATT_LEN(i,j) > 0) {
	    err = ncmpi_inq_att(ncid_out, i, "a", &datatype, &length);
	    IF (err != NC_NOERR)
		error("ncmpi_inq_att: %s", ncmpi_strerror(err));
	    IF (datatype != NC_CHAR)
		error("Unexpected type");
	    IF (length != 1)
		error("Unexpected length");
	    err = ncmpi_get_att_text(ncid_out, i, "a", &value);
	    IF (err != NC_NOERR)
		error("ncmpi_get_att_text: %s", ncmpi_strerror(err));
	    IF (value != 'A')
		error("Unexpected value");
	}                                                   
    }                                                   

    err = ncmpi_close(ncid_out);
    IF (err != NC_NOERR)
        error("ncmpi_close: %s", ncmpi_strerror(err));
    err = ncmpi_delete(scratch, info);
    IF (err != NC_NOERR)
        error("remove of %s failed", scratch);
    return nok;
}


/*
 * Test ncmpi_rename_att
 *    try with bad netCDF handle, check error
 *    try with bad variable handle, check error
 *    try with nonexisting att name, check error
 *    try renaming to existing att name, check error
 *    check that proper rename worked with ncmpi_inq_attid
 *    try in data mode, check error
 */
int
test_ncmpi_rename_att(void)
{
    int ncid;
    int varid;
    int err;
    int i;
    int j;
    MPI_Offset  k;
    int attnum;
    char *attname;
    char name[NC_MAX_NAME];
    char oldname[NC_MAX_NAME];
    char newname[NC_MAX_NAME];
    int nok = 0;      /* count of valid comparisons */
    nc_type datatype;
    nc_type atttype;
    MPI_Offset length;
    size_t attlength;
    char  text[MAX_NELS];
    double value[MAX_NELS];
    double expect;

    err = ncmpi_create(comm, scratch, NC_NOCLOBBER|extra_flags, info, &ncid);
    IF (err != NC_NOERR) {
        error("ncmpi_create: %s", ncmpi_strerror(err));
        return nok;
    }
    err = ncmpi_rename_att(ncid, BAD_VARID, "abc", "newName");
    IF (err != NC_ENOTVAR)
	error("bad var id: status = %d", err);
    ELSE_NOK
    def_dims(ncid);
    def_vars(ncid);
    put_atts(ncid);

    for (i = -1; i < numVars; i++) {
        varid = VARID(i);
        for (j = 0; j < NATTS(i); j++) {
	    attname = ATT_NAME(i,j);
	    err = ncmpi_rename_att(BAD_ID, varid, attname, "newName");
	    IF (err != NC_EBADID)
		error("bad ncid: status = %d", err);
            ELSE_NOK
	    err = ncmpi_rename_att(ncid, varid, "noSuch", "newName");
	    IF (err != NC_ENOTATT)
		error("bad attname: status = %d", err);
            ELSE_NOK
	    strcpy(newname, "new_");
	    strcat(newname, attname);
	    err = ncmpi_rename_att(ncid, varid, attname, newname);
	    IF (err != NC_NOERR)
		error("ncmpi_rename_att: %s", ncmpi_strerror(err));
            ELSE_NOK
	    err = ncmpi_inq_attid(ncid, varid, newname, &attnum);
	    IF (err != NC_NOERR)
		error("ncmpi_inq_attid: %s", ncmpi_strerror(err));
	    IF (attnum != j)
		error("Unexpected attnum");
	}
    }

    /* Close. Reopen & check */
    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
        error("ncmpi_close: %s", ncmpi_strerror(err));
    err = ncmpi_open(comm, scratch, NC_WRITE, info, &ncid);
    IF (err != NC_NOERR)
        error("ncmpi_open: %s", ncmpi_strerror(err));

    for (i = -1; i < numVars; i++) {
        varid = VARID(i);
        for (j = 0; j < NATTS(i); j++) {
	    attname = ATT_NAME(i,j);
	    atttype = ATT_TYPE(i,j);
	    attlength = ATT_LEN(i,j);
            strcpy(newname, "new_");
            strcat(newname, attname);
            err = ncmpi_inq_attname(ncid, varid, j, name);
            IF (err != NC_NOERR)
                error("ncmpi_inq_attname: %s", ncmpi_strerror(err));
            IF (strcmp(name, newname) != 0)
                error("ncmpi_inq_attname: unexpected name");
            err = ncmpi_inq_att(ncid, varid, name, &datatype, &length);
            IF (err != NC_NOERR)
                error("ncmpi_inq_att: %s", ncmpi_strerror(err));
            IF (datatype != atttype)
                error("ncmpi_inq_att: unexpected type");
            IF (length != attlength)
                error("ncmpi_inq_att: unexpected length");
            if (datatype == NC_CHAR) {
                err = ncmpi_get_att_text(ncid, varid, name, text);
                IF (err != NC_NOERR)
                    error("ncmpi_get_att_text: %s", ncmpi_strerror(err));
                for (k = 0; k < attlength; k++) {
                    expect = hash(datatype, -1, &k);
                    IF (text[k] != (char)expect)
                        error("ncmpi_get_att_text: unexpected value");
                }
            } else {
                err = ncmpi_get_att_double(ncid, varid, name, value);
                IF (err != NC_NOERR)
                    error("ncmpi_get_att_double: %s", ncmpi_strerror(err));
                for (k = 0; k < attlength; k++) {
                    expect = hash(datatype, -1, &k);
		    if (inRange(expect, datatype)) {
			IF (!equal(value[k],expect,datatype,NCT_DOUBLE))
			    error("ncmpi_get_att_double: unexpected value");
                    }
                }
            }
        }
    }

	/* Now in data mode */
	/* Try making names even longer. Then restore original names */

    for (i = -1; i < numVars; i++) {
        varid = VARID(i);
        for (j = 0; j < NATTS(i); j++) {
	    attname = ATT_NAME(i,j);
	    strcpy(oldname, "new_");
	    strcat(oldname, attname);
	    strcpy(newname, "even_longer_");
	    strcat(newname, attname);
	    err = ncmpi_rename_att(ncid, varid, oldname, newname);
	    IF (err != NC_ENOTINDEFINE)
		error("longer name in data mode: status = %d", err);
            ELSE_NOK
	    err = ncmpi_rename_att(ncid, varid, oldname, attname);
	    IF (err != NC_NOERR)
		error("ncmpi_rename_att: %s", ncmpi_strerror(err));
            ELSE_NOK
	    err = ncmpi_inq_attid(ncid, varid, attname, &attnum);
	    IF (err != NC_NOERR)
		error("ncmpi_inq_attid: %s", ncmpi_strerror(err));
	    IF (attnum != j)
		error("Unexpected attnum");
	}
    }

    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
        error("ncmpi_close: %s", ncmpi_strerror(err));

    err = ncmpi_delete(scratch, info);
    IF (err != NC_NOERR)
        error("remove of %s failed", scratch);
    return nok;
}


/*
 * Test ncmpi_del_att
 *    try with bad netCDF handle, check error
 *    try with bad variable handle, check error
 *    try with nonexisting att name, check error
 *    check that proper delete worked using:
 *      ncmpi_inq_attid, ncmpi_inq_natts, ncmpi_inq_varnatts
 */
int
test_ncmpi_del_att(void)
{
    int ncid;
    int err, nok=0;
    int i;
    int j;
    int attnum;
    int natts;
    int numatts;
    int varid;
    char *name;                 /* of att */

    err = ncmpi_create(comm, scratch, NC_NOCLOBBER|extra_flags, info, &ncid);
    IF (err != NC_NOERR) {
        error("ncmpi_create: %s", ncmpi_strerror(err));
        return nok;
    }
    err = ncmpi_del_att(ncid, BAD_VARID, "abc");
    IF (err != NC_ENOTVAR)
	error("bad var id: status = %d", err);
    ELSE_NOK
    def_dims(ncid);
    def_vars(ncid);
    put_atts(ncid);

    for (i = -1; i < numVars; i++) {
	varid = VARID(i);
	numatts = NATTS(i);
        for (j = 0; j < numatts; j++) {
	    name = ATT_NAME(i,j);
	    err = ncmpi_del_att(BAD_ID, varid, name);
	    IF (err != NC_EBADID)
		error("bad ncid: status = %d", err);
            ELSE_NOK
	    err = ncmpi_del_att(ncid, varid, "noSuch");
	    IF (err != NC_ENOTATT)
		error("bad attname: status = %d", err);
            ELSE_NOK
	    err = ncmpi_del_att(ncid, varid, name);
	    IF (err != NC_NOERR)
		error("ncmpi_del_att: %s", ncmpi_strerror(err));
            ELSE_NOK
	    err = ncmpi_inq_attid(ncid, varid, name, &attnum);
	    IF (err != NC_ENOTATT)
		error("bad attname: status = %d", err);
	    if (i < 0) {
		err = ncmpi_inq_natts(ncid, &natts);
		IF (err != NC_NOERR)
		    error("ncmpi_inq_natts: %s", ncmpi_strerror(err));
		IF (natts != numatts-j-1)
		    error("natts: expected %d, got %d", numatts-j-1, natts);
	    }
	    err = ncmpi_inq_varnatts(ncid, varid, &natts);
	    IF (err != NC_NOERR)
		error("ncmpi_inq_natts: %s", ncmpi_strerror(err));
	    IF (natts != numatts-j-1)
		error("natts: expected %d, got %d", numatts-j-1, natts);
	}
    }

        /* Close. Reopen & check no attributes left */
    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
        error("ncmpi_close: %s", ncmpi_strerror(err));
    err = ncmpi_open(comm, scratch, NC_WRITE, info, &ncid);
    IF (err != NC_NOERR)
        error("ncmpi_open: %s", ncmpi_strerror(err));
    err = ncmpi_inq_natts(ncid, &natts);
    IF (err != NC_NOERR)
	error("ncmpi_inq_natts: %s", ncmpi_strerror(err));
    IF (natts != 0)
	error("natts: expected %d, got %d", 0, natts);
    for (i = -1; i < numVars; i++) {
	varid = VARID(i);
	err = ncmpi_inq_varnatts(ncid, varid, &natts);
	IF (err != NC_NOERR)
	    error("ncmpi_inq_natts: %s", ncmpi_strerror(err));
	IF (natts != 0)
	    error("natts: expected %d, got %d", 0, natts);
    }

	/* restore attributes. change to data mode. try to delete */
    err = ncmpi_redef(ncid);
    IF (err != NC_NOERR)
        error("ncmpi_redef: %s", ncmpi_strerror(err));
    put_atts(ncid);
    err = ncmpi_enddef(ncid);
    IF (err != NC_NOERR)
        error("ncmpi_enddef: %s", ncmpi_strerror(err));

    for (i = -1; i < numVars; i++) {
	varid = VARID(i);
	numatts = NATTS(i);
        for (j = 0; j < numatts; j++) {
	    name = ATT_NAME(i,j);
	    err = ncmpi_del_att(ncid, varid, name);
	    IF (err != NC_ENOTINDEFINE)
		error("in data mode: status = %d", err);
            ELSE_NOK
	}
    }

    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
        error("ncmpi_close: %s", ncmpi_strerror(err));
    err = ncmpi_delete(scratch, info);
    IF (err != NC_NOERR)
        error("remove of %s failed", scratch);
    return nok;
}


/*
 * Test ncmpi_set_fill
 *    try with bad netCDF handle, check error
 *    try in read-only mode, check error
 *    try with bad new_fillmode, check error
 *    try in data mode, check error
 *    check that proper set to NC_FILL works for record & non-record variables
 *    (note that it is not possible to test NC_NOFILL mode!)
 *    close file & create again for test using attribute _FillValue
 */
int
test_ncmpi_set_fill(void)
{
    int ncid;
    int varid;
    int err;
    int i;
    int j;
    int old_fillmode;
    int nok = 0;      /* count of valid comparisons */
    char text = 0;
    double value = 0;
    double fill;
    MPI_Offset index[MAX_RANK];

	/* bad ncid */
    err = ncmpi_set_fill(BAD_ID, NC_NOFILL, &old_fillmode);
    IF (err != NC_EBADID)
	error("bad ncid: status = %d", err);

	/* try in read-only mode */
    err = ncmpi_open(comm, testfile, NC_NOWRITE, info, &ncid);
    IF (err != NC_NOERR)
        error("ncmpi_open: %s", ncmpi_strerror(err));
    err = ncmpi_set_fill(ncid, NC_NOFILL, &old_fillmode);
    IF (err != NC_EPERM)
	error("read-only: status = %d", err);
    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
        error("ncmpi_close: %s", ncmpi_strerror(err));

	/* create scratch */
    err = ncmpi_create(comm, scratch, NC_NOCLOBBER|extra_flags, info, &ncid);
    IF (err != NC_NOERR) {
        error("ncmpi_create: %s", ncmpi_strerror(err));
        return nok;
    }

	/* BAD_FILLMODE */
    err = ncmpi_set_fill(ncid, BAD_FILLMODE, &old_fillmode);
    IF (err != NC_EINVAL)
        error("bad fillmode: status = %d", err);

	/* proper calls */
    err = ncmpi_set_fill(ncid, NC_NOFILL, &old_fillmode);
    IF (err != NC_NOERR)
        error("ncmpi_set_fill: %s", ncmpi_strerror(err));
    IF (old_fillmode != NC_NOFILL)
        error("Unexpected old fill mode: %d", old_fillmode);
    err = ncmpi_set_fill(ncid, NC_FILL, &old_fillmode);
    IF (err != NC_NOERR)
        error("ncmpi_set_fill: %s", ncmpi_strerror(err));
    IF (old_fillmode != NC_NOFILL)
        error("Unexpected old fill mode: %d", old_fillmode);

	/* define dims & vars */
    def_dims(ncid);
    def_vars(ncid);

	/* Change to data mode. Set fillmode again */
    err = ncmpi_enddef(ncid);
    IF (err != NC_NOERR)
        error("ncmpi_enddef: %s", ncmpi_strerror(err));
    err = ncmpi_set_fill(ncid, NC_FILL, &old_fillmode);
    IF (err != NC_ENOTINDEFINE)
        error("ncmpi_set_fill: %s", ncmpi_strerror(err));

	/* Write record number NRECS to force writing of preceding records */
	/* Assumes variable cr is char vector with UNLIMITED dimension */
    err = ncmpi_inq_varid(ncid, "cr", &varid);
    IF (err != NC_NOERR)
        error("ncmpi_inq_varid: %s", ncmpi_strerror(err));
    index[0] = NRECS;
    for (i=0; i<=index[0]; i++)
        err = ncmpi_fill_var_rec(ncid, varid, i);
    err = ncmpi_put_var1_text_all(ncid, varid, index, &text);
    IF (err != NC_NOERR)
        error("ncmpi_put_var1_text_all: %s", ncmpi_strerror(err));

	/* get all variables & check all values equal default fill */
    for (i = 0; i < numVars; i++) {
        if (var_dimid[i][0] == RECDIM) continue; /* skip record variables */
	switch (var_type[i]) {
	    case NC_CHAR:   fill = NC_FILL_CHAR; break;
	    case NC_BYTE:   fill = NC_FILL_BYTE; break;
	    case NC_SHORT:  fill = NC_FILL_SHORT; break;
	    case NC_INT:   fill = NC_FILL_INT; break;
	    case NC_FLOAT:  fill = NC_FILL_FLOAT; break;
	    case NC_DOUBLE: fill = NC_FILL_DOUBLE; break;
	    case NC_UBYTE: fill = NC_FILL_UBYTE; break;
	    case NC_USHORT: fill = NC_FILL_USHORT; break;
	    case NC_UINT: fill = NC_FILL_UINT; break;
	    case NC_INT64: fill = NC_FILL_INT64; break;
	    case NC_UINT64: fill = NC_FILL_UINT64; break;
	    default: assert(0);
	}
	for (j = 0; j < var_nels[i]; j++) {
            err = toMixedBase(j, var_rank[i], var_shape[i], index);
            IF (err != NC_NOERR)
                error("error in toMixedBase");
	    if (var_type[i] == NC_CHAR) {
		err = ncmpi_get_var1_text_all(ncid, i, index, &text);
		IF (err != NC_NOERR)
		    error("ncmpi_get_var1_text_all failed: %s", ncmpi_strerror(err));
		value = text;
	    } else {
		err = ncmpi_get_var1_double_all(ncid, i, index, &value);
		IF (err != NC_NOERR)
		    error("ncmpi_get_var1_double_all failed: %s", ncmpi_strerror(err));
	    }
	    IF (value != fill && fabs((fill - value)/fill) > DBL_EPSILON)
		error("\n\t\t%s Value expected: %-23.17e,\n\t\t          read: %-23.17e\n",
			var_name[i],fill, value);
	    ELSE_NOK
        }
    }

	/* close scratch & create again for test using attribute _FillValue */
    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
        error("ncmpi_close: %s", ncmpi_strerror(err));
    err = ncmpi_create(comm, scratch, NC_CLOBBER|extra_flags, info, &ncid);
    IF (err != NC_NOERR) {
        error("ncmpi_create: %s", ncmpi_strerror(err));
        return nok;
    }
    def_dims(ncid);
    def_vars(ncid);

	/* set _FillValue = 42 for all vars */
    text = fill = 42;
    for (i = 0; i < numVars; i++) {
	if (var_type[i] == NC_CHAR) {
	    err = ncmpi_put_att_text(ncid, i, "_FillValue", 1, &text);
	    IF (err != NC_NOERR)
		error("ncmpi_put_att_text: %s", ncmpi_strerror(err));
	} else {
	    err = ncmpi_put_att_double(ncid, i, "_FillValue",var_type[i],1,&fill);
	    IF (err != NC_NOERR)
		error("ncmpi_put_att_double: %s", ncmpi_strerror(err));
	}
    }

	/* data mode. write records */
    err = ncmpi_enddef(ncid);
    IF (err != NC_NOERR)
        error("ncmpi_enddef: %s", ncmpi_strerror(err));
    index[0] = NRECS;
    for (i=0; i<=index[0]; i++)
        err = ncmpi_fill_var_rec(ncid, varid, i);
    err = ncmpi_put_var1_text_all(ncid, varid, index, &text);
    IF (err != NC_NOERR)
        error("ncmpi_put_var1_text_all: %s", ncmpi_strerror(err));

	/* get all variables & check all values equal 42 */
    for (i = 0; i < numVars; i++) {
        if (var_dimid[i][0] == RECDIM) continue; /* skip record variables */
	for (j = 0; j < var_nels[i]; j++) {
            err = toMixedBase(j, var_rank[i], var_shape[i], index);
            IF (err != NC_NOERR)
                error("error in toMixedBase");
	    if (var_type[i] == NC_CHAR) {
		err = ncmpi_get_var1_text_all(ncid, i, index, &text);
		IF (err != NC_NOERR)
		    error("ncmpi_get_var1_text_all failed: %s", ncmpi_strerror(err));
		value = text;
	    } else {
		err = ncmpi_get_var1_double_all(ncid, i, index, &value);
		IF (err != NC_NOERR)
		    error("ncmpi_get_var1_double_all failed: %s", ncmpi_strerror(err));
	    }
	    IF (value != fill)
		error(" %s Value expected: %g, read: %g\n", var_name[i],fill, value);
	    ELSE_NOK
        }
    }
    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
        error("ncmpi_close: %s", ncmpi_strerror(err));
    err = ncmpi_delete(scratch, info);
    IF (err != NC_NOERR)
        error("remove of %s failed", scratch);

    return nok;
}


/* This function gets the version of a netCDF file, 1 is for netCDF
   classic, 2 for 64-bit offset format, (someday) 3 for HDF5 format.
*/
#define MAGIC_NUM_LEN 4
static
int ncmpi_get_file_version(char *path, int *version)
{
   FILE *fp;
   char magic[MAGIC_NUM_LEN];

   /* Need two valid pointers - check for NULL. */
   if (!version || !path)
      return NC_EINVAL;

   /* Figure out if this is a netcdf or hdf5 file. */
   if (!(fp = fopen(path, "r")) ||
       fread(magic, MAGIC_NUM_LEN, 1, fp) != 1)
      return errno;
   fclose(fp);
   if (strncmp(magic, "CDF", MAGIC_NUM_LEN-1)==0)
   {
      if (magic[MAGIC_NUM_LEN-1] == NC_FORMAT_CLASSIC || 
         magic[MAGIC_NUM_LEN-1] == NC_FORMAT_CDF2 ||
         magic[MAGIC_NUM_LEN-1] == NC_FORMAT_CDF2)
        *version = magic[MAGIC_NUM_LEN-1];
      else
        return NC_ENOTNC;
   }
   /*   tomorrow, tomorrow, I love you tomorrow, you're always a day
       away! */
   /*if (magic[1] == 'H' && magic[2] == 'D' && magic[3] == 'F')
      *version = 3;*/
   return NC_NOERR;
}

/*
 * Test nc_set_default_format
 *    try with bad default format
 *    try with NULL old_formatp
 *    try in data mode, check error
 *    check that proper set to NC_FILL works for record & non-record variables
 *    (note that it is not possible to test NC_NOFILL mode!)
 *    close file & create again for test using attribute _FillValue
 */
int
test_ncmpi_set_default_format(void)
{
    int ncid, nok=0;
    int err;
    int i;
    int version=1;
    int old_format;

    /* bad format */
    err = ncmpi_set_default_format(BAD_DEFAULT_FORMAT, &old_format);
    IF (err != NC_EINVAL)
       error("bad default format: status = %d", err);
    ELSE_NOK

    /* NULL old_formatp */
    err = ncmpi_set_default_format(NC_FORMAT_CDF2, NULL);
    IF (err != NC_NOERR)
       error("null old_fortmatp: status = %d", err);
    ELSE_NOK

    /* Cycle through available formats. */
    for(i=1; i<5; i++)
    {
       if (i == 3 || i == 4) continue; /* test classic formats only */

       if ((err = ncmpi_set_default_format(i, NULL)))
         error("setting classic format: status = %d", err);
       ELSE_NOK
       if ((err=ncmpi_create(comm, scratch, NC_CLOBBER, info, &ncid)))
         error("bad nc_create: status = %d", err);
       if ((err=ncmpi_put_att_text(ncid, NC_GLOBAL, "testatt", 
                               sizeof("blah"), "blah")))
         error("bad put_att: status = %d", err);
       if ( (err=ncmpi_close(ncid)))
         error("bad close: status = %d", err);
       if ( (err = ncmpi_get_file_version(scratch, &version)) )
         error("bad file version = %d", err);
       if (version != i) {
          if (i == 4) {
              if (version == 3) continue;
              printf("expect version 3 but got %d (file=%s)",version,scratch);
              continue;
          }
          printf("expect version %d but got %d (file=%s)",i,version,scratch);
          error("bad file version = %d", version);
        }
    }

    /* Remove the left-over file. */
    if ((err = ncmpi_delete(scratch, info)))
       error("remove of %s failed", scratch);
    return nok;
}




/*
 * Test ncmpi_delete
 * 	create netcdf file 'scratch.nc' with no data, close it
 * 	delete the file
 */
int
test_ncmpi_delete(void)
{
    int err, nok=0;;
    int ncid;

    err = ncmpi_create(comm, scratch, NC_CLOBBER|extra_flags, info, &ncid);
    IF (err != NC_NOERR)
	error("error creating scratch file %s, status = %d\n", scratch,err);
    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
        error("ncmpi_close: %s", ncmpi_strerror(err));
    err = ncmpi_delete(scratch, info);
    IF (err != NC_NOERR)
	error("remove of %s failed", scratch);
    ELSE_NOK
    return nok;
}

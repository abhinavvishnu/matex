/*
 *  Copyright (C) 2003, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *  $Id: test_read.c 2297 2016-01-06 22:22:33Z wkliao $
 */

#include <sys/types.h> /* open() */
#include <sys/stat.h> /* open() */
#include <fcntl.h> /* open() */
#include <unistd.h> /* unlink(), write() */
#include "tests.h"

/* 
 * Test ncmpi_strerror.
 *    Try on a bad error status.
 *    Test for each defined error status.
 */
int
test_ncmpi_strerror(void)
{
    int i;
    const char *message, *expected_msg;
    int nok=0;

    static const struct {
	int status;
	const char *msg;
    } ncerrs[] = {
        {NC_NOERR, "No error"},
        {NC_EBADID, "NetCDF: Not a valid ID"},
        {NC_ENFILE, "NetCDF: Too many files open"},
        {NC_EEXIST, "NetCDF: File exists && NC_NOCLOBBER"},
        {NC_EINVAL, "NetCDF: Invalid argument"},
        {NC_EPERM, "NetCDF: Write to read only"},
        {NC_ENOTINDEFINE, "NetCDF: Operation not allowed in data mode"},
        {NC_EINDEFINE, "NetCDF: Operation not allowed in define mode"},
        {NC_EINVALCOORDS, "NetCDF: Index exceeds dimension bound"},
        {NC_EMAXDIMS, "NetCDF: NC_MAX_DIMS exceeded"},
        {NC_ENAMEINUSE, "NetCDF: String match to name in use"},
        {NC_ENOTATT, "NetCDF: Attribute not found"},
        {NC_EMAXATTS, "NetCDF: NC_MAX_ATTRS exceeded"},
        {NC_EBADTYPE, "NetCDF: Not a valid data type or _FillValue type mismatch"},
        {NC_EBADDIM, "NetCDF: Invalid dimension ID or name"},
        {NC_EUNLIMPOS, "NetCDF: NC_UNLIMITED in the wrong index"},
        {NC_EMAXVARS, "NetCDF: NC_MAX_VARS exceeded"},
        {NC_ENOTVAR, "NetCDF: Variable not found"},
        {NC_EGLOBAL, "NetCDF: Action prohibited on NC_GLOBAL varid"},
        {NC_ENOTNC, "NetCDF: Unknown file format"},
        {NC_ESTS, "NetCDF: In Fortran, string too short"},
        {NC_EMAXNAME, "NetCDF: NC_MAX_NAME exceeded"},
        {NC_EUNLIMIT, "NetCDF: NC_UNLIMITED size already in use"},
        {NC_ENORECVARS, "NetCDF: nc_rec op when there are no record vars"},
        {NC_ECHAR, "NetCDF: Attempt to convert between text & numbers"},
        {NC_EEDGE, "NetCDF: Start+count exceeds dimension bound"},
        {NC_ESTRIDE, "NetCDF: Illegal stride"},
        {NC_EBADNAME, "NetCDF: Name contains illegal characters"},
        {NC_ERANGE, "NetCDF: Numeric conversion not representable"},
        {NC_ENOMEM, "NetCDF: Memory allocation (malloc) failure"},
        {NC_EVARSIZE, "NetCDF: One or more variable sizes violate format constraints"},
        {NC_EDIMSIZE, "NetCDF: Invalid dimension size"}
    };

    /* Try on a bad error status */
    message = ncmpi_strerror(-666);/* should fail */
    expected_msg = "Unknown Error";
    IF (strncmp(message, expected_msg, strlen(expected_msg)) != 0)
	error("ncmpi_strerror on bad error status returned: %s", message);
    ELSE_NOK

    /* Try on each legitimate error status */
    for (i=0; i<LEN_OF(ncerrs); i++) {
	const char *message = ncmpi_strerror(ncerrs[i].status);
	IF (strcmp(message, ncerrs[i].msg) != 0)
	    error("ncmpi_strerror(%d) should return `%s', not `%s'",
		  ncerrs[i].status, ncerrs[i].msg, message);
        ELSE_NOK
    }
    return nok;
}


/* 
 * Test ncmpi_open.
 * If in read-only section of tests,
 *    Try to open a non-existent netCDF file, check error return.
 *    Open a file that is not a netCDF file, check error return.
 *    Open a netCDF file with a bad mode argument, check error return.
 *    Open a netCDF file with NC_NOWRITE, info mode, try to write, check error.
 *    Try to open a netcdf twice, check whether returned netcdf ids different.
 * If in writable section of tests,
 *    Open a netCDF file with NC_WRITE mode, write something, close it.
 * On exit, any open netCDF files are closed.
 */
#define NOT_NC_FILE "dummy_not_nc_file"
int
test_ncmpi_open(void)
{
    int err, fd;
    int ncid;
    int ncid2;
    int nok=0;
    ssize_t w_len;
    
    /* Try to open a nonexistent file */
    err = ncmpi_open(comm, "tooth-fairy.nc", NC_NOWRITE, info, &ncid);/* should fail */

    /* on some systems, opening an nonexisting file will actually create the
     * file. In this case, we print the error messages on screen and move on
     * to the next test, instead of aborting the entire test.
     */
    if (err == NC_NOERR)
 	fprintf(stderr, "opening a nonexistent file expects to fail, but got NC_NOERR\n");
    else if (err != NC_ENOENT)
	fprintf(stderr, "opening a nonexistent file expects NC_ENOENT, but got %s\n",nc_err_code_name(err));
    else {
        /* printf("Expected error message complaining: \"File tooth-fairy.nc does not exist\"\n"); */
        nok++;
    }

    /* create a not-nc file */
    fd = open(NOT_NC_FILE, O_CREAT|O_WRONLY, 0600);
    w_len = write(fd, "0123456789abcdefghijklmnopqrstuvwxyz", 36);
    assert(w_len >= 0);
    close(fd);

    /* Open a file that is not a netCDF file. */
    err = ncmpi_open(comm, NOT_NC_FILE, NC_NOWRITE, info, &ncid);/* should fail */
    IF (err != NC_ENOTNC)
	error("ncmpi_open of non-netCDF file: expecting NC_ENOTNC but got %s", nc_err_code_name(err));
    ELSE_NOK

    /* delete the not-nc file */
    unlink(NOT_NC_FILE);

    /* Open a netCDF file in read-only mode, check that write fails */
    err = ncmpi_open(comm, testfile, NC_NOWRITE, info, &ncid);
    IF (err != NC_NOERR)
	error("ncmpi_open: %s", ncmpi_strerror(err));
    ELSE_NOK
    err = ncmpi_redef(ncid);	/* should fail */
    IF (err != NC_EPERM)
	error("ncmpi_redef of read-only file should fail");
    /* Opened OK, see if can open again and get a different netCDF ID */
    err = ncmpi_open(comm, testfile, NC_NOWRITE, info, &ncid2);
    IF (err != NC_NOERR)
	error("ncmpi_open: %s", ncmpi_strerror(err));
    else {
	ncmpi_close(ncid2);
        nok++;
    }
    IF (ncid2 == ncid)
	error("netCDF IDs for first and second ncmpi_open calls should differ");

    if (! read_only) {		/* tests using netCDF scratch file */
	err = ncmpi_create(comm, scratch, NC_NOCLOBBER|extra_flags, info, &ncid2);
	IF (err != NC_NOERR) 
	    error("ncmpi_create: %s", ncmpi_strerror(err));
	else 
	    ncmpi_close(ncid2);
	err = ncmpi_open(comm, scratch, NC_WRITE, info, &ncid2);
	IF (err != NC_NOERR) 
	    error("ncmpi_open: %s", ncmpi_strerror(err));
	else {
	    ncmpi_close(ncid2);
            nok++;
        }
	err = ncmpi_delete(scratch, info);
	IF (err != NC_NOERR) 
	    error("remove of %s failed", scratch);
    }

    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
	error("ncmpi_close: %s", ncmpi_strerror(err));
    return nok;
}


/* 
 * Test ncmpi_close.
 *    Try to close a netCDF file twice, check whether second close fails.
 *    Try on bad handle, check error return.
 *    Try in define mode and data mode.
 */
int
test_ncmpi_close(void)
{
    int ncid, nok=0;
    int err = ncmpi_open(comm, testfile, NC_NOWRITE, info, &ncid);

    IF (err != NC_NOERR)
	error("ncmpi_open: %s", ncmpi_strerror(err));

    /* Close a netCDF file twice, second time should fail */
    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
	error("ncmpi_close failed: %s", ncmpi_strerror(err));
    ELSE_NOK
    err = ncmpi_close(ncid);
    IF (err != NC_EBADID)
	error("ncmpi_close of closed file should have failed");
    ELSE_NOK
    
    /* Try with a bad netCDF ID */
    err = ncmpi_close(BAD_ID);/* should fail */
    IF (err != NC_EBADID)
	error("ncmpi_close with bad netCDF ID returned wrong error (%d)", err);
    ELSE_NOK

    /* Close in data mode */
    err = ncmpi_open(comm, testfile, NC_NOWRITE, info, &ncid);
    IF (err != NC_NOERR)
	error("ncmpi_open: %s", ncmpi_strerror(err));
    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
	error("ncmpi_close in data mode failed: %s", ncmpi_strerror(err));
    ELSE_NOK

    if (! read_only) {		/* tests using netCDF scratch file */
        err = ncmpi_create(comm, scratch, NC_NOCLOBBER|extra_flags, info, &ncid);
        IF (err != NC_NOERR) 
            error("ncmpi_create: %s", ncmpi_strerror(err));
	err = ncmpi_close(ncid);
	IF (err != NC_NOERR)
	    error("ncmpi_close in define mode: %s", ncmpi_strerror(err));
        ELSE_NOK
        err = ncmpi_delete(scratch, info);
        IF (err != NC_NOERR)
            error("remove of %s failed", scratch);
    }
    return nok;
}


/* 
 * Test ncmpi_inq.
 *    Try on bad handle, check error return.
 *    Try in data mode, check returned values.
 *    Try asking for subsets of info.
 * If in writable section of tests,
 *    Try in define mode, after adding an unlimited dimension, variable.
 * On exit, any open netCDF files are closed.
 */
int
test_ncmpi_inq(void)
{
    int ncid;
    int ndims;			/* number of dimensions */
    int nvars;			/* number of variables */
    int ngatts;			/* number of global attributes */
    int recdim;			/* id of unlimited dimension */
    int err = ncmpi_open(comm, testfile, NC_NOWRITE, info, &ncid);
    int nok=0;

    IF (err != NC_NOERR)
	error("ncmpi_open: %s", ncmpi_strerror(err));
    
    /* Try on bad handle */
    err = ncmpi_inq(BAD_ID, 0, 0, 0, 0);
    IF (err != NC_EBADID)
	error("bad ncid: status = %d", err);
    ELSE_NOK
    
    err = ncmpi_inq(ncid, &ndims, &nvars, &ngatts, &recdim);
    IF (err != NC_NOERR)
	error("ncmpi_inq: %s", ncmpi_strerror(err));
    else IF (ndims != NDIMS)
	error("ncmpi_inq: wrong number of dimensions returned, %d", ndims);
    else IF (nvars != numVars)
	error("ncmpi_inq: wrong number of variables returned, %d", nvars);
    else IF (ngatts != numGatts)
	error("ncmpi_inq: wrong number of global atts returned, %d", ngatts);
    else IF (recdim != RECDIM)
	error("ncmpi_inq: wrong record dimension ID returned, %d", recdim);
    ELSE_NOK
    
    /* Inguire for no info (useless, but should still work) */
    err = ncmpi_inq(ncid, 0, 0, 0, 0);
    IF (err != NC_NOERR)
	error("ncmpi_inq for no info failed: %s", ncmpi_strerror(err));
    ELSE_NOK

    /* Inguire for subsets of info */
    ngatts = numGatts - 1;	/* wipe out previous correct value */
    err = ncmpi_inq(ncid, 0, 0, &ngatts, 0);
    IF (err != NC_NOERR)
	error("ncmpi_inq for one item failed: %s", ncmpi_strerror(err));
    else IF (ngatts != numGatts)
	error("ncmpi_inq subset: wrong number of global atts returned, %d", ngatts);
    ELSE_NOK
    ndims = NDIMS - 1;
    nvars = numVars - 1;
    err = ncmpi_inq(ncid, &ndims, &nvars, 0, 0);
    IF (err != NC_NOERR)
	error("ncmpi_inq for two items failed: %s", ncmpi_strerror(err));
    else IF (ndims != NDIMS)
	error("ncmpi_inq subset: wrong number of dimensions returned, %d", ndims);
    else IF (nvars != numVars)
	error("ncmpi_inq subset: wrong number of variables returned, %d", nvars);
    ELSE_NOK

    if (! read_only) {		/* tests using netCDF scratch file */
	int ncid2;		/* for scratch netCDF dataset */

        err = ncmpi_create(comm, scratch, NC_NOCLOBBER|extra_flags, info, &ncid2);
        IF (err != NC_NOERR) {
            error("ncmpi_create: %s", ncmpi_strerror(err));
	} else {		/* add dim, var, gatt, check inq */
	    int ndims0;
	    int nvars0;
	    int ngatts0;
	    int recdim0;
	    err = ncmpi_enddef(ncid2); /* enter data mode */
	    err = ncmpi_inq(ncid2, &ndims0, &nvars0, &ngatts0, &recdim0);
	    IF (err != NC_NOERR)
		error("ncmpi_inq: %s", ncmpi_strerror(err));
            ELSE_NOK
	    err = ncmpi_redef(ncid2); /* enter define mode */
	    /* Check that inquire still works in define mode */
	    err = ncmpi_inq(ncid2, &ndims, &nvars, &ngatts, &recdim);
	    IF (err != NC_NOERR)
		error("ncmpi_inq in define mode: %s", ncmpi_strerror(err));
	    else IF (ndims != ndims0)
		error("ncmpi_inq in define mode: ndims wrong, %d", ndims);
	    else IF (nvars != nvars0)
		error("ncmpi_inq in define mode: nvars wrong, %d", nvars);
	    else IF (ngatts != ngatts0)
		error("ncmpi_inq in define mode: ngatts wrong, %d", ngatts);
	    else IF (recdim != recdim0)
		error("ncmpi_inq in define mode: recdim wrong, %d", recdim);
            ELSE_NOK

	    {
		int did, vid;
		/* Add dim, var, global att */
		err = ncmpi_def_dim(ncid2, "inqd", 1L, &did);
		IF (err != NC_NOERR)
		    error("ncmpi_def_dim: %s", ncmpi_strerror(err));
		err = ncmpi_def_var(ncid2, "inqv", NC_FLOAT, 0, 0, &vid);
		IF (err != NC_NOERR)
		    error("ncmpi_def_var: %s", ncmpi_strerror(err));
	    }
	    err = ncmpi_put_att_text(ncid2, NC_GLOBAL, "inqa", 1+strlen("stuff"),
				   "stuff");
	    IF (err != NC_NOERR)
		error("ncmpi_put_att_text: %s", ncmpi_strerror(err));

	    /* Make sure ncmpi_inq sees the additions while in define mode */
	    err = ncmpi_inq(ncid2, &ndims, &nvars, &ngatts, &recdim);
	    IF (err != NC_NOERR)
		error("ncmpi_inq in define mode: %s", ncmpi_strerror(err));
	    else IF (ndims != ndims0 + 1)
		error("ncmpi_inq in define mode: ndims wrong, %d", ndims);
	    else IF (nvars != nvars0 + 1)
		error("ncmpi_inq in define mode: nvars wrong, %d", nvars);
	    else IF (ngatts != ngatts0 + 1)
		error("ncmpi_inq in define mode: ngatts wrong, %d", ngatts);
            ELSE_NOK
	    err = ncmpi_enddef(ncid2);
	    IF (err != NC_NOERR)
		error("ncmpi_enddef: %s", ncmpi_strerror(err));

	    /* Make sure ncmpi_inq stills sees additions in data mode */
	    err = ncmpi_inq(ncid2, &ndims, &nvars, &ngatts, &recdim);
	    IF (err != NC_NOERR)
		error("ncmpi_inq failed in data mode: %s", ncmpi_strerror(err));
	    else IF (ndims != ndims0 + 1)
		error("ncmpi_inq in define mode: ndims wrong, %d", ndims);
	    else IF (nvars != nvars0 + 1)
		error("ncmpi_inq in define mode: nvars wrong, %d", nvars);
	    else IF (ngatts != ngatts0 + 1)
		error("ncmpi_inq in define mode: ngatts wrong, %d", ngatts);
            ELSE_NOK
	    ncmpi_close(ncid2);
	    err = ncmpi_delete(scratch, info);
	    IF (err != NC_NOERR)
		error("remove of %s failed", scratch);
	}
    }

    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
	error("ncmpi_close: %s", ncmpi_strerror(err));
    return nok;
}


int
test_ncmpi_inq_natts(void)
{
    int ncid;
    int ngatts;			/* number of global attributes */
    int err, nok=0;

    err = ncmpi_inq_natts(BAD_ID, &ngatts);
    IF (err != NC_EBADID)
	error("bad ncid: status = %d", err);
    ELSE_NOK
    err = ncmpi_open(comm, testfile, NC_NOWRITE, info, &ncid);
    IF (err != NC_NOERR)
	error("ncmpi_open: %s", ncmpi_strerror(err));
    err = ncmpi_inq_natts(ncid, &ngatts);
    IF (err != NC_NOERR)
	error("ncmpi_inq_natts: %s", ncmpi_strerror(err));
    else IF (ngatts != numGatts)
	error("ncmpi_inq_natts: wrong number of global atts returned, %d", ngatts);
    ELSE_NOK
    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
	error("ncmpi_close: %s", ncmpi_strerror(err));
    return nok;
}


int
test_ncmpi_inq_ndims(void)
{
    int ncid;
    int ndims;
    int err;
    int nok=0;

    err = ncmpi_inq_ndims(BAD_ID, &ndims);
    IF (err != NC_EBADID)
	error("bad ncid: status = %d", err);
    ELSE_NOK
    err = ncmpi_open(comm, testfile, NC_NOWRITE, info, &ncid);
    IF (err != NC_NOERR)
	error("ncmpi_open: %s", ncmpi_strerror(err));
    err = ncmpi_inq_ndims(ncid, &ndims);
    IF (err != NC_NOERR)
	error("ncmpi_inq_ndims: %s", ncmpi_strerror(err));
    else IF (ndims != NDIMS)
	error("ncmpi_inq_ndims: wrong number returned, %d", ndims);
    ELSE_NOK
    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
	error("ncmpi_close: %s", ncmpi_strerror(err));
    return nok;
}


int
test_ncmpi_inq_nvars(void)
{
    int ncid;
    int nvars;
    int err;
    int nok=0;

    err = ncmpi_inq_nvars(BAD_ID, &nvars);
    IF (err != NC_EBADID)
	error("bad ncid: status = %d", err);
    ELSE_NOK
    err = ncmpi_open(comm, testfile, NC_NOWRITE, info, &ncid);
    IF (err != NC_NOERR)
	error("ncmpi_open: %s", ncmpi_strerror(err));
    err = ncmpi_inq_nvars(ncid, &nvars);
    IF (err != NC_NOERR)
	error("ncmpi_inq_nvars: %s", ncmpi_strerror(err));
    else IF (nvars != numVars)
	error("ncmpi_inq_nvars: wrong number returned, %d", nvars);
    ELSE_NOK
    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
	error("ncmpi_close: %s", ncmpi_strerror(err));
    return nok;
}


int
test_ncmpi_inq_unlimdim(void)
{
    int ncid;
    int unlimdim;
    int err;
    int nok=0;

    err = ncmpi_inq_unlimdim(BAD_ID, &unlimdim);
    IF (err != NC_EBADID)
	error("bad ncid: status = %d", err);
    ELSE_NOK
    err = ncmpi_open(comm, testfile, NC_NOWRITE, info, &ncid);
    IF (err != NC_NOERR)
	error("ncmpi_open: %s", ncmpi_strerror(err));
    err = ncmpi_inq_unlimdim(ncid, &unlimdim);
    IF (err != NC_NOERR)
	error("ncmpi_inq_unlimdim: %s", ncmpi_strerror(err));
    else IF (unlimdim != RECDIM)
	error("ncmpi_inq_unlimdim: wrong number returned, %d", unlimdim);
    ELSE_NOK
    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
	error("ncmpi_close: %s", ncmpi_strerror(err));
    return nok;
}


int
test_ncmpi_inq_dimid(void)
{
    int ncid;
    int dimid;
    int i;
    int err;
    int nok=0;

    err = ncmpi_open(comm, testfile, NC_NOWRITE, info, &ncid);
    IF (err != NC_NOERR)
	error("ncmpi_open: %s", ncmpi_strerror(err));
    err = ncmpi_inq_dimid(ncid, "noSuch", &dimid);
    IF (err != NC_EBADDIM)
	error("bad dim name: status = %d", err);
    ELSE_NOK
    for (i = 0; i < NDIMS; i++) {
	err = ncmpi_inq_dimid(BAD_ID, dim_name[i], &dimid);
	IF (err != NC_EBADID)
	    error("bad ncid: status = %d", err);
        ELSE_NOK
	err = ncmpi_inq_dimid(ncid, dim_name[i], &dimid);
	IF (err != NC_NOERR)
	    error("ncmpi_inq_dimid: %s", ncmpi_strerror(err));
	else IF (dimid != i)
	    error("expected %d, got %d", i, dimid);
        ELSE_NOK
    }
    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
	error("ncmpi_close: %s", ncmpi_strerror(err));
    return nok;
}


int
test_ncmpi_inq_dim(void)
{
    int ncid;
    int i;
    int err;
    char name[NC_MAX_NAME];
    MPI_Offset length;
    int nok=0;

    err = ncmpi_open(comm, testfile, NC_NOWRITE, info, &ncid);
    IF (err != NC_NOERR)
	error("ncmpi_open: %s", ncmpi_strerror(err));
    for (i = 0; i < NDIMS; i++) {
	err = ncmpi_inq_dim(BAD_ID, i, name, &length);
        IF (err != NC_EBADID)
	    error("bad ncid: status = %d", err);
        ELSE_NOK
	err = ncmpi_inq_dim(ncid, BAD_DIMID, name, &length);
        IF (err != NC_EBADDIM)
	    error("bad dimid: status = %d", err);
        ELSE_NOK
	err = ncmpi_inq_dim(ncid, i, 0, 0);
	IF (err != NC_NOERR)
	    error("ncmpi_inq_dim: %s", ncmpi_strerror(err));
        ELSE_NOK
	err = ncmpi_inq_dim(ncid, i, name, &length);
	IF (err != NC_NOERR)
	    error("ncmpi_inq_dim: %s", ncmpi_strerror(err));
	else IF (strcmp(dim_name[i],name)) 
	    error("name expected: %s, got: %s",dim_name[i],name);
	else IF (dim_len[i] != length)
	    error("size expected: %d, got: %d",dim_len[i],length);
        ELSE_NOK
	err = ncmpi_inq_dim(ncid, i, name, 0);
        IF (err != NC_NOERR)
	    error("ncmpi_inq_dim: %s", ncmpi_strerror(err));
	else IF (strcmp(dim_name[i],name)) 
	    error("name expected: %s, got: %s",dim_name[i],name);
        ELSE_NOK
	err = ncmpi_inq_dim(ncid, i, 0, &length);
        IF (err != NC_NOERR)
	    error("ncmpi_inq_dim: %s", ncmpi_strerror(err));
	else IF (dim_len[i] != length)
	    error("size expected: %d, got: %d",dim_len[i],length);
        ELSE_NOK
    }
    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
	error("ncmpi_close: %s", ncmpi_strerror(err));
    return nok;
}


int
test_ncmpi_inq_dimlen(void)
{
    int ncid;
    int i;
    int err;
    MPI_Offset  length;
    int nok=0;

    err = ncmpi_open(comm, testfile, NC_NOWRITE, info, &ncid);
    IF (err != NC_NOERR)
	error("ncmpi_open: %s", ncmpi_strerror(err));
    for (i = 0; i < NDIMS; i++) {
	err = ncmpi_inq_dimlen(BAD_ID, i, &length);
        IF (err != NC_EBADID)
	    error("bad ncid: status = %d", err);
        ELSE_NOK
	err = ncmpi_inq_dimlen(ncid, BAD_DIMID, &length);
        IF (err != NC_EBADDIM)
	    error("bad dimid: status = %d", err);
        ELSE_NOK
	err = ncmpi_inq_dimlen(ncid, i, &length);
	IF (err != NC_NOERR)
	    error("ncmpi_inq_dimlen: %s", ncmpi_strerror(err));
	else IF (dim_len[i] != length)
	    error("size expected: %d, got: %d",dim_len[i],length);
        ELSE_NOK
    }
    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
	error("ncmpi_close: %s", ncmpi_strerror(err));
    return nok;
}


int
test_ncmpi_inq_dimname(void)
{
    int ncid;
    int i;
    int err;
    char name[NC_MAX_NAME];
    int nok=0;

    err = ncmpi_open(comm, testfile, NC_NOWRITE, info, &ncid);
    IF (err != NC_NOERR)
	error("ncmpi_open: %s", ncmpi_strerror(err));
    for (i = 0; i < NDIMS; i++) {
	err = ncmpi_inq_dimname(BAD_ID, i, name);
        IF (err != NC_EBADID)
	    error("bad ncid: status = %d", err);
        ELSE_NOK
	err = ncmpi_inq_dimname(ncid, BAD_DIMID, name);
        IF (err != NC_EBADDIM)
	    error("bad dimid: status = %d", err);
        ELSE_NOK
	err = ncmpi_inq_dimname(ncid, i, name);
	IF (err != NC_NOERR)
	    error("ncmpi_inq_dimname: %s", ncmpi_strerror(err));
	else IF (strcmp(dim_name[i],name)) 
	    error("name expected: %s, got: %s",dim_name[i],name);
        ELSE_NOK
    }
    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
	error("ncmpi_close: %s", ncmpi_strerror(err));
    return nok;
}


int
test_ncmpi_inq_varid(void)
{
    int ncid;
    int varid;
    int i;
    int err;
    int nok=0;

    err = ncmpi_open(comm, testfile, NC_NOWRITE, info, &ncid);
    IF (err != NC_NOERR)
	error("ncmpi_open: %s", ncmpi_strerror(err));

    err = ncmpi_inq_varid(ncid, "noSuch", &varid);
    IF (err != NC_ENOTVAR)
	error("bad ncid: status = %d", err);
    ELSE_NOK

    for (i = 0; i < numVars; i++) {
	err = ncmpi_inq_varid(BAD_ID, var_name[i], &varid);
        IF (err != NC_EBADID)
	    error("bad ncid: status = %d", err);
        ELSE_NOK
	err = ncmpi_inq_varid(ncid, var_name[i], &varid);
        IF (err != NC_NOERR)
	    error("ncmpi_inq_varid: %s", ncmpi_strerror(err));
	else IF (varid != i)
	    error("expected %d, got %d", i, varid);
        ELSE_NOK
    }

    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
	error("ncmpi_close: %s", ncmpi_strerror(err));
    return nok;
}


int
test_ncmpi_inq_var(void)
{
    int ncid;
    int i;
    int err;
    char name[NC_MAX_NAME];
    nc_type datatype;
    int ndims;
    int dimids[MAX_RANK];
    int natts;
    int nok=0;

    err = ncmpi_open(comm, testfile, NC_NOWRITE, info, &ncid);
    IF (err != NC_NOERR)
	error("ncmpi_open: %s", ncmpi_strerror(err));
    for (i = 0; i < numVars; i++) {
	err = ncmpi_inq_var(BAD_ID, i, name, &datatype, &ndims, dimids, &natts);
        IF (err != NC_EBADID)
	    error("bad ncid: status = %d", err);
        ELSE_NOK
	err = ncmpi_inq_var(ncid,BAD_VARID,name,&datatype,&ndims,dimids,&natts);
        IF (err != NC_ENOTVAR)
	    error("bad var id: status = %d", err);
        ELSE_NOK
	err = ncmpi_inq_var(ncid, i, 0, 0, 0, 0, 0);
	IF (err != NC_NOERR)
	    error("ncmpi_inq_var: %s", ncmpi_strerror(err));
        ELSE_NOK
	err = ncmpi_inq_var(ncid, i, name, &datatype, &ndims, dimids, &natts);
	IF (err != NC_NOERR)
	    error("ncmpi_inq_var: %s", ncmpi_strerror(err));
	else IF (strcmp(var_name[i],name)) 
	    error("name expected: %s, got: %s",var_name[i],name);
	else IF (var_type[i] != datatype)
	    error("type expected: %d, got: %d",var_type[i],datatype);
	else IF (var_rank[i] != ndims)
	    error("ndims expected: %d, got: %d",var_rank[i],ndims);
	else IF (!int_vec_eq(var_dimid[i],dimids,ndims))
	    error("unexpected dimid");
	else IF (var_natts[i] != natts)
	    error("natts expected: %d, got: %d",var_natts[i],natts);
        ELSE_NOK
	err = ncmpi_inq_var(ncid, i, name, 0, 0, 0, 0);
        IF (err != NC_NOERR)
	    error("ncmpi_inq_var: %s", ncmpi_strerror(err));
	else IF (strcmp(var_name[i],name)) 
	    error("name expected: %s, got: %s",var_name[i],name);
        ELSE_NOK
	err = ncmpi_inq_var(ncid, i, 0, &datatype, 0, 0, 0);
        IF (err != NC_NOERR)
	    error("ncmpi_inq_var: %s", ncmpi_strerror(err));
        else IF (var_type[i] != datatype)
            error("type expected: %d, got: %d",var_type[i],datatype);
        ELSE_NOK
	err = ncmpi_inq_var(ncid, i, 0, 0, &ndims, 0, 0);
        IF (err != NC_NOERR)
	    error("ncmpi_inq_var: %s", ncmpi_strerror(err));
        else IF (var_rank[i] != ndims)
            error("ndims expected: %d, got: %d",var_rank[i],ndims);
        ELSE_NOK
	err = ncmpi_inq_var(ncid, i, 0, 0, 0, dimids, 0);
        IF (err != NC_NOERR)
	    error("ncmpi_inq_var: %s", ncmpi_strerror(err));
        else IF (!int_vec_eq(var_dimid[i],dimids,ndims))
            error("unexpected dimid");
        ELSE_NOK
	err = ncmpi_inq_var(ncid, i, 0, 0, 0, 0, &natts);
        IF (err != NC_NOERR)
	    error("ncmpi_inq_var: %s", ncmpi_strerror(err));
        else IF (var_natts[i] != natts)
            error("natts expected: %d, got: %d",var_natts[i],natts);
        ELSE_NOK
    }
    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
	error("ncmpi_close: %s", ncmpi_strerror(err));
    return nok;
}


int
test_ncmpi_inq_vardimid(void)
{
    int ncid;
    int i;
    int err;
    int dimids[MAX_RANK];
    int nok=0;

    err = ncmpi_open(comm, testfile, NC_NOWRITE, info, &ncid);
    IF (err != NC_NOERR)
	error("ncmpi_open: %s", ncmpi_strerror(err));
    for (i = 0; i < numVars; i++) {
	err = ncmpi_inq_vardimid(BAD_ID, i, dimids);
        IF (err != NC_EBADID)
	    error("bad ncid: status = %d", err);
        ELSE_NOK
	err = ncmpi_inq_vardimid(ncid, BAD_VARID, dimids);
        IF (err != NC_ENOTVAR)
	    error("bad var id: status = %d", err);
        ELSE_NOK
	err = ncmpi_inq_vardimid(ncid, i, dimids);
	IF (err != NC_NOERR)
	    error("ncmpi_inq_vardimid: %s", ncmpi_strerror(err));
	else IF (!int_vec_eq(var_dimid[i], dimids, var_rank[i]))
	    error("unexpected dimid");
        ELSE_NOK
    }
    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
	error("ncmpi_close: %s", ncmpi_strerror(err));
    return nok;
}


int
test_ncmpi_inq_varname(void)
{
    int ncid;
    int i;
    int err;
    char name[NC_MAX_NAME];
    int nok=0;

    err = ncmpi_open(comm, testfile, NC_NOWRITE, info, &ncid);
    IF (err != NC_NOERR)
	error("ncmpi_open: %s", ncmpi_strerror(err));
    for (i = 0; i < numVars; i++) {
	err = ncmpi_inq_varname(BAD_ID, i, name);
        IF (err != NC_EBADID)
	    error("bad ncid: status = %d", err);
        nok++;
	err = ncmpi_inq_varname(ncid, BAD_VARID, name);
        IF (err != NC_ENOTVAR)
	    error("bad var id: status = %d", err);
        ELSE_NOK
	err = ncmpi_inq_varname(ncid, i, name);
	IF (err != NC_NOERR)
	    error("ncmpi_inq_varname: %s", ncmpi_strerror(err));
	else IF (strcmp(var_name[i],name)) 
	    error("name expected: %s, got: %s",var_name[i],name);
        ELSE_NOK
    }
    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
	error("ncmpi_close: %s", ncmpi_strerror(err));
    return nok;
}


int
test_ncmpi_inq_varnatts(void)
{
    int ncid;
    int i;
    int err;
    int natts;
    int nok=0;

    err = ncmpi_open(comm, testfile, NC_NOWRITE, info, &ncid);
    IF (err != NC_NOERR)
	error("ncmpi_open: %s", ncmpi_strerror(err));
    for (i = -1; i < numVars; i++) {
	err = ncmpi_inq_varnatts(BAD_ID, i, &natts);
        IF (err != NC_EBADID)
	    error("bad ncid: status = %d", err);
        ELSE_NOK
	err = ncmpi_inq_varnatts(ncid, BAD_VARID, &natts);
        IF (err != NC_ENOTVAR)
	    error("bad var id: status = %d", err);
        ELSE_NOK
	err = ncmpi_inq_varnatts(ncid, VARID(i), &natts);
	IF (err != NC_NOERR)
	    error("ncmpi_inq_varnatts: %s", ncmpi_strerror(err));
        else IF (NATTS(i) != natts)
            error("natts expected: %d, got: %d",NATTS(i),natts);
        ELSE_NOK
    }
    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
	error("ncmpi_close: %s", ncmpi_strerror(err));
    return nok;
}


int
test_ncmpi_inq_varndims(void)
{
    int ncid;
    int i;
    int err;
    int ndims;
    int nok=0;

    err = ncmpi_open(comm, testfile, NC_NOWRITE, info, &ncid);
    IF (err != NC_NOERR)
	error("ncmpi_open: %s", ncmpi_strerror(err));
    for (i = 0; i < numVars; i++) {
	err = ncmpi_inq_varndims(BAD_ID, i, &ndims);
        IF (err != NC_EBADID)
	    error("bad ncid: status = %d", err);
        ELSE_NOK
	err = ncmpi_inq_varndims(ncid, BAD_VARID, &ndims);
        IF (err != NC_ENOTVAR)
	    error("bad var id: status = %d", err);
        ELSE_NOK
	err = ncmpi_inq_varndims(ncid, i, &ndims);
	IF (err != NC_NOERR)
	    error("ncmpi_inq_varndims: %s", ncmpi_strerror(err));
        else IF (var_rank[i] != ndims)
            error("ndims expected: %d, got: %d",var_rank[i],ndims);
        ELSE_NOK
    }
    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
	error("ncmpi_close: %s", ncmpi_strerror(err));
    return nok;
}


int
test_ncmpi_inq_vartype(void)
{
    int ncid;
    int i;
    int err;
    nc_type datatype;
    int nok=0;

    err = ncmpi_open(comm, testfile, NC_NOWRITE, info, &ncid);
    IF (err != NC_NOERR)
	error("ncmpi_open: %s", ncmpi_strerror(err));
    for (i = 0; i < numVars; i++) {
	err = ncmpi_inq_vartype(BAD_ID, i, &datatype);
        IF (err != NC_EBADID)
	    error("bad ncid: status = %d", err);
        ELSE_NOK
	err = ncmpi_inq_vartype(ncid, BAD_VARID, &datatype);
        IF (err != NC_ENOTVAR)
	    error("bad var id: status = %d", err);
        ELSE_NOK
	err = ncmpi_inq_vartype(ncid, i, &datatype);
	IF (err != NC_NOERR)
	    error("ncmpi_inq_vartype: %s", ncmpi_strerror(err));
        else IF (var_type[i] != datatype)
            error("type expected: %d, got: %d", var_type[i], datatype);
        ELSE_NOK
    }
    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
	error("ncmpi_close: %s", ncmpi_strerror(err));
    return nok;
}


/*
 * Test ncmpi_put_var1
 */
int
test_ncmpi_get_var1(void)
{
    int ncid;
    int i;
    int j;
    int err;
    double expect;
    int nok = 0;		/* count of valid comparisons */
    double buf[1];		/* (void *) buffer */
    double value;
    MPI_Offset index[MAX_RANK];
    MPI_Datatype datatype;

    err = ncmpi_open(comm, testfile, NC_NOWRITE, info, &ncid);
    IF (err != NC_NOERR)
	error("ncmpi_open: %s", ncmpi_strerror(err));
    err = ncmpi_begin_indep_data(ncid);
    for (i = 0; i < numVars; i++) {
        datatype = nc_mpi_type(var_type[i]);
	for (j = 0; j < var_rank[i]; j++)
	    index[j] = 0;
        err = ncmpi_get_var1(BAD_ID, i, index, buf, 1, datatype);
        IF (err != NC_EBADID)
	    error("bad ncid: status = %d", err);
        ELSE_NOK
        err = ncmpi_get_var1(ncid, BAD_VARID, index, buf, 1, datatype);
        IF (err != NC_ENOTVAR)
	    error("bad var id: status = %d", err);
        ELSE_NOK
	for (j = 0; j < var_rank[i]; j++) {
	    index[j] = var_shape[i][j];
	    err = ncmpi_get_var1(ncid, i, index, buf, 1, datatype);
            IF (err != NC_EINVALCOORDS)
                error("bad index: status = %d", err);
            ELSE_NOK
	    index[j] = 0;
	}
        for (j = 0; j < var_nels[i]; j++) {
	    err = toMixedBase(j, var_rank[i], var_shape[i], index);
	    IF (err != NC_NOERR)
		error("error in toMixedBase 2");
	    expect = hash( var_type[i], var_rank[i], index );
            if (var_rank[i] == 0 && i%2 )
		err = ncmpi_get_var1(ncid, i, NULL, buf, 1, datatype);
	    else
		err = ncmpi_get_var1(ncid, i, index, buf, 1, datatype);
	    IF (err != NC_NOERR)
		error("%s", ncmpi_strerror(err));
            ELSE_NOK
	    err = nc2dbl( var_type[i], buf, &value );
	    IF (err != NC_NOERR)
		error("error in nc2dbl");
	    if (inRange(expect,var_type[i])) {
		IF (!equal2(value,expect,var_type[i]))
		    error("expected: %G, got: %G", expect, value);
                ELSE_NOK
	    }
        }
    }
    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
	error("ncmpi_close: %s", ncmpi_strerror(err));
    return nok;
}


/*
 * Test ncmpi_get_vara
 * Choose a random point dividing each dim into 2 parts
 * Get 2^rank (nslabs) slabs so defined
 * Each get overwrites buffer, so check after each get.
 */
int
test_ncmpi_get_vara(void)
{
    int ncid, d, i, j, k, err, nels, nslabs;
    int nok = 0;      /* count of valid comparisons */
    MPI_Offset start[MAX_RANK];
    MPI_Offset edge[MAX_RANK];
    MPI_Offset index[MAX_RANK];
    MPI_Offset mid[MAX_RANK];
    MPI_Datatype datatype;
    double buf[MAX_NELS];	/* (void *) buffer */
    double expect;

    err = ncmpi_open(comm, testfile, NC_NOWRITE, info, &ncid);
    IF (err != NC_NOERR)
	error("ncmpi_open: %s", ncmpi_strerror(err));
    err = ncmpi_begin_indep_data(ncid);
    for (i = 0; i < numVars; i++) {
        datatype = nc_mpi_type(var_type[i]);
        assert(var_rank[i] <= MAX_RANK);
        assert(var_nels[i] <= MAX_NELS);
        for (j = 0; j < var_rank[i]; j++) {
            start[j] = 0;
            edge[j] = 1;
        }
        err = ncmpi_get_vara(BAD_ID, i, start, edge, buf, 1, datatype);
        IF (err != NC_EBADID)
	    error("bad ncid: status = %d", err);
        ELSE_NOK
        err = ncmpi_get_vara(ncid, BAD_VARID, start, edge, buf, 1, datatype);
        IF (err != NC_ENOTVAR)
	    error("bad var id: status = %d", err);
        ELSE_NOK
	for (j = 0; j < var_rank[i]; j++) {
	    start[j] = var_shape[i][j];
	    err = ncmpi_get_vara(ncid, i, start, edge, buf, 1, datatype);
            IF (err != NC_EINVALCOORDS)
                error("bad index: status = %d", err);
            ELSE_NOK
	    start[j] = 0;
	    edge[j] = var_shape[i][j] + 1;
	    err = ncmpi_get_vara(ncid, i, start, edge, buf, 1, datatype);
            IF (err != NC_EEDGE)
		error("bad edge: status = %d", err);
            ELSE_NOK
	    edge[j] = 1;
	}
            /* Choose a random point dividing each dim into 2 parts */
            /* get 2^rank (nslabs) slabs so defined */
        nslabs = 1;
        for (j = 0; j < var_rank[i]; j++) {
            mid[j] = roll( var_shape[i][j] );
            nslabs *= 2;
        }
            /* bits of k determine whether to get lower or upper part of dim */
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
            if (var_rank[i] == 0 && i%2 )
		err = ncmpi_get_vara(ncid, i, NULL, NULL, buf, nels, datatype);
	    else
		err = ncmpi_get_vara(ncid, i, start, edge, buf, nels, datatype);
	    IF (err != NC_NOERR) {
		error("%s", ncmpi_strerror(err));
	    } else {
	        nok++;
		for (j = 0; j < nels; j++) {
                    double got;
                    char *p = (char *) buf;
                    p += j * nctypelen(var_type[i]);
		    err = nc2dbl( var_type[i], p, & got );
		    IF (err != NC_NOERR)
			error("error in nc2dbl");
		    err = toMixedBase(j, var_rank[i], edge, index);
		    IF (err != NC_NOERR)
			error("error in toMixedBase 1");
		    for (d = 0; d < var_rank[i]; d++)
			index[d] += start[d];
		    expect = hash(var_type[i], var_rank[i], index);
		    if (inRange(expect,var_type[i])) {
			IF (!equal2(got,expect,var_type[i])) {
			    error("value read not that expected");
			    if (verbose) {
				error("\n");
				error("varid: %d, ", i);
				error("var_name: %s, ", var_name[i]);
				error("element number: %d ", j);
				error("expect: %g", expect);
				error("got: %g", got);
			    }
			}
		    }
		}
	    }
	}
    }
    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
        error("ncmpi_close: %s", ncmpi_strerror(err));
    return nok;
}


/*
 * Test ncmpi_get_vars
 * Choose a random point dividing each dim into 2 parts
 * Get 2^rank (nslabs) slabs so defined
 * Each get overwrites buffer, so check after each get.
 */
int
test_ncmpi_get_vars(void)
{
    int ncid;
    int d;
    int i;
    int j;
    int k;
    int m;
    int err;
    int nels;
    int nslabs;
    int nstarts;	/* number of different starts */
    int nok = 0;	/* total count of valid comparisons */
    int n;		/* count of valid comparisons within var */
    MPI_Offset start[MAX_RANK];
    MPI_Offset edge[MAX_RANK];
    MPI_Offset index[MAX_RANK];
    MPI_Offset index2[MAX_RANK];
    MPI_Offset mid[MAX_RANK];
    MPI_Offset count[MAX_RANK];
    MPI_Offset sstride[MAX_RANK];
    MPI_Offset stride[MAX_RANK];
    MPI_Datatype datatype;
    double buf[MAX_NELS];     /* (void *) buffer */
    char *p;			/* (void *) pointer */
    double expect;
    double got;

    err = ncmpi_open(comm, testfile, NC_NOWRITE, info, &ncid);
    IF (err != NC_NOERR)
	error("ncmpi_open: %s", ncmpi_strerror(err));
    err = ncmpi_begin_indep_data(ncid);
    for (i = 0; i < numVars; i++) {
        datatype = nc_mpi_type(var_type[i]);
        assert(var_rank[i] <= MAX_RANK);
        assert(var_nels[i] <= MAX_NELS);
        for (j = 0; j < var_rank[i]; j++) {
            start[j] = 0;
            edge[j] = 1;
            stride[j] = 1;
        }
        err = ncmpi_get_vars(BAD_ID, i, start, edge, stride, buf, 1, datatype);
        IF (err != NC_EBADID)
	    error("bad ncid: status = %d", err);
        ELSE_NOK
        err = ncmpi_get_vars(ncid, BAD_VARID, start, edge, stride, buf, 1, datatype);
        IF (err != NC_ENOTVAR)
	    error("bad var id: status = %d", err);
        ELSE_NOK
	for (j = 0; j < var_rank[i]; j++) {
	    start[j] = var_shape[i][j];
	    err = ncmpi_get_vars(ncid, i, start, edge, stride, buf, 1, datatype);
            IF (err != NC_EINVALCOORDS)
                error("bad index: status = %d", err);
            ELSE_NOK
	    start[j] = 0;
	    edge[j] = var_shape[i][j] + 1;
	    err = ncmpi_get_vars(ncid, i, start, edge, stride, buf, 1, datatype);
            IF (err != NC_EEDGE)
		error("bad edge: status = %d", err);
            ELSE_NOK
	    edge[j] = 1;
	    stride[j] = 0;
	    err = ncmpi_get_vars(ncid, i, start, edge, stride, buf, 1, datatype);
            IF (err != NC_ESTRIDE)
		error("bad stride: status = %d", err);
            ELSE_NOK
	    stride[j] = 1;
	}
        /* Choose a random point dividing each dim into 2 parts */
        /* get 2^rank (nslabs) slabs so defined */
        nslabs = 1;
        for (j = 0; j < var_rank[i]; j++) {
            mid[j] = roll( var_shape[i][j] );
            nslabs *= 2;
        }
        /* bits of k determine whether to get lower or upper part of dim */
	/* choose random stride from 1 to edge */
	n = 0;
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
		if (var_rank[i] == 0 && i%2 )
		    err = ncmpi_get_vars(ncid, i, NULL, NULL, NULL, buf, 1, datatype);
		else
		    err = ncmpi_get_vars(ncid, i, index, count, stride, buf, nels, datatype);
		IF (err != NC_NOERR) {
		    error("%s", ncmpi_strerror(err));
		} else {
		    nok++;
		    for (j = 0; j < nels; j++) {
			p = (char *) buf;
			p += j * nctypelen(var_type[i]);
			err = nc2dbl( var_type[i], p, & got );
			IF (err != NC_NOERR)
			    error("error in nc2dbl");
			err = toMixedBase(j, var_rank[i], count, index2);
			IF (err != NC_NOERR)
			    error("error in toMixedBase 1");
			for (d = 0; d < var_rank[i]; d++)
			    index2[d] = index[d] + index2[d] * stride[d];
			expect = hash(var_type[i], var_rank[i], index2);
			if (inRange(expect,var_type[i])) {
			    IF (!equal2(got,expect,var_type[i])) {
				error("value read not that expected");
				if (verbose) {
				    error("\n");
				    error("varid: %d, ", i);
				    error("var_name: %s, ", var_name[i]);
				    error("element number: %d ", j);
				    error("expect: %g, ", expect);
				    error("got: %g ", got);
				}
			    }
			}
			n++;
		    }
		}
	    }
	}
	IF (n != var_nels[i]) {
	    error("count != nels");
	    if (verbose) {
		error("\n");
		error("varid: %d, ", i);
		error("var_name: %s, ", var_name[i]);
		error("count: %d, ", n);
		error("nels: %d ", var_nels[i]);
	    }
	}
    }
    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
        error("ncmpi_close: %s", ncmpi_strerror(err));
    return nok;
}


/*
 * Test ncmpi_get_varm
 * Choose a random point dividing each dim into 2 parts
 * Get 2^rank (nslabs) slabs so defined
 * Choose random stride from 1 to edge
 * Buffer should end up being bit image of external variable.
 * So all gets for a variable store in different elements of buffer
 */
int
test_ncmpi_get_varm(void)
{
    int ncid;
    int i;
    int j;
    int k;
    int m;
    int err;
    int nslabs;
    int nstarts;	/* number of different starts */
    int nok = 0;	/* total count of valid comparisons */
    MPI_Offset start[MAX_RANK];
    MPI_Offset edge[MAX_RANK];
    MPI_Offset index[MAX_RANK];
    MPI_Offset mid[MAX_RANK];
    MPI_Offset count[MAX_RANK];
    MPI_Offset sstride[MAX_RANK];
    MPI_Offset stride[MAX_RANK];
    MPI_Offset imap[MAX_RANK];
    MPI_Offset imap2[MAX_RANK];
    MPI_Offset nels;
    MPI_Datatype datatype;
    double buf[MAX_NELS];	/* (void *) buffer */
    char *p;			/* (void *) pointer */
    double expect;
    double got;

    err = ncmpi_open(comm, testfile, NC_NOWRITE, info, &ncid);
    IF (err != NC_NOERR)
	error("ncmpi_open: %s", ncmpi_strerror(err));
    err = ncmpi_begin_indep_data(ncid);
    for (i = 0; i < numVars; i++) {
        datatype = nc_mpi_type(var_type[i]);
        assert(var_rank[i] <= MAX_RANK);
        assert(var_nels[i] <= MAX_NELS);
        for (j = 0; j < var_rank[i]; j++) {
            start[j] = 0;
            edge[j] = 1;
            stride[j] = 1;
        }
        if (var_rank[i] > 0) {
            j = var_rank[i] - 1;
            /* imap[j] = nctypelen(var_type[i]); */
            imap[j] = 1; /* in numbers of elements */
            for (; j > 0; j--)
                imap[j-1] = imap[j] * var_shape[i][j];
        }
        err = ncmpi_get_varm(BAD_ID, i, start, edge, stride, imap, buf, 1, datatype);
        IF (err != NC_EBADID)
	    error("bad ncid: status = %d", err);
        ELSE_NOK
        err = ncmpi_get_varm(ncid, BAD_VARID, start, edge, stride, imap, buf, 1, datatype);
        IF (err != NC_ENOTVAR)
	    error("bad var id: status = %d", err);
        ELSE_NOK
	for (j = 0; j < var_rank[i]; j++) {
	    start[j] = var_shape[i][j];
	    err = ncmpi_get_varm(ncid, i, start, edge, stride, imap, buf, 1, datatype);
            IF (err != NC_EINVALCOORDS)
                error("bad index: status = %d", err);
            ELSE_NOK
	    start[j] = 0;
	    edge[j] = var_shape[i][j] + 1;
	    err = ncmpi_get_varm(ncid, i, start, edge, stride, imap, buf, 1, datatype);
            IF (err != NC_EEDGE)
		error("bad edge: status = %d", err);
            ELSE_NOK
	    edge[j] = 1;
	    stride[j] = 0;
	    err = ncmpi_get_varm(ncid, i, start, edge, stride, imap, buf, 1, datatype);
            IF (err != NC_ESTRIDE)
		error("bad stride: status = %d", err);
            ELSE_NOK
	    stride[j] = 1;
	}
            /* Choose a random point dividing each dim into 2 parts */
            /* get 2^rank (nslabs) slabs so defined */
        nslabs = 1;
        for (j = 0; j < var_rank[i]; j++) {
            mid[j] = roll( var_shape[i][j] );
            nslabs *= 2;
        }
            /* bits of k determine whether to get lower or upper part of dim */
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
		if (var_rank[i] == 0 && i%2 ) {
		    err = ncmpi_get_varm(ncid, i, NULL, NULL, NULL, NULL, buf, var_nels[i], datatype);
		} else {
		    err = toMixedBase(m, var_rank[i], sstride, index);
		    IF (err != NC_NOERR)
			error("error in toMixedBase");
                    nels = 1;
		    for (j = 0; j < var_rank[i]; j++) {
			count[j] = 1 + (edge[j] - index[j] - 1) / stride[j];
			index[j] += start[j];
                        nels *= count[j];
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
		    err = ncmpi_get_varm(ncid, i, index, count, stride, imap2, p, nels, datatype);
		}
		IF (err != NC_NOERR)
		    error("%s", ncmpi_strerror(err));
                ELSE_NOK
	    }
	}
        p = (char *) buf;
	for (j = 0; j < var_nels[i]; j++) {
            err = toMixedBase(j, var_rank[i], var_shape[i], index);
            IF (err != NC_NOERR)
                error("error in toMixedBase");
            expect = hash( var_type[i], var_rank[i], index);
	    err = nc2dbl( var_type[i], p, & got );
	    IF (err != NC_NOERR)
		error("error in nc2dbl");
	    if (inRange(expect,var_type[i])) {
		IF (!equal2(got,expect,var_type[i])) {
		    error("value read not that expected");
		    if (verbose) {
			error("\n");
			error("varid: %d, ", i);
			error("var_name: %s, ", var_name[i]);
			error("element number: %d ", j);
			error("expect: %g, ", expect);
			error("got: %g ", got);
		    }
		}
                ELSE_NOK
	    }
            p += nctypelen(var_type[i]);
	}
    }
    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
        error("ncmpi_close: %s", ncmpi_strerror(err));
    return nok;
}


int
test_ncmpi_get_att(void)
{
    int ncid;
    int i;
    int j;
    MPI_Offset k;
    int err;
    double buf[MAX_NELS];	/* (void *) buffer */
    char *p;			/* (void *) pointer */
    double expect;
    double got;
    int nok = 0;      /* count of valid comparisons */

    err = ncmpi_open(comm, testfile, NC_NOWRITE, info, &ncid);
    IF (err != NC_NOERR) 
	error("ncmpi_open: %s", ncmpi_strerror(err));

    for (i = -1; i < numVars; i++) {
        for (j = 0; j < NATTS(i); j++) {
	    err = ncmpi_get_att(BAD_ID, i, ATT_NAME(i,j), buf);
	    IF (err != NC_EBADID) 
		error("bad ncid: status = %d", err);
            ELSE_NOK
	    err = ncmpi_get_att(ncid, BAD_VARID, ATT_NAME(i,j), buf);
	    IF (err != NC_ENOTVAR) 
		error("bad var id: status = %d", err);
            ELSE_NOK
	    err = ncmpi_get_att(ncid, i, "noSuch", buf);
	    IF (err != NC_ENOTATT) 
		error("Bad attribute name: status = %d", err);
            ELSE_NOK
	    err = ncmpi_get_att(ncid, i, ATT_NAME(i,j), buf);
	    IF (err != NC_NOERR) {
		error("%s", ncmpi_strerror(err));
	    } else {
		nok++;
		for (k = 0; k < ATT_LEN(i,j); k++) {
		    expect = hash(ATT_TYPE(i,j), -1, &k );
		    p = (char *) buf;
		    p += k * nctypelen(ATT_TYPE(i,j));
		    err = nc2dbl( ATT_TYPE(i,j), p, &got );
		    IF (err != NC_NOERR)
			error("error in nc2dbl");
		    if (inRange(expect,ATT_TYPE(i,j))) {
			IF (!equal2(got,expect,ATT_TYPE(i,j))) {
			    error("value read not that expected");
			    if (verbose) {
				error("\n");
				error("varid: %d, ", i);
				error("var_name: %s, ",
					i >= 0 ? var_name[i] : "Global");
				error("att_name: %s, ", ATT_NAME(i,j));
				error("element number: %d\n", k);
				error("expect: %-23.16e\n", expect);
				error("   got: %-23.16e", got);
			    }
			}
		    }
                }
	    }
	}
    }

    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
	error("ncmpi_close: %s", ncmpi_strerror(err));
    return nok;
}


int
test_ncmpi_inq_att(void)
{
    int ncid;
    int i;
    int j;
    int err;
    nc_type t;
    MPI_Offset n;
    int nok=0;

    err = ncmpi_open(comm, testfile, NC_NOWRITE, info, &ncid);
    IF (err != NC_NOERR) 
	error("ncmpi_open: %s", ncmpi_strerror(err));

    for (i = -1; i < numVars; i++) {
        for (j = 0; j < NATTS(i); j++) {
	    err = ncmpi_inq_att(BAD_ID, i, ATT_NAME(i,j), &t, &n);
	    IF (err != NC_EBADID) 
		error("bad ncid: status = %d", err);
            ELSE_NOK
	    err = ncmpi_inq_att(ncid, BAD_VARID, ATT_NAME(i,j), &t, &n);
	    IF (err != NC_ENOTVAR) 
		error("bad var id: status = %d", err);
            ELSE_NOK
	    err = ncmpi_inq_att(ncid, i, "noSuch", &t, &n);
	    IF (err != NC_ENOTATT) 
		error("Bad attribute name: status = %d", err);
            ELSE_NOK
	    err = ncmpi_inq_att(ncid, i, ATT_NAME(i,j), &t, &n);
	    IF (err != NC_NOERR) {
		error("%s", ncmpi_strerror(err));
	    } else {
		IF (t != ATT_TYPE(i,j))
		    error("type not that expected");
		else IF (n != ATT_LEN(i,j)) 
		    error("length not that expected");
                ELSE_NOK
	    }
	}
    }

    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
	error("ncmpi_close: %s", ncmpi_strerror(err));
    return nok;
}


int
test_ncmpi_inq_attlen(void)
{
    int ncid;
    int i;
    int j;
    int err;
    MPI_Offset len;
    int nok=0;

    err = ncmpi_open(comm, testfile, NC_NOWRITE, info, &ncid);
    IF (err != NC_NOERR)
        error("ncmpi_open: %s", ncmpi_strerror(err));

    for (i = -1; i < numVars; i++) {
	err = ncmpi_inq_attlen(ncid, i, "noSuch", &len);
	IF (err != NC_ENOTATT)
	    error("Bad attribute name: status = %d", err);
        ELSE_NOK
        for (j = 0; j < NATTS(i); j++) {
            err = ncmpi_inq_attlen(BAD_ID, i, ATT_NAME(i,j), &len);
            IF (err != NC_EBADID)
                error("bad ncid: status = %d", err);
            ELSE_NOK
            err = ncmpi_inq_attlen(ncid, BAD_VARID, ATT_NAME(i,j), &len);
            IF (err != NC_ENOTVAR)
                error("bad varid: status = %d", err);
            ELSE_NOK
            err = ncmpi_inq_attlen(ncid, i, ATT_NAME(i,j), &len);
            IF (err != NC_NOERR) {
                error("%s", ncmpi_strerror(err));
            } else {
		IF (len != ATT_LEN(i,j))
		    error("len not that expected");
                ELSE_NOK
            }
        }
    }

    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
	error("ncmpi_close: %s", ncmpi_strerror(err));
    return nok;
}


int
test_ncmpi_inq_atttype(void)
{
    int ncid;
    int i;
    int j;
    int err;
    nc_type datatype;
    int nok=0;

    err = ncmpi_open(comm, testfile, NC_NOWRITE, info, &ncid);
    IF (err != NC_NOERR)
        error("ncmpi_open: %s", ncmpi_strerror(err));

    for (i = -1; i < numVars; i++) {
	err = ncmpi_inq_atttype(ncid, i, "noSuch", &datatype);
	IF (err != NC_ENOTATT)
	    error("Bad attribute name: status = %d", err);
        ELSE_NOK
        for (j = 0; j < NATTS(i); j++) {
            err = ncmpi_inq_atttype(BAD_ID, i, ATT_NAME(i,j), &datatype);
            IF (err != NC_EBADID)
                error("bad ncid: status = %d", err);
            ELSE_NOK
            err = ncmpi_inq_atttype(ncid, BAD_VARID, ATT_NAME(i,j), &datatype);
            IF (err != NC_ENOTVAR)
                error("bad varid: status = %d", err);
            ELSE_NOK
            err = ncmpi_inq_atttype(ncid, i, ATT_NAME(i,j), &datatype);
            IF (err != NC_NOERR) {
                error("%s", ncmpi_strerror(err));
            } else {
		IF (datatype != ATT_TYPE(i,j))
		    error("type not that expected");
                ELSE_NOK
            }
        }
    }

    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
	error("ncmpi_close: %s", ncmpi_strerror(err));
    return nok;
}


int
test_ncmpi_inq_attname(void)
{
    int ncid;
    int i;
    int j;
    int err;
    char name[NC_MAX_NAME];
    int nok=0;

    err = ncmpi_open(comm, testfile, NC_NOWRITE, info, &ncid);
    IF (err != NC_NOERR)
        error("ncmpi_open: %s", ncmpi_strerror(err));

    for (i = -1; i < numVars; i++) {
	err = ncmpi_inq_attname(ncid, i, BAD_ATTNUM, name);
	IF (err != NC_ENOTATT)
	    error("Bad attribute number: status = %d", err);
        ELSE_NOK
	err = ncmpi_inq_attname(ncid, i, NATTS(i), name);
	IF (err != NC_ENOTATT)
	    error("Bad attribute number: status = %d", err);
        ELSE_NOK
        for (j = 0; j < NATTS(i); j++) {
            err = ncmpi_inq_attname(BAD_ID, i, j, name);
            IF (err != NC_EBADID)
                error("bad ncid: status = %d", err);
            ELSE_NOK
            err = ncmpi_inq_attname(ncid, BAD_VARID, j, name);
            IF (err != NC_ENOTVAR)
                error("bad var id: status = %d", err);
            ELSE_NOK
            err = ncmpi_inq_attname(ncid, i, j, name);
            IF (err != NC_NOERR) {
                error("%s", ncmpi_strerror(err));
            } else {
		IF (strcmp(ATT_NAME(i,j), name) != 0)
		    error("name not that expected");
                ELSE_NOK
            }
        }
    }

    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
	error("ncmpi_close: %s", ncmpi_strerror(err));
    return nok;
}


int
test_ncmpi_inq_attid(void)
{
    int ncid;
    int i;
    int j;
    int err;
    int attnum;
    int nok=0;

    err = ncmpi_open(comm, testfile, NC_NOWRITE, info, &ncid);
    IF (err != NC_NOERR)
        error("ncmpi_open: %s", ncmpi_strerror(err));

    for (i = -1; i < numVars; i++) {
	err = ncmpi_inq_attid(ncid, i, "noSuch", &attnum);
	IF (err != NC_ENOTATT)
	    error("Bad attribute name: status = %d", err);
        ELSE_NOK
        for (j = 0; j < NATTS(i); j++) {
            err = ncmpi_inq_attid(BAD_ID, i, ATT_NAME(i,j), &attnum);
            IF (err != NC_EBADID)
                error("bad ncid: status = %d", err);
            ELSE_NOK
            err = ncmpi_inq_attid(ncid, BAD_VARID, ATT_NAME(i,j), &attnum);
            IF (err != NC_ENOTVAR)
                error("bad varid: status = %d", err);
            ELSE_NOK
            err = ncmpi_inq_attid(ncid, i, ATT_NAME(i,j), &attnum);
            IF (err != NC_NOERR) {
                error("%s", ncmpi_strerror(err));
            } else {
		IF (attnum != j)
		    error("attnum not that expected");
                ELSE_NOK
            }
        }
    }

    err = ncmpi_close(ncid);
    IF (err != NC_NOERR)
	error("ncmpi_close: %s", ncmpi_strerror(err));
    return nok;
}


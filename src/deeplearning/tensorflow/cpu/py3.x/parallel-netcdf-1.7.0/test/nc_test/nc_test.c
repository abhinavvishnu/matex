/*
 *  Copyright (C) 2003, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 *
 *  $Id: nc_test.c 2302 2016-01-10 20:33:45Z wkliao $
 */

#include <unistd.h>

int cdf_format;  /* 1: CDF-1, 2: CDF-2 5: CDF-5 */
int extra_flags; /* if using CDF-2 format, will be set to NC_64BIT_OFFSET
                    if using CDF-5 format, will be set to NC_64BIT_DATA */
int numGatts;  /* number of global attributes */
int numVars;   /* number of variables */
int numTypes;  /* number of netCDF data types to test */

#include "tests.h"

/*
 * Test driver for netCDF-3 interface.  This program performs tests against
 * the netCDF-3 specification for all user-level functions in an
 * implementation of the netCDF library.
 *
 * Unless invoked with "-r" (read_only) option, must be invoked from a
 * directory in which the invoker has write permission.
 *
 * Files:
 * The read-only tests read files:
 *     test.nc (see below)
 *     test_get.c (used merely as an example of a non-netCDF file)
 * 
 * The write tests 
 *     read test.nc (see below) 
 *     write scratch.nc (deleted after each test)
 * 
 * The file test.nc is created by running nc_test with the -c (create) option.
 * It is described by the following global variables.
 */


/* 
 * global variables (defined by function init_gvars) describing file test.nc
 */
char dim_name[NDIMS][3];
MPI_Offset dim_len[NDIMS];
char var_name[NVARS][2+MAX_RANK];
nc_type var_type[NVARS];
size_t var_rank[NVARS];
int var_dimid[NVARS][MAX_RANK];
MPI_Offset var_shape[NVARS][MAX_RANK];
size_t var_nels[NVARS];
size_t var_natts[NVARS];
char att_name[NVARS][MAX_NATTS][2];
char gatt_name[NGATTS][3];
nc_type att_type[NVARS][NGATTS];
nc_type gatt_type[NGATTS];
size_t att_len[NVARS][MAX_NATTS];
size_t gatt_len[NGATTS];

/* 
 * command-line options
 */
static int  create_file;	/* if 1, create file test.nc */
int  read_only;		/* if 1, don't try to change files */
int  verbose;		/* if 1, print details of tests */
int  max_nmpt;		/* max. number of messages per test */

/* 
 * Misc. global variables
 */
int  nfails;		/* number of failures in specific test */
char testfile[128];
char scratch[128];
MPI_Comm comm = MPI_COMM_WORLD; /* mpi communicator for parallel-netcdf */
MPI_Info info;

static void
usage(char *progname)
{
    error("%s [-c | -hrv -n <MAX_NMPT>]\n", progname);
    error("   [-h] Print help\n" );
    error("   [-c] Create file test.nc (Do not do tests)\n" );
    error("   [-r] Just do read-only tests\n" );
    error("   [-v] Verbose mode\n" );
    error("   [-2] (with -c) create file with CDF-2 format\n" );
    error("   [-n <MAX_NMPT>] max. number of messages per test (Default: 8)\n");
    error("   [-d directory] directory for storing input/output files\n");
}

#define NC_TEST(func) {                                                  \
    char func_name[64];                                                  \
    int noks;                                                            \
    nfails = 0;                                                          \
    sprintf(func_name, "test_%s",#func);                                 \
    noks = test_ ## func();                                              \
    nfailsTotal += nfails;                                               \
    if (verbose && nfails == 0) {                                        \
        if (verbose) printf( "*** Testing %-30s ... ",func_name);        \
        if (noks > 0)                                                    \
            printf("%4d good comparisons. ok\n", noks);                  \
        else                                                             \
            printf("\n");                                                \
    }                                                                    \
    else if (nfails > 0) {                                               \
        if (verbose) print( "*** Testing %-30s ... ",func_name);         \
        print("\n\t### %d FAILURES TESTING %s! Stop ... ###\n",          \
              nfails,func_name);                                         \
        goto fn_exit;                                                    \
    }                                                                    \
}

#if 1		/* both CRAY MPP and OSF/1 Alpha systems need this */
#include <signal.h>
#endif /* T90 */

int
main(int argc, char *argv[])
{
    extern char *optarg;
    int c;
    int  nfailsTotal = 0;        /* total number of failures */

#if 1		/* both CRAY MPP and OSF/1 Alpha systems need this */
	/*
	 * Some of the extreme test assignments in this program trigger
         * floating point exceptions on CRAY T90
	 */
	(void) signal(SIGFPE, SIG_IGN);
#endif

    MPI_Init(&argc, &argv);

    cdf_format = 1; 	/* 1: CDF-1, 2: CDF-2 5: CDF-5 */
    extra_flags = 0;    /* NC_64BIT_OFFSET or NC_64BIT_DATA */
    create_file = 0;            /* file test.nc will normally already exist */
    read_only = 0;               /* assume may write in test dir as default */
    verbose = 0;
    max_nmpt = 8;
    strcpy(testfile, "test.nc");    /* read-only testfile */
    strcpy(scratch, "scratch.nc");  /* writable scratch file */

    while ((c = getopt(argc, argv, "c25hrvn:d:")) != EOF)
      switch(c) {
	case 'c':		/* Create file test.nc */
	  create_file = 1;
	  break;
	case 'r':		/* just perform read-only tests */
	  read_only = 1;
	  break;
	case 'v':		/* verbose mode */
	  verbose = 1;
	  break;
	case 'n':		/* verbose mode */
	  max_nmpt = atoi(optarg);
	  break;
	case '2':
	  cdf_format = 2;
	  extra_flags = NC_64BIT_OFFSET;
	  break;
	case '5':
	  cdf_format = 5;
	  extra_flags = NC_64BIT_DATA;
	case 'd':
          sprintf(testfile, "%s/test.nc", optarg);
          sprintf(scratch, "%s/scratch.nc", optarg);
	  break;
	case 'h':
	case '?':
	  usage(argv[0]);
	  return 1;
      }

    MPI_Info_create(&info);
    /* MPI_Info_set(info, "romio_pvfs2_posix_write", "enable"); */

    numGatts = 6;
    numVars  = 136;
    numTypes = 6;
    if (cdf_format == 5) {
        numGatts = NGATTS;
        numVars  = NVARS;
        numTypes = NTYPES;
    }

    /* Initialize global variables defining test file */
    init_gvars();

    if ( create_file ) {
	write_file(testfile);
        MPI_Info_free(&info);
	MPI_Finalize();
	return nfailsTotal > 0;
    }

    /* delete any existing scratch netCDF file */
    if ( ! read_only ) {
        if (access(scratch, F_OK) == 0)
            unlink(scratch);
    }

    char cmd_str[256];
    sprintf(cmd_str, "*** TESTING C   %s for format CDF-%d ", argv[0], cdf_format);
    printf("%-66s ------ ",cmd_str);

    /* Test read-only functions, using pregenerated test-file */
    NC_TEST(ncmpi_strerror);
    NC_TEST(ncmpi_open);
    NC_TEST(ncmpi_close);
    NC_TEST(ncmpi_inq);
    NC_TEST(ncmpi_inq_dimid);
    NC_TEST(ncmpi_inq_dim);
    NC_TEST(ncmpi_inq_dimlen);
    NC_TEST(ncmpi_inq_dimname);
    NC_TEST(ncmpi_inq_varid);
    NC_TEST(ncmpi_inq_var);
    NC_TEST(ncmpi_inq_natts);
    NC_TEST(ncmpi_inq_ndims);
    NC_TEST(ncmpi_inq_nvars);
    NC_TEST(ncmpi_inq_unlimdim);
    NC_TEST(ncmpi_inq_vardimid);
    NC_TEST(ncmpi_inq_varname);
    NC_TEST(ncmpi_inq_varnatts);
    NC_TEST(ncmpi_inq_varndims);
    NC_TEST(ncmpi_inq_vartype);
    NC_TEST(ncmpi_get_var_text);
    NC_TEST(ncmpi_get_var_schar);
    NC_TEST(ncmpi_get_var_short);
    NC_TEST(ncmpi_get_var_int);
    NC_TEST(ncmpi_get_var_long); 
    NC_TEST(ncmpi_get_var_float);
    NC_TEST(ncmpi_get_var_double);
    NC_TEST(ncmpi_get_var_uchar); 
    NC_TEST(ncmpi_get_var_ushort);
    NC_TEST(ncmpi_get_var_uint);
    NC_TEST(ncmpi_get_var_longlong);
    NC_TEST(ncmpi_get_var_ulonglong);
    NC_TEST(ncmpi_get_var1_text);
    NC_TEST(ncmpi_get_var1_schar);
    NC_TEST(ncmpi_get_var1_short);
    NC_TEST(ncmpi_get_var1_int);
    NC_TEST(ncmpi_get_var1_long);
    NC_TEST(ncmpi_get_var1_float);
    NC_TEST(ncmpi_get_var1_double);
    NC_TEST(ncmpi_get_var1_uchar); 
    NC_TEST(ncmpi_get_var1_ushort);
    NC_TEST(ncmpi_get_var1_uint);
    NC_TEST(ncmpi_get_var1_longlong);
    NC_TEST(ncmpi_get_var1_ulonglong);
    NC_TEST(ncmpi_get_var1);
    NC_TEST(ncmpi_get_vara_text);
    NC_TEST(ncmpi_get_vara_schar);
    NC_TEST(ncmpi_get_vara_short);
    NC_TEST(ncmpi_get_vara_int);
    NC_TEST(ncmpi_get_vara_long); 
    NC_TEST(ncmpi_get_vara_float);
    NC_TEST(ncmpi_get_vara_double);
    NC_TEST(ncmpi_get_vara_uchar);
    NC_TEST(ncmpi_get_vara_ushort);
    NC_TEST(ncmpi_get_vara_uint);
    NC_TEST(ncmpi_get_vara_longlong);
    NC_TEST(ncmpi_get_vara_ulonglong);
    NC_TEST(ncmpi_get_vara);
    NC_TEST(ncmpi_get_vars_text);
    NC_TEST(ncmpi_get_vars_schar);
    NC_TEST(ncmpi_get_vars_short);
    NC_TEST(ncmpi_get_vars_int);
    NC_TEST(ncmpi_get_vars_long); 
    NC_TEST(ncmpi_get_vars_float);
    NC_TEST(ncmpi_get_vars_double);
    NC_TEST(ncmpi_get_vars_uchar);
    NC_TEST(ncmpi_get_vars_ushort);
    NC_TEST(ncmpi_get_vars_uint);
    NC_TEST(ncmpi_get_vars_longlong);
    NC_TEST(ncmpi_get_vars_ulonglong);
    NC_TEST(ncmpi_get_vars);
    NC_TEST(ncmpi_get_varm_text);
    NC_TEST(ncmpi_get_varm_schar);
    NC_TEST(ncmpi_get_varm_short);
    NC_TEST(ncmpi_get_varm_int);
    NC_TEST(ncmpi_get_varm_long);
    NC_TEST(ncmpi_get_varm_float);
    NC_TEST(ncmpi_get_varm_double);
    NC_TEST(ncmpi_get_varm_uchar);
    NC_TEST(ncmpi_get_varm_ushort);
    NC_TEST(ncmpi_get_varm_uint);
    NC_TEST(ncmpi_get_varm_longlong);
    NC_TEST(ncmpi_get_varm_ulonglong);
    NC_TEST(ncmpi_get_varm);
    NC_TEST(ncmpi_get_att_text);
    NC_TEST(ncmpi_get_att_schar);
    NC_TEST(ncmpi_get_att_short);
    NC_TEST(ncmpi_get_att_int);
    NC_TEST(ncmpi_get_att_long);
    NC_TEST(ncmpi_get_att_float);
    NC_TEST(ncmpi_get_att_double);
    NC_TEST(ncmpi_get_att_uchar);
    NC_TEST(ncmpi_get_att_ushort);
    NC_TEST(ncmpi_get_att_uint);
    NC_TEST(ncmpi_get_att_longlong);
    NC_TEST(ncmpi_get_att_ulonglong);
    NC_TEST(ncmpi_get_att);
    NC_TEST(ncmpi_inq_att);
    NC_TEST(ncmpi_inq_attname);
    NC_TEST(ncmpi_inq_attid);
    NC_TEST(ncmpi_inq_attlen);
    NC_TEST(ncmpi_inq_atttype);

    /* nonblocking I/O */
    NC_TEST(ncmpi_iget_var_text);
    NC_TEST(ncmpi_iget_var_schar);
    NC_TEST(ncmpi_iget_var_short);
    NC_TEST(ncmpi_iget_var_int);
    NC_TEST(ncmpi_iget_var_long); 
    NC_TEST(ncmpi_iget_var_float);
    NC_TEST(ncmpi_iget_var_double);
    NC_TEST(ncmpi_iget_var_uchar); 
    NC_TEST(ncmpi_iget_var_ushort);
    NC_TEST(ncmpi_iget_var_uint);
    NC_TEST(ncmpi_iget_var_longlong);
    NC_TEST(ncmpi_iget_var_ulonglong);
    NC_TEST(ncmpi_iget_var);
    NC_TEST(ncmpi_iget_var1_text);
    NC_TEST(ncmpi_iget_var1_schar);
    NC_TEST(ncmpi_iget_var1_short);
    NC_TEST(ncmpi_iget_var1_int);
    NC_TEST(ncmpi_iget_var1_long);
    NC_TEST(ncmpi_iget_var1_float);
    NC_TEST(ncmpi_iget_var1_double);
    NC_TEST(ncmpi_iget_var1_uchar); 
    NC_TEST(ncmpi_iget_var1_ushort);
    NC_TEST(ncmpi_iget_var1_uint);
    NC_TEST(ncmpi_iget_var1_longlong);
    NC_TEST(ncmpi_iget_var1_ulonglong);
    NC_TEST(ncmpi_iget_var1);
    NC_TEST(ncmpi_iget_vara_text);
    NC_TEST(ncmpi_iget_vara_schar);
    NC_TEST(ncmpi_iget_vara_short);
    NC_TEST(ncmpi_iget_vara_int);
    NC_TEST(ncmpi_iget_vara_long); 
    NC_TEST(ncmpi_iget_vara_float);
    NC_TEST(ncmpi_iget_vara_double);
    NC_TEST(ncmpi_iget_vara_uchar);
    NC_TEST(ncmpi_iget_vara_ushort);
    NC_TEST(ncmpi_iget_vara_uint);
    NC_TEST(ncmpi_iget_vara_longlong);
    NC_TEST(ncmpi_iget_vara_ulonglong);
    NC_TEST(ncmpi_iget_vara);
    NC_TEST(ncmpi_iget_vars_text);
    NC_TEST(ncmpi_iget_vars_schar);
    NC_TEST(ncmpi_iget_vars_short);
    NC_TEST(ncmpi_iget_vars_int);
    NC_TEST(ncmpi_iget_vars_long); 
    NC_TEST(ncmpi_iget_vars_float);
    NC_TEST(ncmpi_iget_vars_double);
    NC_TEST(ncmpi_iget_vars_uchar);
    NC_TEST(ncmpi_iget_vars_ushort);
    NC_TEST(ncmpi_iget_vars_uint);
    NC_TEST(ncmpi_iget_vars_longlong);
    NC_TEST(ncmpi_iget_vars_ulonglong);
    NC_TEST(ncmpi_iget_vars);
    NC_TEST(ncmpi_iget_varm_text);
    NC_TEST(ncmpi_iget_varm_schar);
    NC_TEST(ncmpi_iget_varm_short);
    NC_TEST(ncmpi_iget_varm_int);
    NC_TEST(ncmpi_iget_varm_long);
    NC_TEST(ncmpi_iget_varm_float);
    NC_TEST(ncmpi_iget_varm_double);
    NC_TEST(ncmpi_iget_varm_uchar);
    NC_TEST(ncmpi_iget_varm_ushort);
    NC_TEST(ncmpi_iget_varm_uint);
    NC_TEST(ncmpi_iget_varm_longlong);
    NC_TEST(ncmpi_iget_varm_ulonglong);
    NC_TEST(ncmpi_iget_varm);

	/* Test write functions */
    if (! read_only) {
	NC_TEST(ncmpi_create);
	NC_TEST(ncmpi_redef);
	/* NC_TEST(ncmpi_enddef);  redundant, as it calls test_ncmpi_redef() */
	NC_TEST(ncmpi_sync);
	NC_TEST(ncmpi_abort);
	NC_TEST(ncmpi_def_dim);
	NC_TEST(ncmpi_rename_dim);
	NC_TEST(ncmpi_def_var);
	NC_TEST(ncmpi_put_var_text);
	NC_TEST(ncmpi_put_var_schar);
	NC_TEST(ncmpi_put_var_short);
	NC_TEST(ncmpi_put_var_int);
	NC_TEST(ncmpi_put_var_long); 
	NC_TEST(ncmpi_put_var_float);
	NC_TEST(ncmpi_put_var_double);
	NC_TEST(ncmpi_put_var_uchar);
	NC_TEST(ncmpi_put_var_ushort);
	NC_TEST(ncmpi_put_var_uint);
	NC_TEST(ncmpi_put_var_longlong);
	NC_TEST(ncmpi_put_var_ulonglong);
	NC_TEST(ncmpi_put_var1_text);
	NC_TEST(ncmpi_put_var1_schar);
	NC_TEST(ncmpi_put_var1_short);
	NC_TEST(ncmpi_put_var1_int);
	NC_TEST(ncmpi_put_var1_long); 
	NC_TEST(ncmpi_put_var1_float);
	NC_TEST(ncmpi_put_var1_double);
	NC_TEST(ncmpi_put_var1_uchar); 
	NC_TEST(ncmpi_put_var1_ushort);
	NC_TEST(ncmpi_put_var1_uint);
	NC_TEST(ncmpi_put_var1_longlong);
	NC_TEST(ncmpi_put_var1_ulonglong);
	NC_TEST(ncmpi_put_var1);
	NC_TEST(ncmpi_put_vara_text);
	NC_TEST(ncmpi_put_vara_schar); 
	NC_TEST(ncmpi_put_vara_short);
	NC_TEST(ncmpi_put_vara_int);
	NC_TEST(ncmpi_put_vara_long);
	NC_TEST(ncmpi_put_vara_float);
	NC_TEST(ncmpi_put_vara_double);
	NC_TEST(ncmpi_put_vara_uchar);
	NC_TEST(ncmpi_put_vara_ushort);
	NC_TEST(ncmpi_put_vara_uint);
	NC_TEST(ncmpi_put_vara_longlong);
	NC_TEST(ncmpi_put_vara_ulonglong);
	NC_TEST(ncmpi_put_vara);
	NC_TEST(ncmpi_put_vars_text);
	NC_TEST(ncmpi_put_vars_schar);
	NC_TEST(ncmpi_put_vars_short);
	NC_TEST(ncmpi_put_vars_int);
	NC_TEST(ncmpi_put_vars_long); 
	NC_TEST(ncmpi_put_vars_float);
	NC_TEST(ncmpi_put_vars_double);
	NC_TEST(ncmpi_put_vars_uchar);
	NC_TEST(ncmpi_put_vars_ushort);
	NC_TEST(ncmpi_put_vars_uint);
	NC_TEST(ncmpi_put_vars_longlong);
	NC_TEST(ncmpi_put_vars_ulonglong);
	NC_TEST(ncmpi_put_vars);
	NC_TEST(ncmpi_put_varm_text);
	NC_TEST(ncmpi_put_varm_schar);
	NC_TEST(ncmpi_put_varm_short);
	NC_TEST(ncmpi_put_varm_int);
	NC_TEST(ncmpi_put_varm_long);
	NC_TEST(ncmpi_put_varm_float);
	NC_TEST(ncmpi_put_varm_double);
	NC_TEST(ncmpi_put_varm_uchar);
	NC_TEST(ncmpi_put_varm_ushort);
	NC_TEST(ncmpi_put_varm_uint);
	NC_TEST(ncmpi_put_varm_longlong);
	NC_TEST(ncmpi_put_varm_ulonglong);
	NC_TEST(ncmpi_put_varm);
	NC_TEST(ncmpi_rename_var);
	NC_TEST(ncmpi_put_att_text);
	NC_TEST(ncmpi_put_att_schar);
	NC_TEST(ncmpi_put_att_short);
	NC_TEST(ncmpi_put_att_int);
	NC_TEST(ncmpi_put_att_long);
	NC_TEST(ncmpi_put_att_float);
	NC_TEST(ncmpi_put_att_double);
	NC_TEST(ncmpi_put_att_uchar);
	NC_TEST(ncmpi_put_att_ushort);
	NC_TEST(ncmpi_put_att_uint);
	NC_TEST(ncmpi_put_att_longlong);
	NC_TEST(ncmpi_put_att_ulonglong);
	NC_TEST(ncmpi_put_att);
	NC_TEST(ncmpi_copy_att);
	NC_TEST(ncmpi_rename_att);
	NC_TEST(ncmpi_del_att);
	NC_TEST(ncmpi_set_fill);
	NC_TEST(ncmpi_delete);

        /* test nonblocking APIs */
	NC_TEST(ncmpi_iput_var_text);
	NC_TEST(ncmpi_iput_var_schar);
	NC_TEST(ncmpi_iput_var_short);
	NC_TEST(ncmpi_iput_var_int);
	NC_TEST(ncmpi_iput_var_long); 
	NC_TEST(ncmpi_iput_var_float);
	NC_TEST(ncmpi_iput_var_double);
	NC_TEST(ncmpi_iput_var_uchar);
	NC_TEST(ncmpi_iput_var_ushort);
	NC_TEST(ncmpi_iput_var_uint);
	NC_TEST(ncmpi_iput_var_longlong);
	NC_TEST(ncmpi_iput_var_ulonglong);
	NC_TEST(ncmpi_iput_var);
	NC_TEST(ncmpi_iput_var1_text);
	NC_TEST(ncmpi_iput_var1_schar);
	NC_TEST(ncmpi_iput_var1_short);
	NC_TEST(ncmpi_iput_var1_int);
	NC_TEST(ncmpi_iput_var1_long); 
	NC_TEST(ncmpi_iput_var1_float);
	NC_TEST(ncmpi_iput_var1_double);
	NC_TEST(ncmpi_iput_var1_uchar); 
	NC_TEST(ncmpi_iput_var1_ushort);
	NC_TEST(ncmpi_iput_var1_uint);
	NC_TEST(ncmpi_iput_var1_longlong);
	NC_TEST(ncmpi_iput_var1_ulonglong);
	NC_TEST(ncmpi_iput_var1);
	NC_TEST(ncmpi_iput_vara_text);
	NC_TEST(ncmpi_iput_vara_schar); 
	NC_TEST(ncmpi_iput_vara_short);
	NC_TEST(ncmpi_iput_vara_int);
	NC_TEST(ncmpi_iput_vara_long);
	NC_TEST(ncmpi_iput_vara_float);
	NC_TEST(ncmpi_iput_vara_double);
	NC_TEST(ncmpi_iput_vara_uchar);
	NC_TEST(ncmpi_iput_vara_ushort);
	NC_TEST(ncmpi_iput_vara_uint);
	NC_TEST(ncmpi_iput_vara_longlong);
	NC_TEST(ncmpi_iput_vara_ulonglong);
	NC_TEST(ncmpi_iput_vara);
	NC_TEST(ncmpi_iput_vars_text);
	NC_TEST(ncmpi_iput_vars_schar);
	NC_TEST(ncmpi_iput_vars_short);
	NC_TEST(ncmpi_iput_vars_int);
	NC_TEST(ncmpi_iput_vars_long); 
	NC_TEST(ncmpi_iput_vars_float);
	NC_TEST(ncmpi_iput_vars_double);
	NC_TEST(ncmpi_iput_vars_uchar);
	NC_TEST(ncmpi_iput_vars_ushort);
	NC_TEST(ncmpi_iput_vars_uint);
	NC_TEST(ncmpi_iput_vars_longlong);
	NC_TEST(ncmpi_iput_vars_ulonglong);
	NC_TEST(ncmpi_iput_vars);
	NC_TEST(ncmpi_iput_varm_text);
	NC_TEST(ncmpi_iput_varm_schar);
	NC_TEST(ncmpi_iput_varm_short);
	NC_TEST(ncmpi_iput_varm_int);
	NC_TEST(ncmpi_iput_varm_long);
	NC_TEST(ncmpi_iput_varm_float);
	NC_TEST(ncmpi_iput_varm_double);
	NC_TEST(ncmpi_iput_varm_uchar);
	NC_TEST(ncmpi_iput_varm_ushort);
	NC_TEST(ncmpi_iput_varm_uint);
	NC_TEST(ncmpi_iput_varm_longlong);
	NC_TEST(ncmpi_iput_varm_ulonglong);
	NC_TEST(ncmpi_iput_varm);
    }

fn_exit:
    MPI_Info_free(&info);

    if (nfailsTotal == 0)  {
        printf(PASS_STR);
    }
    else {
        print("\n%s: expects 0 failures ... ",argv[0]);
        printf(FAIL_STR, nfailsTotal);
    }
    MPI_Finalize();
    return nfailsTotal > 0;
}

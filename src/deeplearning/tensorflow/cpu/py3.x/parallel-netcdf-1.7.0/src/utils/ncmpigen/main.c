/*********************************************************************
 *   Copyright 1993, UCAR/Unidata
 *   See netcdf/COPYRIGHT file for copying and redistribution conditions.
 *   $Header$
 *********************************************************************/

#if HAVE_CONFIG_H
# include <ncconfig.h>
#endif

#include <stdio.h>		/* has getopt() under VMS */
#include <string.h>

#ifdef __hpux
#include <locale.h>  /* setlocale() */
#endif
    
#include <pnetcdf.h>

#include <unistd.h>

#include "generic.h"
#include "ncmpigen.h"
#include "genlib.h"

extern int	ncmpiparse(void);

const char *progname;			/* for error messages */
const char *cdlname;

int c_flag;
int fortran_flag;
int netcdf_flag;
int giantfile_flag;
int giantvar_flag;
int nofill_flag;
char *netcdf_name = NULL;	/* name of output netCDF file to write */

extern FILE *ncmpiin;

static const char* ubasename ( const char* av0 );
static void usage ( void );
int main ( int argc, char** argv );


/* strip off leading path */
static const char *
ubasename(
	const char *av0)
{
	const char *logident ;
#ifdef VMS
#define SEP	']'
#endif
#ifdef MSDOS
#define SEP	'\\'
#endif
#ifndef SEP
#define SEP	'/'
#endif
	if ((logident = strrchr(av0, SEP)) == NULL)
		logident = av0 ;
	else
	    logident++ ;
	return logident ;
}


static void usage(void)
{
    derror("Usage: %s [ -b ] [ -c ] [ -f ] [ -v version ] [ -x ] [ -o outfile]  [ file ... ]",
	   progname);
    derror("PnetCDF library version %s", ncmpi_inq_libvers());
}


int
main(int argc, char *argv[])
{
    extern int optind;
    extern int opterr;
    extern char *optarg;
    int c;
    int ret;
    FILE *fp;

    MPI_Init(&argc, &argv);

#ifdef __hpux
    setlocale(LC_CTYPE,"");
#endif
    
#ifdef MDEBUG
    malloc_debug(2) ;	/* helps find malloc/free errors on Sun */
#endif /* MDEBUG */

    opterr = 1;			/* print error message if bad option */
    progname = ubasename(argv[0]);
    cdlname = "-";

    c_flag = 0;
    fortran_flag = 0;
    netcdf_flag = 0;
    giantfile_flag = 0;
    giantvar_flag = 0;
    nofill_flag = 0;

#if 0
#if _CRAYMPP && 0
    /* initialize CRAY MPP parallel-I/O library */
    (void) par_io_init(32, 32);
#endif
#endif

    while ((c = getopt(argc, argv, "bcfl:no:v:x")) != EOF)
      switch(c) {
	case 'c':		/* for c output. old version of '-lc' */
	  c_flag = 1;
	  break;
	case 'f':		/* for fortran output. old version of '-lf' */
	  fortran_flag = 1;
	  break;
	case 'b':		/* for binary netcdf output, ".nc" extension */
	  netcdf_flag = 1;
	  break;
	case 'l':               /* specify language, instead of -c or -f */
	  {
               char *lang_name = (char *) emalloc(strlen(optarg)+1);
               if (! lang_name) {
                   derror ("%s: out of memory", progname);
                   return(1);
               }
               (void)strcpy(lang_name, optarg);
               if (strcmp(lang_name, "c") == 0 || strcmp(lang_name, "C") == 0) {
                   c_flag = 1;
               }
               else if (strcmp(lang_name, "f77") == 0 || 
                        strcmp(lang_name, "fortran77") == 0 ||
                        strcmp(lang_name, "Fortran77") == 0) {
                   fortran_flag = 1;
               } else {     /* Fortran90, Java, C++, Perl, Python, Ruby, ... */
                   derror("%s: output language %s not implemented", 
                          progname, lang_name);
                   return(1);
               }
           }
	  break;
	case 'n':		/* old version of -b, uses ".cdf" extension */
	  netcdf_flag = -1;
	  break;
	case 'o':		/* to explicitly specify output name */
	  netcdf_flag = 1;
	  netcdf_name = (char *) emalloc(strlen(optarg)+1);
	  if (! netcdf_name) {
	      derror ("%s: out of memory", progname);
	      return(1);
	  }
	  (void)strcpy(netcdf_name,optarg);
	  break;
	case 'x':     /* set nofill mode to speed up creation fo large files */
	  nofill_flag = 1;
	  break;
	case 'v':     /* for creating 64-bit offet files, specify version 2 */
	  {
		  char *version_name = (char *)emalloc(strlen(optarg)+1);
		  if (! version_name) {
			  derror ("%s: out of memory", progname);
			  return (1);
		  }
		  (void)strcpy(version_name, optarg);
		  /* the default version is version 1, with 32-bit offsets */
		  if (strcmp(version_name, "1") == 0 ||
				  strcmp(version_name, "classic") == 0) {
			  giantfile_flag = 0;
		  }
		  /* the 64-bit offset version (2) should only be used if
		   * actually needed */
		  else if (strcmp(version_name, "2") == 0 || 
				  strcmp(version_name, "64-bit-offset") == 0) {
			  giantfile_flag = 1;
		  } else if (strcmp(version_name, "5") == 0 ||
				  strcmp(version_name, 
					  "64-bit-variables") == 0) {
			  giantvar_flag = 1;
		  }
	  }
	  break;
	case '?':
	  usage();
	  return(8);
      }

    if (fortran_flag && c_flag) {
	derror("Only one of -c or -f may be specified");
	return(8);
    }
    if (fortran_flag) {
	    derror("Generating Fortran interface currently not supported yet");
            return(0);
    }

    argc -= optind;
    argv += optind;

    if (argc > 1) {
	derror ("%s: only one input file argument permitted",progname);
	return(6);
    }

    fp = stdin;
    if (argc > 0 && strcmp(argv[0], "-") != 0) {
	if ((fp = fopen(argv[0], "r")) == NULL) {
	    derror ("can't open file %s for reading: ", argv[0]);
	    perror("");
	    return(7);
	}
	cdlname = argv[0];
    }
    ncmpiin = fp;
    ret = ncmpiparse(); 
    MPI_Finalize();
    return ret;
}

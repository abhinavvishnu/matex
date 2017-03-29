/*
 * $Id: fortlib.c 1468 2013-10-26 16:53:18Z wkliao $
 *
 * This file contains support functions for FORTRAN code.  For example,
 * under HP-UX A.09.05, the U77 library doesn't contain the exit()
 * routine -- so we create one here.
 *
 * This file is heavily modified from nf_test/fortlib.c in the serial netcdf
 * source.
 */


#include <stdlib.h>
#include <limits.h>
#include <float.h>
#include <unistd.h>
#include <stdint.h>
#include <mpinetcdf_impl.h>
#include <mpifnetcdf.h>

extern FORTRAN_API void FORT_CALL ud_exit_(int *v1);
extern FORTRAN_API void FORT_CALL ud_abort_(int *v1);
extern FORTRAN_API double FORT_CALL ud_rand_(int *seed);
extern FORTRAN_API int FORT_CALL ud_shift_(int * value, int *amount);
extern FORTRAN_API void FORT_CALL nc_ignorefpe_(int *doit);
extern FORTRAN_API double FORT_CALL min_schar_(void);
extern FORTRAN_API double FORT_CALL min_short_(void);
extern FORTRAN_API double FORT_CALL min_int_(void);
extern FORTRAN_API double FORT_CALL min_int64_(void);
extern FORTRAN_API double FORT_CALL max_schar_(void);
extern FORTRAN_API double FORT_CALL max_short_(void);
extern FORTRAN_API double FORT_CALL max_int_(void);
extern FORTRAN_API double FORT_CALL max_int64_(void);
extern FORTRAN_API double FORT_CALL max_float_(void);
extern FORTRAN_API double FORT_CALL max_double_(void);


#ifdef F77_NAME_UPPER
#define ud_exit_ UD_EXIT
#elif defined(F77_NAME_LOWER_2USCORE)
#define ud_exit_  ud_exit__
#elif !defined(F77_NAME_LOWER_USCORE)
#define ud_exit_  ud_exit
/* Else leave name alone */
#endif

FORTRAN_API void FORT_CALL ud_exit_(int *v1) {
	if (*v1 == 0) {
		MPI_Finalize();
		exit(0);
	}
	MPI_Abort(MPI_COMM_WORLD, *v1);
	return;
}

#ifdef F77_NAME_UPPER
#define ud_abort_ UD_ABORT
#elif defined(F77_NAME_LOWER_2USCORE)
#define ud_abort_  ud_abort__
#elif !defined(F77_NAME_LOWER_USCORE)
#define ud_abort_  ud_abort
/* Else leave name alone */
#endif

FORTRAN_API void FORT_CALL ud_abort_(int *v1) {
	MPI_Abort(MPI_COMM_WORLD, *v1);
	return;
}

#ifdef F77_NAME_UPPER
#define ud_rand_ UD_RAND
#elif defined(F77_NAME_LOWER_2USCORE)
#define ud_rand_  ud_rand__
#elif !defined(F77_NAME_LOWER_USCORE)
#define ud_rand_  ud_rand
/* Else leave name alone */
#endif

/*
* Return a pseudo-random value between 0.0 and 1.0.
*
* We don't use RAND_MAX here because not all compilation
* environments define it (e.g. gcc(1) under SunOS 4.1.3).
*/
FORTRAN_API double FORT_CALL ud_rand_(int *seed) {
	if (*seed != 0)
		srand(*seed);
	return (rand() % 32768 )/ 32767.0;
}

#ifdef F77_NAME_UPPER
#define ud_shift_ UD_SHIFT
#elif defined(F77_NAME_LOWER_2USCORE)
#define ud_shift_  ud_shift__
#elif !defined(F77_NAME_LOWER_USCORE)
#define ud_shift_  ud_shift
/* Else leave name alone */
#endif

FORTRAN_API int FORT_CALL ud_shift_(int * value, int *amount) {
	if(*amount < 0)
		*value >>= -*amount;
	else if (*amount > 0)
		*value <<= *amount;
	return *value;
}

#ifdef F77_NAME_UPPER
#define nc_ignorefpe_ NC_IGNOREFPE
#elif defined(F77_NAME_LOWER_2USCORE)
#define nc_ignorefpe_  nc_ignorefpe__
#elif !defined(F77_NAME_LOWER_USCORE)
#define nc_ignorefpe_  nc_ignorefpe
/* Else leave name alone */
#endif
#include <signal.h>
FORTRAN_API void FORT_CALL nc_ignorefpe_(int *doit)
{
	if(doit)
		(void) signal(SIGFPE, SIG_IGN);
}

#ifdef F77_NAME_UPPER
#define min_schar_ MIN_SCHAR
#elif defined(F77_NAME_LOWER_2USCORE)
#define min_schar_  min_schar__
#elif !defined(F77_NAME_LOWER_USCORE)
#define min_schar_  min_schar
/* Else leave name alone */
#endif
FORTRAN_API double FORT_CALL min_schar_(void) {
	return SCHAR_MIN;
}

#ifdef F77_NAME_UPPER
#define min_short_ MIN_SHORT
#elif defined(F77_NAME_LOWER_2USCORE)
#define min_short_  min_short__
#elif !defined(F77_NAME_LOWER_USCORE)
#define min_short_  min_short
/* Else leave name alone */
#endif
FORTRAN_API double FORT_CALL min_short_(void) {
	return SHRT_MIN;
}

#ifdef F77_NAME_UPPER
#define min_int_ MIN_INT
#elif defined(F77_NAME_LOWER_2USCORE)
#define min_int_  min_int__
#elif !defined(F77_NAME_LOWER_USCORE)
#define min_int_  min_int
/* Else leave name alone */
#endif
FORTRAN_API double FORT_CALL min_int_(void) {
	return INT_MIN;
}

#ifdef F77_NAME_UPPER
#define min_int64_ MIN_INT64
#elif defined(F77_NAME_LOWER_2USCORE)
#define min_int64_  min_int64__
#elif !defined(F77_NAME_LOWER_USCORE)
#define min_int64_  min_int64
/* Else leave name alone */
#endif
FORTRAN_API double FORT_CALL min_int64_(void) {
	return INT64_MIN;
}

#ifdef F77_NAME_UPPER
#define max_schar_ MAX_SCHAR
#elif defined(F77_NAME_LOWER_2USCORE)
#define max_schar_  max_schar__
#elif !defined(F77_NAME_LOWER_USCORE)
#define max_schar_  max_schar
/* Else leave name alone */
#endif
FORTRAN_API double FORT_CALL max_schar_(void) {
	return SCHAR_MAX;
}

#ifdef F77_NAME_UPPER
#define max_short_ MAX_SHORT
#elif defined(F77_NAME_LOWER_2USCORE)
#define max_short_  max_short__
#elif !defined(F77_NAME_LOWER_USCORE)
#define max_short_  max_short
/* Else leave name alone */
#endif
FORTRAN_API double FORT_CALL max_short_(void) {
	return SHRT_MAX;
}

#ifdef F77_NAME_UPPER
#define max_int_ MAX_INT
#elif defined(F77_NAME_LOWER_2USCORE)
#define max_int_  max_int__
#elif !defined(F77_NAME_LOWER_USCORE)
#define max_int_  max_int
/* Else leave name alone */
#endif
FORTRAN_API double FORT_CALL max_int_(void) {
	return INT_MAX;
}

#ifdef F77_NAME_UPPER
#define max_int64_ MAX_INT64
#elif defined(F77_NAME_LOWER_2USCORE)
#define max_int64_  max_int64__
#elif !defined(F77_NAME_LOWER_USCORE)
#define max_int64_  max_int64
/* Else leave name alone */
#endif
FORTRAN_API double FORT_CALL max_int64_(void) {
	return INT64_MAX;
}

#ifdef F77_NAME_UPPER
#define max_float_ MAX_FLOAT
#elif defined(F77_NAME_LOWER_2USCORE)
#define max_float_  max_float__
#elif !defined(F77_NAME_LOWER_USCORE)
#define max_float_  max_float
/* Else leave name alone */
#endif
FORTRAN_API double FORT_CALL max_float_(void) {
	return FLT_MAX;
}

#ifdef F77_NAME_UPPER
#define max_double_ MAX_DOUBLE
#elif defined(F77_NAME_LOWER_2USCORE)
#define max_double_  max_double__
#elif !defined(F77_NAME_LOWER_USCORE)
#define max_double_  max_double
/* Else leave name alone */
#endif
FORTRAN_API double FORT_CALL max_double_(void) {
	return DBL_MAX;
}

#if 0 /* this is implemented in library src now */

#ifdef F77_NAME_UPPER
#define nfmpi_issyserr_ NFMPI_ISSYSERR
#elif defined(F77_NAME_LOWER_2USCORE)
#define nfmpi_issyserr_  nfmpi_issyserr__
#elif !defined(F77_NAME_LOWER_USCORE)
#define nfmpi_issyserr_  nfmpi_issyserr
/* Else leave name alone */
#endif

FORTRAN_API int FORT_CALL nfmpi_issyserr_(int * A1) {
	if (*A1 > 0)
		return 1;
	else 
		return 0;
}


#ifdef F77_NAME_UPPER
#define nfmpi_delete_ NFMPI_DELETE
#elif defined(F77_NAME_LOWER_2USCORE)
#define nfmpi_delete_  nfmpi_delete__
#elif !defined(F77_NAME_LOWER_USCORE)
#define nfmpi_delete_  nfmpi_delete
/* Else leave name alone */
#endif
FORTRAN_API void FORT_CALL nfmpi_delete_(char * name, int *err, int d1) {
    char *p1;

    {char *p = name + d1 - 1;
     int  li;
        while (*p == ' ' && p > name) p--;
        p++;
        p1 = (char *)malloc( p-name + 1 );
        for (li=0; li<(p-name); li++) { p1[li] = name[li]; }
        p1[li] = 0; 
    }

	if ( unlink(p1) != 0 )
		*err = errno;
	else
		*err = 0;
	free(p1);
}

#endif

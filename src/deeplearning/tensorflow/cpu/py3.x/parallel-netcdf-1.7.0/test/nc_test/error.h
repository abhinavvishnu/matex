/*
 *  Copyright (C) 2003, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id: error.h 1468 2013-10-26 16:53:18Z wkliao $ */


#ifdef __cplusplus
extern "C" {
#endif

/* Print error message to stderr, don't exit */
extern void	error (const char *fmt, ...)
#ifdef _GNUC_
__attribute__ ((format (printf, 1, 2)))
#endif
;


void print(const char *fmt, ...)
#ifdef _GNUC_
__attribute__ ((format (printf, 1, 2)))
#endif
;


extern int ifFail(const int expr, const int line, const char *file);

extern void
print_n_size_t(size_t nelems, const MPI_Offset *array);

#ifdef __cplusplus
}
#endif

#define IF(EXPR) if (ifFail(EXPR, __LINE__, __FILE__))
#define ELSE_NOK else {nok++;}

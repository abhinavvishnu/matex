/*
 *  Copyright (C) 2003, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id: rnd.h 2021 2015-04-09 20:29:40Z wkliao $ */
#ifndef _RNDUP

/* useful for aligning memory */
#define	_RNDUP(x, unit)  ((((x) + (unit) - 1) / (unit)) * (unit))
#define	_RNDDOWN(x, unit)  ((x) - ((x)%(unit)))

/* #define M_RND_UNIT	(sizeof(double))
 * SIZEOF_DOUBLE is defined in ncconfig.h
 */
#define M_RND_UNIT	SIZEOF_DOUBLE
#define	M_RNDUP(x) _RNDUP(x, M_RND_UNIT)
#define	M_RNDDOWN(x)  __RNDDOWN(x, M_RND_UNIT)

#endif

/*
 *  Copyright (C) 2003, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id: fbits.h 1468 2013-10-26 16:53:18Z wkliao $ */

#ifndef _FBITS_H_
#define _FBITS_H_

/*
 * Macros for dealing with flag bits.
 */
#define fSet(t, f)       ((t) |= (f))
#define fClr(t, f)       ((t) &= ~(f))
#define fIsSet(t, f)     ((t) & (f))
#define fMask(t, f)     ((t) & ~(f))

/*
 * Propositions
 */
/* a implies b */
#define pIf(a,b) (!(a) || (b))
/* a if and only if b, use == when it makes sense */
#define pIff(a,b) (((a) && (b)) || (!(a) && !(b)))

#endif /*!FBITS_H_*/

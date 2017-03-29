/*
 *  Copyright (C) 2003, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id: testutils.h 2133 2015-09-26 19:16:01Z wkliao $ */


#ifndef _UTILS_H
#define _UTILS_H

#include <limits.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

typedef struct {
	char infname[PATH_MAX];
	char outfname[PATH_MAX];
} params;

void parse_read_args(int argc, char **argv, int rank, params *p);
void parse_write_args(int argc, char **argv, int rank, params *p);

#ifdef PNC_DEBUG
#define PASS_STR "\x1b[32mpass\x1b[0m\n"
#define FAIL_STR "\x1b[31mfail\x1b[0m with %d mismatches\n"
#else
#define PASS_STR "pass\n"
#define FAIL_STR "fail with %d mismatches\n"
#endif

extern char* nc_err_code_name(int err);

#endif

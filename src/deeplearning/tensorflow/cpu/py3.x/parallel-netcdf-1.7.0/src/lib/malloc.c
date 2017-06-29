/*
 *  Copyright (C) 2003, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id: malloc.c 2244 2015-12-19 22:41:28Z wkliao $ */

/* These are routines for allocating and deallocating memory.
   They should be called as NCI_Malloc(size) and
   NCI_Free(ptr). In macro.h, they are macro-replaced to
   NCI_Malloc_fn(size,__LINE__,__FILE__) and
   NCI_Free_fn(ptr,__LINE__,__FILE__).
 */

#if HAVE_CONFIG_H
# include "ncconfig.h"
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h> /* strcpy(), strlen() */

#include <mpi.h>
#include "nc.h"

#ifdef PNC_MALLOC_TRACE

#ifdef HAVE_SEARCH_H
#include <search.h> /* tfind(), tsearch() and tdelete() */
#endif

/* global variables for malloc tracing */
static void       *ncmpii_mem_root;
static MPI_Offset  ncmpii_mem_alloc;
static MPI_Offset  ncmpii_max_mem_alloc;

/*----< ncmpii_init_malloc_tracing() >----------------------------------------*/
void ncmpii_init_malloc_tracing(void)
{
    ncmpii_mem_alloc     = 0;
    ncmpii_max_mem_alloc = 0;
    ncmpii_mem_root      = NULL;
}
#endif

/*----< ncmpii_inq_malloc_size() >--------------------------------------------*/
/* get the current aggregate size allocated by malloc */
int ncmpii_inq_malloc_size(MPI_Offset *size)
{
#ifdef PNC_MALLOC_TRACE
    *size = ncmpii_mem_alloc;
    return 1;
#else
    return 0;
#endif
}

/*----< ncmpii_inq_malloc_max_size() >----------------------------------------*/
/* get the max watermark ever researched by malloc */
int ncmpii_inq_malloc_max_size(MPI_Offset *size)
{
#ifdef PNC_MALLOC_TRACE
    *size = ncmpii_max_mem_alloc;
    return 1;
#else
    return 0;
#endif
}

#ifdef PNC_MALLOC_TRACE

#ifndef MAX
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#endif

typedef struct {
    void   *self;
    void   *buf;
    size_t  size;
    int     lineno;
    char   *func;
    char   *filename;
} ncmpii_mem_entry;

inline static
int ncmpii_cmp(const void *a, const void *b) {
    ncmpii_mem_entry *fa = (ncmpii_mem_entry*)a;
    ncmpii_mem_entry *fb = (ncmpii_mem_entry*)b;

    if (fa->buf > fb->buf) return  1;
    if (fa->buf < fb->buf) return -1;
    return 0;
}

inline static
void walker(const void *node, const VISIT which, const int depth) {
    ncmpii_mem_entry *f;
    f = *(ncmpii_mem_entry **)node;
    if (which == preorder || which == leaf)
        printf("Warning: malloc yet to be freed (buf=%p size=%zd filename=%s func=%s line=%d)\n", f->buf, f->size, f->filename, f->func, f->lineno);
}
#endif

/* check if there is any malloc residue */
int ncmpii_inq_malloc_list(void)
{
#ifdef PNC_MALLOC_TRACE
    /* check if malloc tree is empty */
    if (ncmpii_mem_root != NULL)
        twalk(ncmpii_mem_root, walker);
    return 1;
#else
    return 0;
#endif
}

#ifdef PNC_MALLOC_TRACE
/*----< ncmpii_add_mem_entry() >----------------------------------------------*/
/* add a new malloc entry to the table */
static
void ncmpii_add_mem_entry(void       *buf,
                          size_t      size,
                          const int   lineno,
                          const char *func,
                          const char *filename)
{
    /* use C tsearch utility */
    ncmpii_mem_entry *node = (ncmpii_mem_entry*) malloc(sizeof(ncmpii_mem_entry));
    node->self     = node;
    node->buf      = buf;
    node->size     = size;
    node->lineno   = lineno;
    node->func     = (char*)malloc(strlen(func)+1);
    node->filename = (char*)malloc(strlen(filename)+1);
    strcpy(node->func, func);
    node->func[strlen(func)] = '\0';
    strcpy(node->filename, filename);
    node->filename[strlen(filename)] = '\0';

    /* search and add a new item */
    void *ret = tsearch(node, &ncmpii_mem_root, ncmpii_cmp);
    if (ret == NULL) {
        printf("Error: tsearch()\n");
        return;
    }
    ncmpii_mem_alloc += (MPI_Offset)size;
    ncmpii_max_mem_alloc = MAX(ncmpii_max_mem_alloc, ncmpii_mem_alloc);
}

/*----< ncmpii_del_mem_entry() >----------------------------------------------*/
/* delete a malloc entry from the table */
static
void ncmpii_del_mem_entry(void *buf)
{
    /* use C tsearch utility */
    if (ncmpii_mem_root != NULL) {
        ncmpii_mem_entry node;
        node.buf  = buf;
        void *ret = tfind(&node, &ncmpii_mem_root, ncmpii_cmp);
        ncmpii_mem_entry **found = (ncmpii_mem_entry**) ret;
        if (ret == NULL) {
            printf("Error: tfind() buf=%p\n", buf);
            return;
        }
        /* free space for func and filename */
        free((*found)->func);
        free((*found)->filename);

        /* subtract the space amount to be freed */
        ncmpii_mem_alloc -= (MPI_Offset)((*found)->size);
        void *tmp = (*found)->self;
        ret = tdelete(&node, &ncmpii_mem_root, ncmpii_cmp);
        if (ret == NULL) {
            printf("Error: tdelete() buf=%p\n", buf);
            return;
        }
        free(tmp);
    }
}
#endif

/*----< NCI_Malloc_fn() >-----------------------------------------------------*/
void *NCI_Malloc_fn(size_t      size,
                    const int   lineno,
                    const char *func,
                    const char *filename)
{
#ifdef NC_DEBUG
    if (size == 0)
        fprintf(stderr, "Attempt to malloc zero-size in file %s, line %d\n", filename, lineno);
#endif
    if (size == 0) return NULL;
    void *buf = malloc(size);
    if (!buf) {
	fprintf(stderr, "malloc(%zd) failed in file %s, line %d\n", size, filename, lineno);
	MPI_Abort(MPI_COMM_WORLD, 1);
    }
#ifdef PNC_MALLOC_TRACE
    ncmpii_add_mem_entry(buf, size, lineno, func, filename);
#endif
    return buf;
}


/*----< NCI_Calloc_fn() >-----------------------------------------------------*/
void *NCI_Calloc_fn(size_t      nelem,
                    size_t      elsize,
                    const int   lineno,
                    const char *func,
                    const char *filename)
{
#ifdef NC_DEBUG
    if (nelem == 0 || elsize == 0)
        fprintf(stderr, "Attempt to calloc zero-size in file %s, line %d\n", filename, lineno);
#endif
    if (nelem == 0 || elsize == 0) return NULL;
    void *buf =calloc(nelem, elsize);
    if (!buf) {
	fprintf(stderr, "calloc(%zd, %zd) failed in file %s, line %d\n", nelem, elsize, filename, lineno);
	MPI_Abort(MPI_COMM_WORLD, 1);
    }
#ifdef PNC_MALLOC_TRACE
    ncmpii_add_mem_entry(buf, nelem * elsize, lineno, func, filename);
#endif
    return buf;
}


/*----< NCI_Realloc_fn() >----------------------------------------------------*/
void *NCI_Realloc_fn(void       *ptr,
                     size_t      size,
                     const int   lineno,
                     const char *func,
                     const char *filename)
{
#ifdef NC_DEBUG
    if (size == 0)
        fprintf(stderr, "Attempt to realloc zero-size in file %s, line %d\n", filename, lineno);
#endif
#ifdef PNC_MALLOC_TRACE
    if (ptr != NULL) ncmpii_del_mem_entry(ptr);
#endif
    if (size == 0) return NULL;
    void *buf = (void *) realloc(ptr, size);
    if (!buf) {
	fprintf(stderr, "realloc failed in file %s, line %d\n", filename, lineno);
	MPI_Abort(MPI_COMM_WORLD, 1);
    }
#ifdef PNC_MALLOC_TRACE
    ncmpii_add_mem_entry(buf, size, lineno, func, filename);
#endif
    return buf;
}


/*----< NCI_Free_fn() >-------------------------------------------------------*/
void NCI_Free_fn(void       *ptr,
                 const int   lineno,
                 const char *func,
                 const char *filename)
{
#ifdef NC_DEBUG
    if (ptr == NULL)
	fprintf(stderr, "Attempt to free null pointer in file %s, line %d\n", filename, lineno);
#endif
    if (ptr == NULL) return;
#ifdef PNC_MALLOC_TRACE
    ncmpii_del_mem_entry(ptr);
#endif
    free(ptr);
}



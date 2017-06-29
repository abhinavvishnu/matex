/*
 *  Copyright (C) 2003, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id: ncmpivalid.c 2335 2016-03-03 14:55:12Z wkliao $ */

#if HAVE_CONFIG_H
# include <ncconfig.h>
#endif

#include <assert.h>
#include <sys/types.h>  /* open() */
#include <sys/stat.h>   /* open() */
#include <fcntl.h>      /* open() */
#include <unistd.h>     /* read() */
#include <string.h>
#include <stdio.h>
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#include <errno.h>

#include <mpi.h>

#include <ncx.h>
#include <macro.h>

/*
 * "magic number" at beginning of file: 0x43444601 (big endian) 
 */
static const schar ncmagic[] = {'C', 'D', 'F', 0x01}; 

/* Prototypes for functions used only in this file */
static int val_get_NCtype(bufferinfo *gbp, NCtype *typep);
static int val_get_size_t(bufferinfo *gbp, MPI_Offset *sp);
static int val_get_NC_string(bufferinfo *gbp, NC_string **ncstrpp);
static int val_get_NC_dim(bufferinfo *gbp, NC_dim **dimpp);
static int val_get_NC_dimarray(bufferinfo *gbp, NC_dimarray *ncap);
static int val_get_nc_type(bufferinfo *gbp, nc_type *typep);
static int val_get_NC_attrV(bufferinfo *gbp, NC_attr *attrp);
static int val_get_NC_attr(bufferinfo *gbp, NC_attr **attrpp);
static int val_get_NC_attrarray(bufferinfo *gbp, NC_attrarray *ncap);
static int val_get_NC_var(bufferinfo *gbp, NC_var **varpp);
static int val_get_NC_vararray(bufferinfo *gbp, NC_vararray *ncap);
static int val_get_NC(NC *ncp);

static int val_fetch(bufferinfo *gbp, MPI_Offset fsize);
static int val_check_buffer(bufferinfo *gbp, MPI_Offset nextread);


#define ABORT {printf("Abort: at line=%d func=%s\n", __LINE__,__func__); fflush(stdout); MPI_Abort(MPI_COMM_WORLD, -1);}

/* Begin Of get NC */

/*
 * Fetch the next header chunk.
 */
static int
val_fetch(bufferinfo *gbp, MPI_Offset fsize) {
  ssize_t nn = 0;
  MPI_Offset slack;        /* any leftover data in the buffer */
  MPI_Aint pos_addr, base_addr;

  assert(gbp->base != NULL);
  
#ifdef HAVE_MPI_GET_ADDRESS
    MPI_Get_address(gbp->pos,  &pos_addr);
    MPI_Get_address(gbp->base, &base_addr);
#else
    MPI_Address(gbp->pos,  &pos_addr);
    MPI_Address(gbp->base, &base_addr);
#endif
  slack = gbp->size - (pos_addr - base_addr);
  /* . if gbp->pos and gbp->base are the same, there is no leftover buffer data
   *   to worry about.  
   * In the other extreme, where gbp->size == (gbp->pos - gbp->base), then all
   * data in the buffer has been consumed */
  if (slack == gbp->size) slack = 0;

  memset(gbp->base, 0, gbp->size);
  gbp->pos = gbp->base;
  gbp->index = 0;

  if (-1 == lseek(gbp->nciop->fd, gbp->offset-slack, SEEK_SET)) {
      printf("Error at line %d: lseek %s\n",__LINE__,strerror(errno));
      return -1;
  }
  nn = read(gbp->nciop->fd, gbp->base, gbp->size);
  if (nn < gbp->size) {
      printf("Error: Unexpected EOF ");
      return -1;
  }
  gbp->offset += (gbp->size - slack);

  return NC_NOERR;
}

/*
 * Ensure that 'nextread' bytes are available.
 */
static int
val_check_buffer(bufferinfo *gbp,
                 MPI_Offset  nextread)
{
    MPI_Aint pos_addr, base_addr;

#ifdef HAVE_MPI_GET_ADDRESS
    MPI_Get_address(gbp->pos,  &pos_addr);
    MPI_Get_address(gbp->base, &base_addr);
#else
    MPI_Address(gbp->pos,  &pos_addr);
    MPI_Address(gbp->base, &base_addr);
#endif
    if (pos_addr + nextread <= base_addr + gbp->size)
        return NC_NOERR;

    return val_fetch(gbp, MIN(gbp->size, nextread));
} 

static int
val_get_NCtype(bufferinfo *gbp, NCtype *typep) {
  unsigned int type = 0;
  int status = val_check_buffer(gbp, X_SIZEOF_INT);
  if (status != NC_NOERR) {
    printf("NC component type is expected for ");
    return status;
  }

  status = ncmpix_get_uint32((const void**)(&gbp->pos), &type);
  if (status != NC_NOERR)
    return status;
  *typep = (NCtype) type;
  return NC_NOERR;
}

static int
val_get_size_t(bufferinfo *gbp, MPI_Offset *sp) {
  int sizeof_t = (gbp->version == 5) ? 8 : 4; 
  int status = val_check_buffer(gbp, sizeof_t);
  if (status != NC_NOERR) {
    printf("size is expected for ");
    return status; 
  }
  if (gbp->version == 5) {
      unsigned long long tmp=0;
      status = ncmpix_get_uint64((const void **)(&gbp->pos), &tmp);
      *sp = (MPI_Offset)tmp;
  }
  else {
      unsigned int tmp=0;
      status = ncmpix_get_uint32((const void **)(&gbp->pos), &tmp);
      *sp = (MPI_Offset)tmp;
  }
  return status;
}

static int
val_get_NC_string(bufferinfo *gbp, NC_string **ncstrpp) {
  int status;
  MPI_Offset  nchars = 0, padding, bufremain, strcount; 
  NC_string *ncstrp;
  char *cpos, pad[X_ALIGN-1];
  MPI_Aint pos_addr, base_addr;

  status = val_get_size_t(gbp, &nchars);
  if (status != NC_NOERR) {
    printf("the name string of ");
    return status;
  }

  ncstrp = ncmpii_new_NC_string(nchars, NULL);
  if (ncstrp == NULL)
    return NC_ENOMEM;

  padding = _RNDUP(X_SIZEOF_CHAR * ncstrp->nchars, X_ALIGN)
            - X_SIZEOF_CHAR * ncstrp->nchars;
#ifdef HAVE_MPI_GET_ADDRESS
    MPI_Get_address(gbp->pos,  &pos_addr);
    MPI_Get_address(gbp->base, &base_addr);
#else
    MPI_Address(gbp->pos,  &pos_addr);
    MPI_Address(gbp->base, &base_addr);
#endif
  bufremain = gbp->size - (pos_addr - base_addr);
  cpos = ncstrp->cp;

  while (nchars > 0) {
    if (bufremain > 0) {
      strcount = MIN(bufremain, X_SIZEOF_CHAR * nchars); 
      (void) memcpy(cpos, gbp->pos, strcount);
      nchars -= strcount/X_SIZEOF_CHAR;
      gbp->pos = (void *)((char *)gbp->pos + strcount);
      cpos += strcount; 
      bufremain -= strcount;
    } else {
      status = val_fetch(gbp, MIN(gbp->size, X_SIZEOF_CHAR * nchars));
      if(status != NC_NOERR) {
	printf("fetching the name string of ");
        ncmpii_free_NC_string(ncstrp);
        return status;
      } 
      bufremain = gbp->size;
    }
  }

  memset(pad, 0, X_ALIGN-1);
  status = val_check_buffer(gbp, padding);
  if(status != NC_NOERR) {
    printf("fetching padding for the name string of ");
    ncmpii_free_NC_string(ncstrp);
    return status;
  } 
  if (memcmp(gbp->pos, pad, padding) != 0) {
    printf("Error @ [0x%8.8Lx]: \n\tPadding should be 0x00 for the name string alignment of ", (long long unsigned)
	   (((size_t) gbp->pos - (size_t) gbp->base) + gbp->offset - gbp->size));
    ncmpii_free_NC_string(ncstrp);
    return NC_EINVAL;
  }
  gbp->pos = (void *)((char *)gbp->pos + padding);
  
  *ncstrpp = ncstrp;
  
  return NC_NOERR;  
}

static int
val_get_NC_dim(bufferinfo *gbp, NC_dim **dimpp) {
  int status;
  NC_string *ncstrp;
  NC_dim *dimp;

  status = val_get_NC_string(gbp, &ncstrp);
  if (status != NC_NOERR) 
    return status;

  dimp = ncmpii_new_x_NC_dim(ncstrp);
  if(dimp == NULL)
    return NC_ENOMEM;

  status = val_get_size_t(gbp, &dimp->size);
  if(status != NC_NOERR) {
    printf("\"%s\" - ", ncstrp->cp);
    ncmpii_free_NC_dim(dimp); /* frees name */
    return status;
  }

  *dimpp = dimp;

  return NC_NOERR;
}

static int
val_get_NC_dimarray(bufferinfo *gbp, NC_dimarray *ncap) {
  int status;
  NCtype type = NC_UNSPECIFIED; 
  NC_dim **dpp, **end;
  int dim;
  MPI_Offset tmp;

  assert(gbp != NULL && gbp->pos != NULL);
  assert(ncap != NULL);
  assert(ncap->value == NULL);

  status = val_get_NCtype(gbp, &type);
  if(status != NC_NOERR) {
    printf("preamble of ");
    return status; 
  }

  status = val_get_size_t(gbp, &tmp);
  if(status != NC_NOERR) {
    printf("the length of ");
    return status;
  }
  ncap->ndefined = tmp; /* number of allowable defined variables < 2^32 */

  if(ncap->ndefined == 0) {
    if (type != NC_DIMENSION && type != NC_UNSPECIFIED) {
      printf("Error @ [0x%8.8Lx]: \n\tInvalid NC component type, while ",
	      (long long unsigned) (((size_t) gbp->pos - (size_t) gbp->base) + gbp->offset - gbp->size - 2 * X_SIZEOF_SIZE_T));
      printf("NC_DIMENSION or NC_UNSPECIFIED is expected for ");
      return NC_EINVAL;
    }
  } else {
    if(type != NC_DIMENSION) {
      printf("Error @ [0x%8.8Lx]: \n\tInvalid NC component type, while ",
	      (long long unsigned) (((size_t) gbp->pos - (size_t) gbp->base) + gbp->offset - gbp->size - 2 * X_SIZEOF_SIZE_T));
      printf("NC_DIMENSION is expected since number of dimensions is %d for ", ncap->ndefined);
      return NC_EINVAL;
    }

    ncap->value = (NC_dim **) NCI_Malloc(ncap->ndefined * sizeof(NC_dim *));
    if(ncap->value == NULL)
      return NC_ENOMEM;
    ncap->nalloc = ncap->ndefined;

    dpp = ncap->value;
    end = &dpp[ncap->ndefined];
    for( /*NADA*/ dim = 0; dpp < end; dpp++, dim++) {
      status = val_get_NC_dim(gbp, dpp);
      if (status != NC_NOERR) {
	printf("dimension[%d] in ", dim);
        ncap->ndefined = dpp - ncap->value;
        ncmpii_free_NC_dimarray(ncap);
        return status;
      }
    }
  }

  return NC_NOERR;
}

static int
val_get_nc_type(bufferinfo *gbp, nc_type *typep) {
    /* NCtype is 4-byte integer */
    unsigned int type = 0;
    int status = val_check_buffer(gbp, 4);
    if (status != NC_NOERR) return status;

    /* get a 4-byte integer */
    status = ncmpix_get_uint32((const void**)(&gbp->pos), &type);
    gbp->index += X_SIZEOF_INT;
    if (status != NC_NOERR) return status;

  if (   type != NC_BYTE
      && type != NC_UBYTE
      && type != NC_CHAR
      && type != NC_SHORT
      && type != NC_USHORT
      && type != NC_INT
      && type != NC_UINT
      && type != NC_FLOAT
      && type != NC_DOUBLE
      && type != NC_INT64
      && type != NC_UINT64) {
    printf("Error @ [0x%8.8Lx]: \n\tUnknown data type for the values of ",
	   (long long unsigned) (((size_t) gbp->pos - (size_t) gbp->base) + gbp->offset - gbp->size - X_SIZEOF_INT));
    return NC_EINVAL; 
  }
 
  *typep = (nc_type) type;

  return NC_NOERR;
}

/*
 * Get the values of an attribute  
 */
static int
val_get_NC_attrV(bufferinfo *gbp, NC_attr *attrp) {
    int status;
    void *value = attrp->xvalue;
    char pad[X_ALIGN-1]; 
    MPI_Offset nvalues = attrp->nelems, esz, padding, bufremain, attcount;
    MPI_Aint pos_addr, base_addr;

    esz = ncmpix_len_nctype(attrp->type);
    padding = attrp->xsz - esz * nvalues;
#ifdef HAVE_MPI_GET_ADDRESS
    MPI_Get_address(gbp->pos,  &pos_addr);
    MPI_Get_address(gbp->base, &base_addr);
#else
    MPI_Address(gbp->pos,  &pos_addr);
    MPI_Address(gbp->base, &base_addr);
#endif
    bufremain = gbp->size - (pos_addr - base_addr);

  while (nvalues > 0) {
    if (bufremain > 0) {
      attcount = MIN(bufremain, esz * nvalues);
      (void) memcpy(value, gbp->pos, attcount);
      nvalues -= attcount/esz;
      gbp->pos = (void *)((char *)gbp->pos + attcount);
      value = (void *)((char *)value + attcount);
      bufremain -= attcount;
    } else {
      status = val_fetch(gbp, MIN(gbp->size, esz * nvalues));
      if(status != NC_NOERR) {
	printf("fetching the values of ");
        return status;
      }
      bufremain = gbp->size;
    }
  }
 
  memset(pad, 0, X_ALIGN-1);
  if (memcmp(gbp->pos, pad, padding) != 0) {
    printf("Error @ [0x%8.8Lx]: \n\tPadding should be 0x00 for the values alignment of ",
           (long long unsigned) (((size_t) gbp->pos - (size_t) gbp->base) + gbp->offset - gbp->size)); 
    return NC_EINVAL;
  }
  gbp->pos = (void *)((char *)gbp->pos + padding);

  return NC_NOERR;
}

static int
val_get_NC_attr(bufferinfo *gbp, NC_attr **attrpp) {
  NC_string *strp;
  int status;
  nc_type type; 
  MPI_Offset nelems;
  NC_attr *attrp;

  status = val_get_NC_string(gbp, &strp);
  if(status != NC_NOERR)
    return status;

  status = val_get_nc_type(gbp, &type);
  if(status != NC_NOERR) {
    printf("\"%s\" - ", strp->cp);
    ncmpii_free_NC_string(strp);
    return status;
  }

  status = val_get_size_t(gbp, &nelems); 
  if(status != NC_NOERR) {
    printf("the values of \"%s\" - ", strp->cp);
    ncmpii_free_NC_string(strp);
    return status;
  }

  attrp = ncmpii_new_x_NC_attr(strp, type, nelems);
  if(attrp == NULL) {
    ncmpii_free_NC_string(strp);
    return status;
  }

  status = val_get_NC_attrV(gbp, attrp);
  if(status != NC_NOERR) {
    printf("\"%s\" - ", strp->cp);
    ncmpii_free_NC_attr(attrp); /* frees strp */ 
    return status;
  }

  *attrpp = attrp; 
  
  return NC_NOERR; 
}

static int
val_get_NC_attrarray(bufferinfo *gbp, NC_attrarray *ncap){
  int status;
  NCtype type = NC_UNSPECIFIED;
  NC_attr **app, **end;
  int att;
  MPI_Offset tmp;

  assert(gbp != NULL && gbp->pos != NULL);
  assert(ncap != NULL);
  assert(ncap->value == NULL);

  status = val_get_NCtype(gbp, &type);
  if(status != NC_NOERR) {
    printf("preamble of ");
    return status; 
  }

  status = val_get_size_t(gbp, &tmp);
  if(status != NC_NOERR) {
    printf("the length of ");
    return status;
  }
  ncap->ndefined = tmp; /* number of allowable defined variables < 2^32 */

  if(ncap->ndefined == 0) {
    if (type != NC_ATTRIBUTE && type != NC_UNSPECIFIED) {
      printf("Error @ [0x%8.8Lx]: \n\tInvalid NC component type, while ",
              (long long unsigned) (((size_t) gbp->pos - (size_t) gbp->base) + gbp->offset - gbp->size - 2 * X_SIZEOF_SIZE_T));
      printf("NC_ATTRIBUTE or NC_UNSPECIFIED is expected for "); 
      return NC_EINVAL;
    }
  } else {
    if(type != NC_ATTRIBUTE) {
      printf("Error @ [0x%8.8Lx]: \n\tInvalid NC component type, while ",
              (long long unsigned) (((size_t) gbp->pos - (size_t) gbp->base) + gbp->offset - gbp->size - 2 * X_SIZEOF_SIZE_T));
      printf("NC_ATTRIBUTE is expected since number of attributes is %d for ", (int)ncap->ndefined);  
      return NC_EINVAL;
    }

    ncap->value = (NC_attr **) NCI_Malloc(ncap->ndefined * sizeof(NC_attr *));
    if(ncap->value == NULL)
      return NC_ENOMEM;
    ncap->nalloc = ncap->ndefined; 

    app = ncap->value;
    end = &app[ncap->ndefined];
    for( /*NADA*/ att = 0; app < end; app++, att++) {
      status = val_get_NC_attr(gbp, app);
      if (status != NC_NOERR) {
	printf("attribute[%d] of ", att);
        ncap->ndefined = app - ncap->value;
        ncmpii_free_NC_attrarray(ncap);
        return status;
      }
    }
  }
  
  return NC_NOERR;
}

static int
val_get_NC_var(bufferinfo *gbp, NC_var **varpp) {
  NC_string *strp;
  int status;
  MPI_Offset ndims, *tmp_dim;
  size_t dim;
  NC_var *varp;

  status = val_get_NC_string(gbp, &strp);
  if(status != NC_NOERR)
    return status;

  status = val_get_size_t(gbp, &ndims);
  if(status != NC_NOERR) {
     printf("the dimid list of \"%s\" - ", strp->cp);
     ncmpii_free_NC_string(strp); 
     return status;
  }

  varp = ncmpii_new_x_NC_var(strp, ndims);
  if(varp == NULL) {
    ncmpii_free_NC_string(strp);
    return NC_ENOMEM;
  }

  for (dim = 0; dim < ndims; dim++ ) {
    status = val_check_buffer(gbp, (gbp->version == 5 ? 8 : 4));
    if(status != NC_NOERR) {
      printf("the dimid[%d] is expected for \"%s\" - ", (int)dim, strp->cp);
      ncmpii_free_NC_var(varp);
      return status;
    }
    tmp_dim = (MPI_Offset*) (varp->dimids + dim);
    if (gbp->version == 5) {
        unsigned long long tmp=0;
        status = ncmpix_get_uint64((const void **)(&gbp->pos), &tmp);
        *tmp_dim = (MPI_Offset)tmp;
    }
    else {
        unsigned int tmp=0;
        status = ncmpix_get_uint32((const void **)(&gbp->pos), &tmp);
        *tmp_dim = (MPI_Offset)tmp;
    }
    if(status != NC_NOERR) {
      ncmpii_free_NC_var(varp);
      return status;
    }
  }

  status = val_get_NC_attrarray(gbp, &varp->attrs);
  if(status != NC_NOERR) {
    printf("ATTRIBUTE list of \"%s\" - ", strp->cp);
    ncmpii_free_NC_var(varp);
    return status;
  }

  status = val_get_nc_type(gbp, &varp->type);
  if(status != NC_NOERR) {
    printf("\"%s\" - ", strp->cp);
    ncmpii_free_NC_var(varp);
    return status;
  } 

  status = val_get_size_t(gbp, &varp->len);
  if(status != NC_NOERR) {
    printf("the data of  \"%s\" - ", strp->cp);
    ncmpii_free_NC_var(varp);
    return status;
  }

  status = val_check_buffer(gbp, (gbp->version == 5 ? 8 : 4));
  if(status != NC_NOERR) {
    printf("offset is expected for the data of \"%s\" - ", strp->cp);
    ncmpii_free_NC_var(varp);
    return status;
  }
  if (gbp->version == 1) {
      unsigned int tmp=0;
      status = ncmpix_get_uint32((const void **)(&gbp->pos), &tmp);
      varp->begin = (MPI_Offset)tmp;
  }
  else {
      unsigned long long tmp=0;
      status = ncmpix_get_uint64((const void **)(&gbp->pos), &tmp);
      varp->begin = (MPI_Offset)tmp;
  }
  if(status != NC_NOERR) {
    ncmpii_free_NC_var(varp);
    return status;
  }

  *varpp = varp;
  return NC_NOERR;
}

static int
val_get_NC_vararray(bufferinfo *gbp, NC_vararray *ncap) {
  int status;
  NCtype type = NC_UNSPECIFIED;
  NC_var **vpp, **end;
  int var;
  MPI_Offset tmp;

  assert(gbp != NULL && gbp->pos != NULL);
  assert(ncap != NULL);
  assert(ncap->value == NULL); 

  status = val_get_NCtype(gbp, &type);
  if(status != NC_NOERR) {
    printf("preamble of ");
    return status;
  }
 
  status = val_get_size_t(gbp, &tmp);
  if(status != NC_NOERR) {
    printf("the length of ");
    return status;
  }
  ncap->ndefined = tmp; /* number of allowable defined variables < 2^32 */
 
  if(ncap->ndefined == 0) {
    if (type != NC_VARIABLE && type != NC_UNSPECIFIED) {
      printf("Error @ [0x%8.8Lx]: \n\tInvalid NC component type, while ",
              (long long unsigned) (((size_t) gbp->pos - (size_t) gbp->base) + gbp->offset - gbp->size - 2 * X_SIZEOF_SIZE_T));
      printf("NC_VARIABLE or NC_UNSPECIFIED is expected for ");
      return NC_EINVAL;
    }
  } else {
    if(type != NC_VARIABLE) {
      printf("Error @ [0x%8.8Lx]: \n\tInvalid NC component type, while ",
              (long long unsigned) (((size_t) gbp->pos - (size_t) gbp->base) + gbp->offset - gbp->size - 2 * X_SIZEOF_SIZE_T));
      printf("NC_VARIABLE is expected since number of variables is %d for ", ncap->ndefined);        
      return NC_EINVAL;
    }
 
    ncap->value = (NC_var **) NCI_Malloc(ncap->ndefined * sizeof(NC_var *));
    if(ncap->value == NULL)
      return NC_ENOMEM; 
    ncap->nalloc = ncap->ndefined;

    vpp = ncap->value;
    end = &vpp[ncap->ndefined];
    for( /*NADA*/ var = 0; vpp < end; vpp++, var++) {
      status = val_get_NC_var(gbp, vpp);
      if (status != NC_NOERR) {
        printf("variable[%d] in ", var);
        ncap->ndefined = vpp - ncap->value;
        ncmpii_free_NC_vararray(ncap);
        return status;
      }
    }
  }

  return NC_NOERR;
}

static int
val_get_NC(NC *ncp) {
    int status;
    bufferinfo getbuf;
    schar magic[sizeof(ncmagic)];
    MPI_Offset nrecs = 0;
    MPI_Aint pos_addr, base_addr;

    assert(ncp != NULL);

    /* Initialize the get buffer that stores the header read from the file */
    getbuf.nciop = ncp->nciop;
    getbuf.offset = 0;     /* read from start of the file */
    getbuf.put_size = 0;   /* amount of writes so far in bytes */
    getbuf.get_size = 0;   /* amount of reads  so far in bytes */

    /* CDF-5's minimum header size is 4 bytes more than CDF-1 and CDF-2's */
    getbuf.size = _RNDUP( MAX(MIN_NC_XSZ+4, ncp->chunk), X_ALIGN );
    if (getbuf.size > NC_DEFAULT_CHUNKSIZE)
        getbuf.size = NC_DEFAULT_CHUNKSIZE;

    getbuf.pos = getbuf.base = (void *)NCI_Malloc(getbuf.size);
    getbuf.index = 0;

    /* Fetch the next header chunk. The chunk is 'gbp->size' bytes big */
    status = val_fetch(&getbuf, sizeof(magic));
    if (status != NC_NOERR) {
        printf("magic number (C D F \\001) is expected!\n");
        return status;
    }
  
    /* First get the file format information, magic */
    memset(magic, 0, sizeof(magic));
    status = ncmpix_getn_schar_schar((const void **)(&getbuf.pos),
                                     sizeof(magic), magic);
    getbuf.index += sizeof(magic);

    if (memcmp(magic, ncmagic, sizeof(ncmagic)-1) != 0) {
        printf("Error @ [0x%8.8x]: \n\tUnknow magic number, while (C D F \\001, \\002, or \\005) is expected!\n", (unsigned) 0);
        NCI_Free(getbuf.base);
        return NC_ENOTNC;
    }

    /* check version number in last byte of magic */
    if (magic[sizeof(ncmagic)-1] == 0x1) {
        getbuf.version = 1;
        fSet(ncp->flags, NC_32BIT);
    } else if (magic[sizeof(ncmagic)-1] == 0x2) {
        getbuf.version = 2;
        fSet(ncp->flags, NC_64BIT_OFFSET);
        if (sizeof(MPI_Offset) != 8) {
            /* take the easy way out: if we can't support all CDF-2
             * files, return immediately */
            NCI_Free(getbuf.base);
            return NC_ESMALL;
        }
    } else if (magic[sizeof(ncmagic)-1] == 0x5) {
        getbuf.version = 5;
        fSet(ncp->flags, NC_64BIT_DATA);
        if (sizeof(MPI_Offset) != 8) {
            NCI_Free(getbuf.base);
            return NC_ESMALL;
        }
    } else {
        NCI_Free(getbuf.base);
        return NC_ENOTNC;
    }

    /* status = val_check_buffer(&getbuf, X_SIZEOF_SIZE_T); */
    status = val_check_buffer(&getbuf, (getbuf.version == 1) ? 4 : 8);
    if (status != NC_NOERR) {
        printf("number of records is expected!\n");
        NCI_Free(getbuf.base);
        return status;
    }

    /* get numrecs from getbuf into ncp */
    if (getbuf.version == 5) {
        unsigned long long tmp=0;
        status = ncmpix_get_uint64((const void **)(&getbuf.pos), &tmp);
        nrecs = (MPI_Offset)tmp;
    }
    else {
        unsigned int tmp=0;
        status = ncmpix_get_uint32((const void **)(&getbuf.pos), &tmp);
        nrecs = (MPI_Offset)tmp;
    }
    if (status != NC_NOERR) {
        NCI_Free(getbuf.base);
        return status;
    }

    if (getbuf.version == 5)
        getbuf.index += X_SIZEOF_INT64;
    else
        getbuf.index += X_SIZEOF_SIZE_T;

    ncp->numrecs = nrecs;

#ifdef HAVE_MPI_GET_ADDRESS
    MPI_Get_address(getbuf.pos,  &pos_addr);
    MPI_Get_address(getbuf.base, &base_addr);
#else
    MPI_Address(getbuf.pos,  &pos_addr);
    MPI_Address(getbuf.base, &base_addr);
#endif
    assert(pos_addr < base_addr + getbuf.size);

    /* get dim_list from getbuf into ncp */
    status = val_get_NC_dimarray(&getbuf, &ncp->dims);
    if (status != NC_NOERR) {
        printf("DIMENSION list!\n");
        NCI_Free(getbuf.base);
        return status;
    }

    status = val_get_NC_attrarray(&getbuf, &ncp->attrs); 
    if (status != NC_NOERR) {
        printf("GLOBAL ATTRIBUTE list!\n");
        NCI_Free(getbuf.base);
        return status;
    }

  status = val_get_NC_vararray(&getbuf, &ncp->vars);
  if(status != NC_NOERR) {
    printf("VARIABLE list!\n");
    NCI_Free(getbuf.base);
    return status; 
  }

  ncp->xsz = ncmpii_hdr_len_NC(ncp);
  status = ncmpii_NC_computeshapes(ncp);
  NCI_Free(getbuf.base);

  return status;
}

/* End Of get NC */

int main(int argc, char **argv) {

    char *ncfile;
    int status;
    NC *ncp;
    struct stat ncfilestat;

    MPI_Init(&argc, &argv);

    if (argc != 2) {
        printf("Usage: %s <ncfile>\n", argv[0]);
        MPI_Finalize();
        return 1;
    } 

    ncfile = argv[1];

    /* open the netCDF file */
    ncp = ncmpii_new_NC(NULL);
    if (ncp == NULL) {
        printf("ncmpii_new_NC(): Not enough memory!\n");
        ABORT
    }

    ncp->nciop = ncmpiio_new(ncfile, NC_NOWRITE);
    if (ncp->nciop == NULL) {
        ncmpii_free_NC(ncp);
        printf("ncmpiio_new(): Not enough memory!\n");
        ABORT
    }

    if ( (*((int *)&ncp->nciop->fd) = open(ncfile, O_RDONLY)) < 0 ) {
        printf("Can not open file: %s\n", ncfile);
        ncmpiio_free(ncp->nciop);
        ncmpii_free_NC(ncp);
        ABORT
    }

    /* read to validate the header */
    status = val_get_NC(ncp);
    if (status !=  NC_NOERR) {
        printf("Error at line %d (%s)\n",__LINE__,ncmpi_strerror(status));
        close(ncp->nciop->fd);
        ncmpiio_free(ncp->nciop);
        ncmpii_free_NC(ncp);
        ABORT
    }

    /* check data size */
    fstat(ncp->nciop->fd, &ncfilestat);
    if ( ncp->begin_rec + ncp->recsize * ncp->numrecs < ncfilestat.st_size ) {
        printf("Error: \n\tData size is larger than defined!\n");
        close(ncp->nciop->fd);
        ncmpiio_free(ncp->nciop);
        ncmpii_free_NC(ncp);
        return 0;  
    } else if ( ncp->numrecs > 0 &&
                ncp->begin_rec + ncp->recsize * (ncp->numrecs - 1) > ncfilestat.st_size ) {
        printf("Error: \n\tData size is less than expected!\n");
        printf("\tbegin_rec=%lld recsize=%lld numrecs=%lld ncfilestat.st_size=%lld\n",ncp->begin_rec, ncp->recsize, ncp->numrecs, (long long) ncfilestat.st_size);
        close(ncp->nciop->fd);
        ncmpiio_free(ncp->nciop);
        ncmpii_free_NC(ncp);
        return 0;
    }


  /* close the file */

  close(ncp->nciop->fd);
  ncmpiio_free(ncp->nciop);
  ncmpii_free_NC(ncp);

  printf("The netCDF file is validated!\n");

    MPI_Finalize();
    return 0;
}

/*
 *  Copyright (C) 2003, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id: ncmpidtype.h 1468 2013-10-26 16:53:18Z wkliao $ */

#ifndef NCMPI_DTYPE_H
#define NCMPI_DTYPE_H

int ncmpii_dtype_decode(MPI_Datatype dtype,
			MPI_Datatype *ptype,
			int *el_size,
			MPI_Offset *nelems,
			int *isderived,
			int *iscontig_of_ptypes);

int ncmpii_data_repack(void *inbuf,
                       MPI_Offset incount,
                       MPI_Datatype intype,
                       void *outbuf,
                       MPI_Offset outcount,
                       MPI_Datatype outtype);

#endif

!
!   Copyright (C) 2014, Northwestern University and Argonne National Laboratory
!   See COPYRIGHT notice in top-level directory.
!
! $Id: vard_int.F 2239 2015-12-18 18:21:09Z wkliao $

!
! This example shows how to use the vard API nfmpi_put_vard() and
! nfmpi_get_vard() to write and read 2D record and fixed-size
! variables.
!
!    To compile:
!        mpif77 -O2 vard_int.f -o vard_int -lpnetcdf
!
! Example commands for MPI run and outputs from running ncmpidump on
! the NC file produced by this example program:
!
!    % mpiexec -n 4 ./vard_int /pvfs2/wkliao/testfile.nc
!
! The expected results from the output file contents are:
!
!  % ncmpidump /pvfs2/wkliao/testfile.nc
!    netcdf testfile {
!    // file format: CDF-1
!    dimensions:
!           REC_DIM = UNLIMITED ; // (2 currently)
!           X = 12 ;
!           FIX_DIM = 2 ;
!    variables:
!           int rec_var(REC_DIM, X) ;
!           int fix_var(FIX_DIM, X) ;
!    data:
!
!     rec_var =
!       0, 1, 2, 100, 101, 102, 200, 201, 202, 300, 301, 302,
!       10, 11, 12, 110, 111, 112, 210, 211, 212, 310, 311, 312 ;
!
!     fix_var =
!       0, 1, 2, 100, 101, 102, 200, 201, 202, 300, 301, 302,
!       10, 11, 12, 110, 111, 112, 210, 211, 212, 310, 311, 312 ;
!    }
!
      subroutine check(err, message)
          implicit none
          include "mpif.h"
          include "pnetcdf.inc"
          integer err
          character message*(*)

          ! It is a good idea to check returned value for possible error
          if (err .NE. NF_NOERR) then
              write(6,*) message//' '//nfmpi_strerror(err)
              call MPI_Abort(MPI_COMM_WORLD, -1, err)
          end if
      end ! subroutine check

      program main
          implicit none
          include "mpif.h"
          include "pnetcdf.inc"
          character*128 filename, cmd
          integer err, ierr, nprocs, rank, i, j, get_args, dummy
          integer cmode, ncid, varid0, varid1, dimid(2)
          integer*8 NX, NY, len, bufcount
          integer*8 start(2), count(2)
          PARAMETER(NX=3, NY=2)
          integer buf(NX, NY)
          integer buftype, rec_filetype, fix_filetype
          integer array_of_sizes(2), array_of_subsizes(2)
          integer array_of_starts(2), blocklengths(2)
          integer*8 malloc_size, sum_size, recsize, two
#ifdef SIZEOF_MPI_AINT_IS_4
          integer disps(2)
#else
          integer*8 disps(2)
#endif
          logical verbose

          call MPI_Init(err)
          call MPI_Comm_rank(MPI_COMM_WORLD, rank, err)
          call MPI_Comm_size(MPI_COMM_WORLD, nprocs, err)

          two = 2
          ! take filename from command-line argument if there is any
          if (rank .EQ. 0) then
              verbose = .TRUE.
              filename = "testfile.nc"
              ierr = get_args(2, cmd, filename, verbose, dummy)
          endif
          call MPI_Bcast(ierr, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, err)
          if (ierr .EQ. 0) goto 999

          call MPI_Bcast(verbose, 1, MPI_LOGICAL, 0, MPI_COMM_WORLD,
     +                   err)
          call MPI_Bcast(filename, 256, MPI_CHARACTER, 0,
     +                   MPI_COMM_WORLD, err)

          start(1) = NX * rank
          start(2) = 0
          count(1) = NX
          count(2) = 2

          ! initialized buffer contents
          do j=1, INT(count(2))
             do i=1, INT(count(1))
                 buf(i, j) = rank*100 + j*10 + i
             enddo
          enddo

          ! create file, truncate it if exists
          cmode = NF_CLOBBER
          err = nfmpi_create(MPI_COMM_WORLD, filename, cmode,
     +                       MPI_INFO_NULL, ncid)
          call check(err, 'In nfmpi_create: ')

          ! define 2 dimensions
          err = nfmpi_def_dim(ncid, "REC_DIM", NFMPI_UNLIMITED,dimid(2))
          call check(err, 'In nfmpi_def_dim REC_DIM: ')
          len = NX * nprocs
          err = nfmpi_def_dim(ncid, "X", len, dimid(1))
          call check(err, 'In nfmpi_def_dim X: ')

          ! define 2D record variables of integer type
          err = nfmpi_def_var(ncid, "rec_var", NF_INT, 2, dimid, varid0)
          call check(err, 'In nfmpi_def_var: rec_var ')

          ! define 2D fixed-size variable of integer type
          err = nfmpi_def_dim(ncid, "FIX_DIM", two, dimid(2))
          call check(err, 'In nfmpi_def_dim RECV_DIM: ')
          err = nfmpi_def_var(ncid, "fix_var", NF_INT, 2, dimid, varid1)
          call check(err, 'In nfmpi_def_var: fix_var ')

          ! do not forget to exit define mode
          err = nfmpi_enddef(ncid)
          call check(err, 'In nfmpi_enddef: ')

          ! create a file type for the record variable */
          err = nfmpi_inq_recsize(ncid, recsize)
          call check(err, 'In nfmpi_inq_recsize: ')
          blocklengths(1) = INT(count(1))
          blocklengths(2) = INT(count(1))
          disps(1) = start(1)*4
          disps(2) = recsize + start(1)*4
          call MPI_Type_create_hindexed(2, blocklengths, disps,
     +                                  MPI_INTEGER, rec_filetype, err)
          call MPI_Type_commit(rec_filetype, err)

          ! create a file type for the fixed-size variable
          array_of_sizes(1) = INT(NX*nprocs)
          array_of_sizes(2) = 2
          array_of_subsizes(1) = INT(count(1))
          array_of_subsizes(2) = INT(count(2))
          array_of_starts(1) = INT(start(1))
          array_of_starts(2) = INT(start(2))
          call MPI_Type_create_subarray(2, array_of_sizes,
     +         array_of_subsizes, array_of_starts, MPI_ORDER_FORTRAN,
     +         MPI_INTEGER, fix_filetype, err)
          call MPI_Type_commit(fix_filetype, err)

          bufcount = count(1) * count(2)
          buftype = MPI_INTEGER

          ! write the record variable */
          err = nfmpi_put_vard_all(ncid, varid0, rec_filetype, buf,
     +                             bufcount, buftype)
          call check(err, 'In nfmpi_put_vard_all: ')

          ! check if the number of records changed to 2
          err = nfmpi_inq_unlimdim(ncid, dimid(2))
          call check(err, 'In nfmpi_inq_unlimdim: ')
          err = nfmpi_inq_dimlen(ncid, dimid(2), len)
          call check(err, 'In nfmpi_inq_dimlen: ')
          if (len .NE. 2) then
              print*, 'Error: number of records should be 2 but got ',
     +                 len
          endif

          ! write the fixed-size variable
          err = nfmpi_put_vard_all(ncid, varid1, fix_filetype, buf,
     +                             bufcount, buftype)
          call check(err, 'In nfmpi_put_vard_all: ')

          ! close the file
          err = nfmpi_close(ncid)
          call check(err, 'In nfmpi_close: ')

          ! open the same file and read back for validate */
          err = nfmpi_open(MPI_COMM_WORLD, filename, NF_NOWRITE,
     +                     MPI_INFO_NULL, ncid)
          call check(err, 'In nfmpi_open: ')

          err = nfmpi_inq_varid(ncid, "rec_var", varid0)
          call check(err, 'In nfmpi_inq_varid rec_var: ')
          err = nfmpi_inq_varid(ncid, "fix_var", varid1)
          call check(err, 'In nfmpi_inq_varid fix_var: ')

          ! PnetCDF start() argument starts with 1
          start(1) = start(1) + 1
          start(2) = start(2) + 1

          ! read back record variable
          err = nfmpi_get_vard_all(ncid, varid0, rec_filetype, buf,
     +                             bufcount, buftype)
          call check(err, 'In nfmpi_get_vard_all: ')

          ! read back fixed-size variable
          err = nfmpi_get_vard_all(ncid, varid1, fix_filetype, buf,
     +                             bufcount, buftype)
          call check(err, 'In nfmpi_get_vard_all: ')

          ! close the file
          err = nfmpi_close(ncid)
          call check(err, 'In nfmpi_close: ')

          call MPI_Type_free(rec_filetype, err)
          call MPI_Type_free(fix_filetype, err)

          ! check if there is any PnetCDF internal malloc residue
 998      format(A,I13,A)
          err = nfmpi_inq_malloc_size(malloc_size)
          if (err .EQ. NF_NOERR) then
              call MPI_Reduce(malloc_size, sum_size, 1, MPI_OFFSET,
     +                        MPI_SUM, 0, MPI_COMM_WORLD, err)
              if (rank .EQ. 0 .AND. sum_size .GT. 0)
     +            print 998,
     +            'heap memory allocated by PnetCDF internally has ',
     +            sum_size/1048576, ' MiB yet to be freed'
          endif

 999      call MPI_Finalize(err)
          ! call EXIT(0) ! EXIT() is a GNU extension
      end ! program main


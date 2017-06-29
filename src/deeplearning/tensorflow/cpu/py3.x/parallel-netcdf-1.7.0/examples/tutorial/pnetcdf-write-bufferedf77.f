!
!   Copyright (C) 2012, Northwestern University and Argonne National Laboratory
!   See COPYRIGHT notice in top-level directory.
!
! $Id: pnetcdf-write-bufferedf77.f 2245 2015-12-20 18:39:52Z wkliao $

      program main

      implicit none
      include "mpif.h"
      include "pnetcdf.inc"

      integer i, j, ncid, varid, cmode, err, rank, nprocs
      integer dimid(2), req(2), status(2)
      integer*8 start(2)
      integer*8 count(2)
      integer*8 stride(2)
      integer*8 imap(2)
      integer*8 bufsize
      integer*8 put_size, dim_size
      real  var(6,4)
      character*256 filename

      call MPI_INIT(err)
      call MPI_COMM_RANK(MPI_COMM_WORLD, rank, err)
      call MPI_COMM_SIZE(MPI_COMM_WORLD, nprocs, err)

      filename = "testfile.nc"
      cmode = IOR(NF_CLOBBER, NF_64BIT_DATA)
      err = nfmpi_create(MPI_COMM_WORLD, filename, cmode,
     +                   MPI_INFO_NULL, ncid)
      if (err .NE. NF_NOERR) print*,'Error at nfmpi_create ',
     +                           nfmpi_strerror(err)

      ! define a variable of a (4*nprocs) x 6 integer array in the nc file
      dim_size = 4
      err = nfmpi_def_dim(ncid, 'X', dim_size*nprocs, dimid(1))
      if (err .NE. NF_NOERR) print*,'Error at nfmpi_def_dim ',
     +                           nfmpi_strerror(err)

      dim_size = 6
      err = nfmpi_def_dim(ncid, 'Y', dim_size, dimid(2))
      if (err .NE. NF_NOERR) print*,'Error at nfmpi_def_dim ',
     +                           nfmpi_strerror(err)

      err = nfmpi_def_var(ncid, 'var', NF_INT64, 2, dimid, varid)
      if (err .NE. NF_NOERR) print*,'Error at nfmpi_def_var ',
     +                           nfmpi_strerror(err)

      err = nfmpi_enddef(ncid)
      if (err .NE. NF_NOERR) print*,'Error at nfmpi_enddef ',
     +                           nfmpi_strerror(err)

      ! set the contents of the local write buffer var, a 4 x 6 real array
      ! for example, for rank == 2, var(4,6) =
      !     48, 54, 60, 65,
      !     49, 55, 61, 67,
      !     50, 56, 62, 68,
      !     51, 57, 63, 69,
      !     52, 58, 64, 70,
      !     53, 59, 65, 71
      do j = 1, 4
         do i = 1, 6
            var(i,j) = (j-1)*6+(i-1) + rank*24
         enddo
      enddo

      ! bufsize must be max of data type converted before and after
      bufsize = 4*6*8
      err = nfmpi_buffer_attach(ncid, bufsize)
      if (err .NE. NF_NOERR) print*,'Error at nfmpi_buffer_attach ',
     +                           nfmpi_strerror(err)

      ! write var to the NC variable in the matrix transposed way
      count(1)  = 2
      count(2)  = 6
      stride(1) = 1
      stride(2) = 1
      imap(1)   = 6
      imap(2)   = 1

      req(1) = NF_REQ_NULL  ! actually not necessary, added for testing
      req(2) = NF_REQ_NULL  ! actually not necessary, added for testing

      ! write to the 1st two columns of the variable in matrix transposed way
      start(1)  = 1 + rank*4
      start(2)  = 1
      err = nfmpi_bput_varm_real(ncid, varid, start, count, stride,
     +                           imap, var(1,1), req(1))
      if (err .NE. NF_NOERR) print*,'Error at nfmpi_bput_varm_real ',
     +                           nfmpi_strerror(err)

      ! write to the 2nd two columns of the variable in transposed way
      start(1)  = 3 + rank*4
      start(2)  = 1
      err = nfmpi_bput_varm_real(ncid, varid, start, count, stride,
     +                           imap, var(1,3), req(2))
      if (err .NE. NF_NOERR) print*,'Error at nfmpi_bput_varm_real ',
     +                           nfmpi_strerror(err)

      err = nfmpi_wait_all(ncid, 2, req, status)
      if (err .NE. NF_NOERR) print*,'Error at nfmpi_wait_all ',
     +                           nfmpi_strerror(err)

      ! check each bput status
      do i = 1, 2
          if (status(i) .ne. NF_NOERR) then
              print*,'Error at bput status ', nfmpi_strerror(status(i))
          endif
      enddo

      err = nfmpi_buffer_detach(ncid)
      if (err .NE. NF_NOERR) print*,'Error at nfmpi_buffer_detach ',
     +                           nfmpi_strerror(err)

      ! The output from command "ncmpidump test.nc" is shown below if run
      ! this example on 4 processes.
      !
      ! netcdf test {
      ! // file format: CDF-5 (big variables)
      ! dimensions:
      !        Y = 6 ;
      !        X = 16 ;
      ! variables:
      !        int64 var(Y, X) ;
      !data:
      !
      ! var =
      !  0,  6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90,
      !  1,  7, 13, 19, 25, 31, 37, 43, 49, 55, 61, 67, 73, 79, 85, 91,
      !  2,  8, 14, 20, 26, 32, 38, 44, 50, 56, 62, 68, 74, 80, 86, 92,
      !  3,  9, 15, 21, 27, 33, 39, 45, 51, 57, 63, 69, 75, 81, 87, 93,
      !  4, 10, 16, 22, 28, 34, 40, 46, 52, 58, 64, 70, 76, 82, 88, 94,
      !  5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65, 71, 77, 83, 89, 95 ;
      !
      ! note that the display of ncmpidump is in C array dimensional order

      err = nfmpi_inq_put_size(ncid, put_size)
      if (err .NE. NF_NOERR) print*,'Error at nfmpi_inq_put_size ',
     +                           nfmpi_strerror(err)
      ! print*,'pnetcdf reports total put size by this proc =', put_size

      err = nfmpi_close(ncid)
      if (err .NE. NF_NOERR) print*,'Error at nfmpi_close ',
     +                           nfmpi_strerror(err)

      CALL MPI_Finalize(err)
      end ! program


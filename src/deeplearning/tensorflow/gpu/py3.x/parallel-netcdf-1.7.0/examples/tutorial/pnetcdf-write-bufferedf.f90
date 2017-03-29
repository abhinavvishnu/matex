!
!   Copyright (C) 2012, Northwestern University and Argonne National Laboratory
!   See COPYRIGHT notice in top-level directory.
!
! $Id: pnetcdf-write-bufferedf.f90 2245 2015-12-20 18:39:52Z wkliao $

      program main

      use mpi
      use pnetcdf
      implicit none

      integer i, j, ncid, varid, cmode, err, rank, nprocs
      integer dimid(2), req(2), status(2)
      integer(kind=MPI_OFFSET_KIND) start(2)
      integer(kind=MPI_OFFSET_KIND) count(2)
      integer(kind=MPI_OFFSET_KIND) stride(2)
      integer(kind=MPI_OFFSET_KIND) imap(2)
      integer(kind=MPI_OFFSET_KIND) bufsize
      integer(kind=MPI_OFFSET_KIND) put_size
      real  var(6,4)
      character(len=256) filename

      call MPI_INIT(err)
      call MPI_COMM_RANK(MPI_COMM_WORLD, rank, err)
      call MPI_COMM_SIZE(MPI_COMM_WORLD, nprocs, err)

      filename = "testfile.nc"
      cmode = IOR(NF90_CLOBBER, NF90_64BIT_DATA)
      err = nf90mpi_create(MPI_COMM_WORLD, filename, cmode, &
                         MPI_INFO_NULL, ncid)
      if (err < NF90_NOERR) print*,'Error at nf90mpi_create ', &
                                   nf90mpi_strerror(err)

      ! define a variable of a (4*nprocs) x 6 integer array in the nc file
      err = nf90mpi_def_dim(ncid, 'X', 4_MPI_OFFSET_KIND*nprocs, dimid(1))
      if (err < NF90_NOERR) print*,'Error at nf90mpi_def_dim ', &
                                   nf90mpi_strerror(err)

      err = nf90mpi_def_dim(ncid, 'Y', 6_MPI_OFFSET_KIND, dimid(2))
      if (err < NF90_NOERR) print*,'Error at nf90mpi_def_dim ', &
                                   nf90mpi_strerror(err)

      err = nf90mpi_def_var(ncid, 'var', NF90_INT64, dimid, varid)
      if (err < NF90_NOERR) print*,'Error at nf90mpi_def_var ', &
                                   nf90mpi_strerror(err)

      err = nf90mpi_enddef(ncid)
      if (err < NF90_NOERR) print*,'Error at nf90mpi_enddef ', &
                                   nf90mpi_strerror(err)

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
      err = nf90mpi_buffer_attach(ncid, bufsize)
      if (err < NF90_NOERR) print*,'Error at nf90mpi_buffer_attach ', &
                                   nf90mpi_strerror(err)

      ! write var to the NC variable in the matrix transposed way
      count(1)  = 2
      count(2)  = 6
      stride(1) = 1
      stride(2) = 1
      imap(1)   = 6
      imap(2)   = 1

      req(:) = NF90_REQ_NULL  ! actually not necessary, added for testing

      ! write to the 1st two columns of the variable in matrix transposed way
      start(1)  = 1 + rank*4
      start(2)  = 1
      err = nf90mpi_bput_var(ncid, varid, var(1:,1:), req(1), &
                             start, count, stride, imap)
      if (err < NF90_NOERR) print*,'Error at nf90mpi_bput_varm_real ', &
                                   nf90mpi_strerror(err)

      ! write to the 2nd two columns of the variable in transposed way
      start(1)  = 3 + rank*4
      start(2)  = 1
      err = nf90mpi_bput_var(ncid, varid, var(1:,3:), req(2), &
                             start, count, stride, imap)
      if (err < NF90_NOERR) print*,'Error at nf90mpi_bput_varm_real ', &
                                   nf90mpi_strerror(err)

      err = nf90mpi_wait_all(ncid, 2, req, status)
      if (err < NF90_NOERR) print*,'Error at nf90mpi_wait_all ', &
                                   nf90mpi_strerror(err)

      ! check each bput status
      do i = 1, 2
          if (status(i) .ne. NF90_NOERR) then
              print*,'Error at bput status ', &
                     nf90mpi_strerror(status(i))
          endif
      enddo

      err = nf90mpi_buffer_detach(ncid)
      if (err < NF90_NOERR) print*,'Error at nf90mpi_buffer_detach ', &
                                   nf90mpi_strerror(err)

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

      err = nf90mpi_inq_put_size(ncid, put_size)
      if (err < NF90_NOERR) print*,'Error at nf90mpi_inq_put_size ', &
                                   nf90mpi_strerror(err)
      ! print*,'pnetcdf reports total put size by this proc =', put_size

      err = nf90mpi_close(ncid)
      if (err < NF90_NOERR) print*,'Error at nf90mpi_close ', &
                                   nf90mpi_strerror(err)

      CALL MPI_Finalize(err)
      end program


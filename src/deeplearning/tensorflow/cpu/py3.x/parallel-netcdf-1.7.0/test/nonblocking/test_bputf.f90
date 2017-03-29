!
!   Copyright (C) 2012, Northwestern University
!   See COPYRIGHT notice in top-level directory.
!
!   $Id: test_bputf.f90 2131 2015-09-25 22:33:12Z wkliao $

      program main

      use mpi
      use pnetcdf
      implicit none

      logical verbose
      integer i, j, ncid, varid, err, ierr, rank, nprocs, info
      integer no_err, cmode, get_args
      integer dimid(2), req(2), status(2)
      integer(kind=MPI_OFFSET_KIND) start(2)
      integer(kind=MPI_OFFSET_KIND) count(2)
      integer(kind=MPI_OFFSET_KIND) stride(2)
      integer(kind=MPI_OFFSET_KIND) imap(2)
      integer(kind=MPI_OFFSET_KIND) bufsize
      real  var(6,4)
      character(len=256) :: filename, cmd, msg

      call MPI_INIT(ierr)
      call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)
      call MPI_COMM_SIZE(MPI_COMM_WORLD, nprocs, ierr)

      if (rank .EQ. 0) then
          filename = "testfile.nc"
          err = get_args(cmd, filename)
      endif
      call MPI_Bcast(err, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
      if (err .EQ. 0) goto 999

      call MPI_Bcast(filename, 256, MPI_CHARACTER, 0, MPI_COMM_WORLD, ierr)

      verbose = .FALSE.
      if (nprocs > 1 .AND. rank .EQ. 0 .AND. verbose) then
          print*,'Warning: ',trim(cmd), &
                 ' is designed to run on 1 process'
      endif

      call MPI_Info_create(info, ierr)
      ! call MPI_Info_set(info, "romio_pvfs2_posix_write","enable",ierr)

      cmode = IOR(NF90_CLOBBER, NF90_64BIT_DATA)
      err = nf90mpi_create(MPI_COMM_WORLD, filename, cmode,  &
                           info, ncid)
      if (err < NF90_NOERR) print*,'Error at nf90mpi_create ', &
                                   nf90mpi_strerror(err)

      call MPI_Info_free(info, ierr)

      ! define a variable of a 4 x 6 integer array in the nc file
      err = nf90mpi_def_dim(ncid, 'X', 4_MPI_OFFSET_KIND, dimid(1))
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

      ! set the contents of write buffer var, a 6 x 4 real array
      !     50, 56, 62, 68,
      !     51, 57, 63, 69,
      !     52, 58, 64, 70,
      !     53, 59, 65, 71,
      !     54, 60, 66, 72,
      !     55, 61, 67, 73
      do j = 1, 4
         do i = 1, 6
            var(i,j) = (j-1)*6+(i-1) + 50
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
      imap(2)   = 1   ! imap would be {1, 4} if not transposing

      if (rank .GT. 0) then
         count(1)  = 0
         count(2)  = 0
      endif

      ! write the first two columns of the NC variable in the matrix transposed way
      start(1)  = 1
      start(2)  = 1
      err = nf90mpi_bput_var(ncid, varid, var(1:,1:), req(1), start, count, &
                             stride, imap)
      if (err < NF90_NOERR) print*,'Error at nf90mpi_bput_var', &
                                   nf90mpi_strerror(err)

      ! write the second two columns of the NC variable in the matrix transposed way
      start(1)  = 3
      start(2)  = 1
      err = nf90mpi_bput_var(ncid, varid, var(1:,3:), req(2), start, count, &
                             stride, imap)
      if (err < NF90_NOERR) print*,'Error at nf90mpi_bput_var', &
                                   nf90mpi_strerror(err)

      err = nf90mpi_wait_all(ncid, 2, req, status)
      if (err < NF90_NOERR) print*,'Error at nf90mpi_wait_all ', &
                                   nf90mpi_strerror(err)

      ! check each bput status
      do i = 1, 2
          if (status(i) .ne. NF90_NOERR) then
              print*,'Error at bput status ', nf90mpi_strerror(status(i))
          endif
      enddo

      err = nf90mpi_buffer_detach(ncid)
      if (err < NF90_NOERR) print*,'Error at nf90mpi_buffer_detach ', &
                                   nf90mpi_strerror(err)

      ! the output from command "ncmpidump -v var test.nc" should be:
      !      var =
      !       50, 56, 62, 68,
      !       51, 57, 63, 69,
      !       52, 58, 64, 70,
      !       53, 59, 65, 71,
      !       54, 60, 66, 72,
      !       55, 61, 67, 73 ;
      ! note that the display of ncmpidump is in C array dimensional order

      ! check if the contents of write buffer have been altered (should not be)
      no_err = 0
      if (rank .EQ. 0) then
         do j = 1, 4
            do i = 1, 6
               if (var(i,j) .NE. (j-1)*6+(i-1) + 50) then
! #ifdef PRINT_ERR_ON_SCREEN
!                  ! this error is a pntecdf internal error, if occurs */
!                  print*, &
!                  'Error: nf90mpi_bput_var write buffer has been altered at j=', &
!                  j,' i=',i,' var=',var(i,j)
! #endif
                   no_err = no_err + 1
                endif
            enddo
         enddo
      endif

      err = nf90mpi_close(ncid)
      if (err < NF90_NOERR) print*,'Error at nf90mpi_close ', &
                                   nf90mpi_strerror(err)

      if (rank .EQ. 0) then
          msg = '*** TESTING F90 '//trim(cmd)//' for bput_var'
          call pass_fail(no_err, msg)
      endif

 999  CALL MPI_Finalize(ierr)

      end program


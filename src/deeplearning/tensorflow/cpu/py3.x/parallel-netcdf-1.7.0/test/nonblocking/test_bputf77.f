!
!   Copyright (C) 2012, Northwestern University and Argonne National Lab
!   See COPYRIGHT notice in top-level directory.
!
!   $Id: test_bputf77.f 2224 2015-12-16 06:10:36Z wkliao $
!

       INTEGER FUNCTION XTRIM(STRING)
           CHARACTER*(*) STRING
           INTEGER I, N
           N = LEN(STRING)
           DO I = N, 1, -1
              IF (STRING(I:I) .NE. ' ') GOTO 10
           ENDDO
 10        XTRIM = I
       END ! FUNCTION XTRIM

      program main

      implicit none
      include "mpif.h"
      include "pnetcdf.inc"

      logical verbose
      integer i, j, ncid, varid, err, ierr, rank, nprocs, info
      integer no_err, cmode, get_args, XTRIM
      integer dimid(2), req(2), status(2)
      integer*8 start(2)
      integer*8 count(2)
      integer*8 stride(2)
      integer*8 imap(2)
      integer*8 bufsize, dim_size
      real  var(6,4)
      character*256 filename, cmd, msg

      call MPI_INIT(ierr)
      call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)
      call MPI_COMM_SIZE(MPI_COMM_WORLD, nprocs, ierr)

      if (rank .EQ. 0) then
          filename = "testfile.nc"
          err = get_args(cmd, filename)
      endif
      call MPI_Bcast(err, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
      if (err .EQ. 0) goto 999

      call MPI_Bcast(filename, 256, MPI_CHARACTER, 0, MPI_COMM_WORLD,
     +               ierr)

      verbose = .FALSE.
      if (nprocs .GT. 1 .AND. rank .EQ. 0 .AND. verbose) then
          print*,'Warning: ',cmd(1:XTRIM(cmd)),
     +           ' is designed to run on 1 process'
      endif

      call MPI_Info_create(info, ierr)
      ! call MPI_Info_set(info, "romio_pvfs2_posix_write","enable",ierr)

      cmode = IOR(NF_CLOBBER, NF_64BIT_DATA)
      err = nfmpi_create(MPI_COMM_WORLD, 'testfile.nc', cmode,
     +                   info, ncid)
      if (err .NE. NF_NOERR) print*,'Error at nfmpi_create ',
     +                           nfmpi_strerror(err)

      call MPI_Info_free(info, ierr)

      ! define a variable of a 4 x 6 integer array in the nc file
      dim_size = 4
      err = nfmpi_def_dim(ncid, 'X', dim_size, dimid(1))
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
      err = nfmpi_buffer_attach(ncid, bufsize)
      if (err .NE. NF_NOERR) print*,'Error at nfmpi_buffer_attach ',
     +                           nfmpi_strerror(err)

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
      err = nfmpi_bput_varm_real(ncid, varid, start, count, stride,
     +                           imap, var(1,1), req(1))
      if (err .NE. NF_NOERR) print*,'Error at nfmpi_bput_varm_real ',
     +                           nfmpi_strerror(err)

      ! write the second two columns of the NC variable in the matrix transposed way
      start(1)  = 3
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
!                  'Error: bput_varm write buffer has been altered at j=', &
!                  j,' i=',i,' var=',var(i,j)
! #endif
                   no_err = no_err + 1
                endif
            enddo
         enddo
      endif

      err = nfmpi_close(ncid)
      if (err .NE. NF_NOERR) print*,'Error at nfmpi_close ',
     +                           nfmpi_strerror(err)

      if (rank .EQ. 0) then
         msg = '*** TESTING F77 '//cmd(1:XTRIM(cmd))//
     +         ' for bput_varm_real API'
         call pass_fail(no_err, msg)
      endif

 999  CALL MPI_Finalize(ierr)

      end ! program


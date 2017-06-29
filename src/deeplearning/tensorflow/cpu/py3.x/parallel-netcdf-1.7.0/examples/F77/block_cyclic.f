!
!  Copyright (C) 2012, Northwestern University and Argonne National Laboratory
!  See COPYRIGHT notice in top-level directory.
!
! $Id: block_cyclic.f 2325 2016-02-28 07:49:13Z wkliao $


! This example, generalized from column_wise.f, makes a number of nonblocking
! API calls, each writes a block of columns into a 2D integer array. In other
! words, the I/O pattern is a blocked cyclic along X dimension.
!
! Each process writes NX rows in total. The block length is controlled by
! block_len. In this example, block_len is set to 2. Blocks are layout in a
! cyclic fashion in the file. This example can test if PnetCDF can coalesce
! file offsets and lengths when constructing a merged filetype.
! 
! The compile and run commands are given below, together with an ncmpidump of
! the output file. In this example, block_len = 2.
!
!    % mpif77 -O2 -o block_cyclic block_cyclic.f -lpnetcdf
!    % mpiexec -n 4 ./block_cyclic /pvfs2/wkliao/testfile.nc
!
!    % ncmpidump /pvfs2/wkliao/testfile.nc
!    netcdf testfile {
!    // file format: CDF-5 (big variables)
!    dimensions:
!            Y = 10 ;
!            X = 16 ;
!    variables:
!            int var(Y, X) ;
!    data:
!
!     var =
!      10, 10, 11, 11, 12, 12, 13, 13, 10, 10, 11, 11, 12, 12, 13, 13,
!      10, 10, 11, 11, 12, 12, 13, 13, 10, 10, 11, 11, 12, 12, 13, 13,
!      10, 10, 11, 11, 12, 12, 13, 13, 10, 10, 11, 11, 12, 12, 13, 13,
!      10, 10, 11, 11, 12, 12, 13, 13, 10, 10, 11, 11, 12, 12, 13, 13,
!      10, 10, 11, 11, 12, 12, 13, 13, 10, 10, 11, 11, 12, 12, 13, 13,
!      10, 10, 11, 11, 12, 12, 13, 13, 10, 10, 11, 11, 12, 12, 13, 13,
!      10, 10, 11, 11, 12, 12, 13, 13, 10, 10, 11, 11, 12, 12, 13, 13,
!      10, 10, 11, 11, 12, 12, 13, 13, 10, 10, 11, 11, 12, 12, 13, 13,
!      10, 10, 11, 11, 12, 12, 13, 13, 10, 10, 11, 11, 12, 12, 13, 13,
!      10, 10, 11, 11, 12, 12, 13, 13, 10, 10, 11, 11, 12, 12, 13, 13 ;
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
          endif
      end ! subroutine check

      program main
          implicit none
          include "mpif.h"
          include "pnetcdf.inc"

          integer NY, NX
          PARAMETER(NX=10, NY=4)

          character*128 filename, cmd
          integer i, j, rank, nprocs, err, ierr, num_reqs, get_args
          integer ncid, cmode, varid, dimid(2), stride, block_len
          integer buf(NX, NY)
          integer reqs(NY), sts(NY)
          integer*8 G_NY, myOff, block_start,
     +                                  global_nx, global_ny
          integer*8 start(2), count(2)
          integer*8 malloc_size, sum_size
          logical verbose
          integer dummy, info

          call MPI_Init(err)
          call MPI_Comm_rank(MPI_COMM_WORLD, rank, err)
          call MPI_Comm_size(MPI_COMM_WORLD, nprocs, err)

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

          ! set an MPI-IO hint to disable file offset alignment for fix-sized
          ! variables
          call MPI_Info_create(info, err)
          call MPI_Info_set(info, "nc_var_align_size", "1", err)

          ! create file, truncate it if exists
          cmode = IOR(NF_CLOBBER, NF_64BIT_DATA)
          err = nfmpi_create(MPI_COMM_WORLD, filename, cmode,
     +                       info, ncid)
          call check(err, 'In nfmpi_create: ')

          call MPI_Info_free(info, err)

          ! the global array is NX * (NY * nprocs) */
          G_NY  = NY * nprocs
          myOff = NY * rank

          ! define dimensions
          global_nx = NX
          err = nfmpi_def_dim(ncid, 'Y', global_nx, dimid(2))
          call check(err, 'In nfmpi_def_dim Y')

          err = nfmpi_def_dim(ncid, 'X', G_NY, dimid(1))
          call check(err, 'In nfmpi_def_dim X')

          err = nfmpi_def_var(ncid, 'var', NF_INT, 2, dimid, varid)
          call check(err, 'In nfmpi_def_var var')

          ! do not forget to exit define mode
          err = nfmpi_enddef(ncid)
          call check(err, 'In nfmpi_enddef: ')

          ! First, fill the entire array with zeros, using a blocking I/O.
          ! Every process writes a subarray of size NX * NY
          do i=1, NY 
          do j=1, NX
             buf(j,i) = 0
          enddo
          enddo
          start(1) = myOff+1
          start(2) = 1
          count(1) = NY
          count(2) = NX
          err = nfmpi_put_vara_int_all(ncid, varid, start, count, buf)
          call check(err, 'In nfmpi_put_vara_int_all: ')

          ! initialize the buffer with rank ID
          do i=1, NY 
          do j=1, NX
             buf(j,i) = rank+10
          enddo
          enddo

          ! each proc writes NY columns of the 2D array, block_len controls the
          ! the number of contiguous columns at a time
          block_start = 0
          block_len   = 2  ! can be 1, 2, 3, ..., NY
          if (block_len .GT. NY) block_len = NY

          start(1) = rank * block_len + 1
          start(2) = 1
          count(1) = 1
          count(2) = NX
          num_reqs = 0

          do i=1, NY
             num_reqs = num_reqs + 1
             err = nfmpi_iput_vara_int(ncid, varid, start, count,
     +                                 buf(1,i), reqs(num_reqs))
             call check(err, 'In nfmpi_iput_vara_int: ')

             if (MOD(i, block_len) .EQ. 0) then
                 stride = NY-i
                 if (stride .GT. block_len) stride = block_len
                 block_start = block_start + block_len * nprocs
                 start(1) = block_start + stride * rank + 1
             else
                 start(1) = start(1) + 1
             endif
          enddo

          err = nfmpi_wait_all(ncid, num_reqs, reqs, sts)
          call check(err, 'In nfmpi_wait_all: ')

          ! check status of all requests
          do i=1, num_reqs
             if (sts(i) .NE. NF_NOERR) then
                 print*, "Error: nonblocking write fails on request",
     +                   i, ' ', nfmpi_strerror(sts(i))
             endif
          enddo

          err = nfmpi_close(ncid)
          call check(err, 'In nfmpi_close: ')

          ! open an existing file created earlier for read
          err = nfmpi_open(MPI_COMM_WORLD, filename, NF_NOWRITE,
     +                     MPI_INFO_NULL, ncid)
          call check(err, 'In nfmpi_open: ')

          ! the global array is NX * (NY * nprocs) */
          err = nfmpi_inq_dimid(ncid, "X", dimid(1))
          call check(err, 'In nfmpi_inq_dimid X: ')
          err = nfmpi_inq_dimlen(ncid, dimid(1), global_ny)
          call check(err, 'In nfmpi_inq_dimlen: ')
          ! G_NY must == NY * nprocs
          ! myNY must == global_ny / nprocs

          err = nfmpi_inq_dimid(ncid, "Y", dimid(2))
          call check(err, 'In nfmpi_inq_dimid Y: ')
          err = nfmpi_inq_dimlen(ncid, dimid(2), global_nx)
          call check(err, 'In nfmpi_inq_dimlen: ')
          ! global_nx must == NX

          err = nfmpi_inq_varid(ncid, "var", varid)
          call check(err, 'In nfmpi_inq_varid: ')

          ! initialize the buffer with -1, so a read error can be pinpointed
          do i=1, NY 
          do j=1, NX
             buf(j,i) = -1
          enddo
          enddo

          ! each proc reads NY columns of the 2D array, block_len controls the
          ! the number of contiguous columns at a time */
          block_start = 0
          block_len   = 2  ! can be 1, 2, 3, ..., NY
          if (block_len .GT. NY) block_len = NY

          start(1) = rank * block_len + 1
          start(2) = 1
          count(1) = 1
          count(2) = global_nx
          num_reqs = 0

          do i=1, NY
             num_reqs = num_reqs + 1
             err = nfmpi_iget_vara_int(ncid, varid, start, count,
     +                                 buf(1,i), reqs(num_reqs))
             call check(err, 'In nfmpi_iget_vara_int: ')

             if (MOD(i, block_len) .EQ. 0) then
                 stride = NY-i
                 if (stride .GT. block_len) stride = block_len
                 block_start = block_start + block_len * nprocs
                 start(1) = block_start + stride * rank + 1
             else
                 start(1) = start(1) + 1
             endif
          enddo

          err = nfmpi_wait_all(ncid, num_reqs, reqs, sts)
          call check(err, 'In nfmpi_wait_all: ')

          ! check status of all requests
          do i=1, num_reqs
             if (sts(i) .NE. NF_NOERR) then
                 print*, "Error: nonblocking write fails on request",
     +                   i, ' ', nfmpi_strerror(sts(i))
             endif
          enddo

          err = nfmpi_close(ncid)
          call check(err, 'In nfmpi_close: ')

          ! check the read contents */
          do j=1, NY
             do i=1, NX
                  if (buf(i,j) .NE. rank+10) then
                      print*, 'Read contents mismatch at buf i=', i,
     +                        ' j=',j,' =', buf(i,j),' (should be ',
     +                        rank+10, ')\n'
                  endif
             enddo
          enddo

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

!
!
!  Copyright (C) 2012, Northwestern University and Argonne National
!  Laboratory
!  See COPYRIGHT notice in top-level directory.
!
! $Id: hints.f90 2139 2015-10-08 01:59:10Z wkliao $

!
! This example sets two PnetCDF hints:
!    nc_header_align_size and nc_var_align_size
! and prints the hint values as well as the header size, header extent, and
! two variables' starting file offsets.
!
! The compile and run commands are given below.
!
!    % mpif90 -O2 -o hints hints.f90 -lpnetcdf
!
!    % mpiexec -n 4 ./hints /pvfs2/wkliao/testfile.nc
!
!    nc_header_align_size      set to = 1024
!    nc_var_align_size         set to = 512
!    nc_header_read_chunk_size set to = 256
!    header size                      = 252
!    header extent                    = 1024
!    var_zy start file offset         = 1024
!    var_yx start file offset         = 3072
!


      subroutine check(err, message)
          use mpi
          use pnetcdf
          implicit none
          integer err
          character(len=*) message

          ! It is a good idea to check returned value for possible error
          if (err .NE. NF90_NOERR) then
              write(6,*) trim(message), trim(nf90mpi_strerror(err))
              call MPI_Abort(MPI_COMM_WORLD, -1, err)
          end if
      end subroutine check

      subroutine print_hints(ncid, varid0, varid1)
          use mpi
          use pnetcdf
          implicit none

          integer ncid, varid0, varid1

          ! local variables
          character*(MPI_MAX_INFO_VAL) value
          integer err, len, info_used
          logical flag
          integer(kind=MPI_OFFSET_KIND) header_size, header_extent
          integer(kind=MPI_OFFSET_KIND) var_zy_start, var_yx_start
          integer(kind=MPI_OFFSET_KIND) h_align, v_align, h_chunk

          h_align=-1
          v_align=-1
          h_chunk=-1

          err = nf90mpi_inq_header_size  (ncid, header_size)
          call check(err, 'In nf90mpi_inq_header_size: ')
          err = nf90mpi_inq_header_extent(ncid, header_extent)
          call check(err, 'In nf90mpi_inq_header_extent: ')
          err = nf90mpi_inq_varoffset(ncid, varid0, var_zy_start)
          call check(err, 'In nf90mpi_inq_varoffset varid0: ')
          err = nf90mpi_inq_varoffset(ncid, varid1, var_yx_start)
          call check(err, 'In nf90mpi_inq_varoffset varid1: ')

          err = nf90mpi_inq_file_info(ncid, info_used)
          call check(err, 'In nf90mpi_inq_file_info : ')

          call MPI_Info_get_valuelen(info_used, "nc_header_align_size", &
                                     len, flag, err)
          if (flag) then
              call MPI_Info_get(info_used, "nc_header_align_size", &
                                len+1, value, flag, err)
              read(value, '(i16)') h_align
          endif
          call MPI_Info_get_valuelen(info_used, "nc_var_align_size", &
                                     len, flag, err)
          if (flag) then
              call MPI_Info_get(info_used, "nc_var_align_size", len+1, &
                                value, flag, err)
              read(value, '(i16)') v_align
          endif
          call MPI_Info_get_valuelen(info_used, &
                                     "nc_header_read_chunk_size", &
                                     len, flag, err)
          if (flag) then
              call MPI_Info_get(info_used, "nc_header_read_chunk_size", &
                                len+1, value, flag, err)
              read(value, '(i16)') h_chunk
          endif
          call MPI_Info_free(info_used, err)

          if (h_align .EQ. -1) then
              print*,"nc_header_align_size      is NOT set"
          else
              print*,"nc_header_align_size      set to = ", h_align
          endif
          if (v_align .EQ. -1) then
              print*,"nc_var_align_size         is NOT set"
          else
              print*,"nc_var_align_size         set to = ", v_align
          endif
          if (h_chunk .EQ. -1) then
              print*,"nc_header_read_chunk_size is NOT set"
          else
              print*,"nc_header_read_chunk_size set to = ", h_chunk
          endif

          print*,"header size                      = ", header_size
          print*,"header extent                    = ", header_extent
          print*,"var_zy start file offset         = ", var_zy_start
          print*,"var_yx start file offset         = ", var_yx_start
      end subroutine print_hints

      subroutine put_vara(ncid, varid, ny, nx, start, count)
          use mpi
          use pnetcdf
          implicit none

          integer ncid, varid, ny, nx
          integer(KIND=MPI_OFFSET_KIND) start(2), count(2)

          integer i, j, rank, err
          integer buf(nx, ny)

          call MPI_Comm_rank(MPI_COMM_WORLD, rank, err)

          do j=1, ny
             do i=1, nx
                buf(i,j) = rank ! j*nx + i
             enddo
          enddo

          err = nf90mpi_put_var_all(ncid, varid, buf, start, count)
          call check(err, 'In nf90mpi_put_var_all : ')

      end subroutine put_vara

      program main
      use mpi
      use pnetcdf
      implicit none

      character(len = 256) :: filename, cmd
      integer NZ, NY, NX
      integer ncid, rank, nprocs, info, cmode, err, ierr, get_args, dummy
      integer varid0, varid1, dimid(3), dimid2(2)
      integer(KIND=MPI_OFFSET_KIND) start(2), count(2), llen
      integer(kind=MPI_OFFSET_KIND) malloc_size, sum_size
      logical verbose

      call MPI_Init(err)
      call MPI_Comm_rank(MPI_COMM_WORLD, rank, err)
      call MPI_Comm_size(MPI_COMM_WORLD, nprocs, err)

      NZ = 5
      NY = 5
      NX = 5

      if (rank .EQ. 0) then
          verbose = .TRUE.
          filename = "testfile.nc"
          ierr = get_args(2, cmd, filename, verbose, dummy)
      endif
      call MPI_Bcast(ierr, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, err)
      if (ierr .EQ. 0) goto 999

      call MPI_Bcast(verbose, 1, MPI_LOGICAL, 0, MPI_COMM_WORLD, err)
      call MPI_Bcast(filename, 256, MPI_CHARACTER, 0, MPI_COMM_WORLD, err)

      call MPI_Info_create(info, err)
      call MPI_Info_set(info, "nc_header_align_size",      "1024", err)
      call MPI_Info_set(info, "nc_var_align_size",         "512",  err)
      call MPI_Info_set(info, "nc_header_read_chunk_size", "256",  err)
      ! note that set the above values to 1 to disable the alignment

      ! create a new file for writing
      cmode = NF90_CLOBBER + NF90_64BIT_DATA
      err = nf90mpi_create(MPI_COMM_WORLD, filename, cmode, info, ncid)
      call check(err, 'In nf90mpi_create : ')
      call MPI_Info_free(info, err)

      ! define 3 dimensions
      llen = NZ*nprocs
      err = nf90mpi_def_dim(ncid, "Z", llen, dimid(1))
      call check(err, 'In nf90mpi_def_dim Z : ')
      llen = NY*nprocs
      err = nf90mpi_def_dim(ncid, "Y", llen, dimid(2))
      call check(err, 'In nf90mpi_def_dim Y : ')
      llen = NX*nprocs
      err = nf90mpi_def_dim(ncid, "X", llen, dimid(3))
      call check(err, 'In nf90mpi_def_dim X : ')

      ! define a variable of size (NZ * nprocs) * (NY * nprocs)
      dimid2(1) = dimid(1)
      dimid2(2) = dimid(2)
      err = nf90mpi_def_var(ncid, "var_zy", NF90_INT,   dimid2, varid0)
      call check(err, 'In nf90mpi_def_var var_zy : ')
      ! define a variable of size (NY * nprocs) * (NX * nprocs)
      dimid2(1) = dimid(2)
      dimid2(2) = dimid(3)
      err = nf90mpi_def_var(ncid, "var_yx", NF90_FLOAT, dimid2, varid1)
      call check(err, 'In nf90mpi_def_var var_yx : ')

      err = nf90mpi_enddef(ncid)
      call check(err, 'In nf90mpi_enddef : ')

      ! var_zy is partitioned along Z dimension
      start(1) = NZ * rank + 1
      start(2) = 1
      count(1) = NZ
      count(2) = NY * nprocs
      call put_vara(ncid, varid0, NZ, NY * nprocs, start, count)

      ! var_yx is partitioned along X dimension
      start(1) = 1
      start(2) = NX * rank + 1
      count(1) = NY * nprocs
      count(2) = NX
      call put_vara(ncid, varid1, NY * nprocs, NX, start, count)

      if (rank .EQ. 0 .AND. verbose) call print_hints(ncid, varid0, varid1)

      err = nf90mpi_close(ncid)

      ! check if there is any PnetCDF internal malloc residue
 998  format(A,I13,A)
      err = nf90mpi_inq_malloc_size(malloc_size)
      if (err == NF90_NOERR) then
          call MPI_Reduce(malloc_size, sum_size, 1, MPI_OFFSET, &
                          MPI_SUM, 0, MPI_COMM_WORLD, err)
          if (rank .EQ. 0 .AND. sum_size .GT. 0_MPI_OFFSET_KIND) print 998, &
              'heap memory allocated by PnetCDF internally has ',  &
              sum_size/1048576, ' MiB yet to be freed'
      endif

 999  call MPI_Finalize(err)
      ! call EXIT(0) ! EXIT() is a GNU extension

      end program


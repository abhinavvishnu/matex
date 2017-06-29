!
!   Copyright (C) 2014, Northwestern University
!   See COPYRIGHT notice in top-level directory.
!
! $Id: transpose.f90 2139 2015-10-08 01:59:10Z wkliao $

!
!    This example shows how to use varm API to write six 3D integer
!    array variables into a file. Each variable in the file is a
!    dimensional transposed array from the one stored in memory. In
!    memory, a 3D array is partitioned among all processes in a
!    block-block-block fashion and in XYZ (i.e. Fortran) order. Note
!    the variable and dimension naming below is in Fortran order. The
!    dimension structures of the transposed six arrays are
!       integer XYZ_var(X, Y, Z)      XYZ -> XYZ (no transpose)
!       integer XZY_var(X, Z, Y)      XYZ -> XZY
!       integer YXZ_var(Y, X, Z)      XYZ -> YXZ
!       integer YZX_var(Y, Z, X)      XYZ -> YZX
!       integer ZXY_var(Z, X, Y)      XYZ -> ZXY
!       integer ZYX_var(Z, Y, X)      XYZ -> ZYX
!
!    To compile:
!        mpif90 -O2 transpose.f90 -o transpose -lpnetcdf
!    To run:
!        mpiexec -n num_processes ./transpose [filename] [len]
!    where len decides the size of local array, which is
!    (len+2) x (len+1) x len. So, each variable is of size
!    (len+2)*(len+1)*len * nprocs * sizeof(int)
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

      program main
          use mpi
          use pnetcdf
          implicit none

          character(LEN=128) filename, cmd
          integer err, nprocs, rank, lower_dims, ierr, get_args, dummy
          integer cmode, ncid, psizes(3), dimids(3), dimidsT(3)
          integer XYZ_id, XZY_id, YZX_id, YXZ_id, ZYX_id, ZXY_id
          integer(kind=MPI_OFFSET_KIND) gsizes(3), starts(3)
          integer(kind=MPI_OFFSET_KIND) counts(3), strides(3), imap(3)
          integer(kind=MPI_OFFSET_KIND) startsT(3), countsT(3)
          integer(kind=MPI_OFFSET_KIND) malloc_size, sum_size
          integer i, j, k, nx, ny, nz
          PARAMETER(nx=4, ny=3, nz=2)
          integer buf(nx,ny,nz)
          logical verbose

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

          call MPI_Bcast(verbose, 1, MPI_LOGICAL, 0, MPI_COMM_WORLD, err)
          call MPI_Bcast(filename, 256, MPI_CHARACTER, 0, MPI_COMM_WORLD, err)

          ! calculate number of processes along each dimension
          psizes = 0
          call MPI_Dims_create(nprocs, 3, psizes, err)
          if (verbose .AND. rank .EQ. 0) print*, "psizes= ",psizes(1:3)

          ! for each MPI rank, find its local rank IDs along each dimension in
          ! starts()
          lower_dims = 1
          do i=1, 3
              starts(i) = MOD(rank / lower_dims, psizes(i))
              lower_dims = lower_dims * psizes(i)
          enddo
          if (verbose) print*, "proc ",rank,": dim rank= ",starts(1:3)

          strides = 1
          gsizes  = (/ nx, ny, nz /)
          do i=1, 3
             starts(i) = starts(i) * gsizes(i) + 1 ! start indices
             counts(i) = gsizes(i)                 ! array elements
             gsizes(i) = gsizes(i) * psizes(i)     ! global array size
          enddo

          if (verbose) then
              print*, "proc ",rank,": starts= ",starts(1:3)
              print*, "proc ",rank,": counts= ",counts(1:3)
          endif

          ! initialize buffer with contiguous numbers
          do k=1, nz
          do j=1, ny
          do i=1, nx
              buf(i, j, k) = INT((starts(3)+k-2)*gsizes(2)*gsizes(1) + &
                                 (starts(2)+j-2)*gsizes(1) + &
                                 (starts(1)+i-2))
          enddo
          enddo
          enddo
          if (verbose .AND. rank .EQ. 0) print*, "buf= ",buf

          ! create file, truncate it if exists
          cmode = IOR(NF90_CLOBBER, NF90_64BIT_DATA)
          cmode = NF90_CLOBBER
          err = nf90mpi_create(MPI_COMM_WORLD, filename, cmode, &
                               MPI_INFO_NULL, ncid)
          call check(err, 'In nf90mpi_create: ')

          ! define dimensions X, Y, Z
          err = nf90mpi_def_dim(ncid, "X", gsizes(1), dimids(1))
          call check(err, 'In nf90mpi_def_dim X: ')
          err = nf90mpi_def_dim(ncid, "Y", gsizes(2), dimids(2))
          call check(err, 'In nf90mpi_def_dim Y: ')
          err = nf90mpi_def_dim(ncid, "Z", gsizes(3), dimids(3))
          call check(err, 'In nf90mpi_def_dim Z: ')

          ! define variable with no transposed file layout: XYZ
          err = nf90mpi_def_var(ncid, "XYZ_var", NF90_INT, dimids, &
                                XYZ_id)
          call check(err, 'In nf90mpi_def_var XYZ_var: ')

          ! define variable with transposed file layout: XYZ -> XZY
          dimidsT = (/ dimids(1), dimids(3), dimids(2) /)
          err = nf90mpi_def_var(ncid, "XZY_var", NF90_INT, dimidsT, &
                                XZY_id)
          call check(err, 'In nf90mpi_def_var XZY_var: ')

          ! define variable with transposed file layout: XYZ -> YXZ
          dimidsT = (/ dimids(2), dimids(1), dimids(3) /)
          err = nf90mpi_def_var(ncid, "YXZ_var", NF90_INT, dimidsT, &
                                YXZ_id)
          call check(err, 'In nf90mpi_def_var YXZ_var: ')

          ! define variable with transposed file layout: XYZ -> YZX
          dimidsT = (/ dimids(2), dimids(3), dimids(1) /)
          err = nf90mpi_def_var(ncid, "YZX_var", NF90_INT, dimidsT, &
                                YZX_id)
          call check(err, 'In nf90mpi_def_var YZX_var: ')

          ! define variable with transposed file layout: XYZ -> ZXY
          dimidsT = (/ dimids(3), dimids(1), dimids(2) /)
          err = nf90mpi_def_var(ncid, "ZXY_var", NF90_INT, dimidsT, &
                                ZXY_id)
          call check(err, 'In nf90mpi_def_var ZXY_var: ')

          ! define variable with transposed file layout: XYZ -> ZYX
          dimidsT = (/ dimids(3), dimids(2), dimids(1) /)
          err = nf90mpi_def_var(ncid, "ZYX_var", NF90_INT, dimidsT, &
                                ZYX_id)
          call check(err, 'In nf90mpi_def_var ZYX_var: ')

          ! do not forget to exit define mode
          err = nf90mpi_enddef(ncid)
          call check(err, 'In nf90mpi_enddef: ')

          ! now we are in data mode

          ! write the whole variable in file: XYZ
          err = nf90mpi_put_var_all(ncid, XYZ_id, buf, starts, counts)
          call check(err, 'In nf90mpi_put_vara_int_all XYZ: ')

          ! write the transposed variable:  XYZ -> XZY
          imap    = (/ 1_MPI_OFFSET_KIND, counts(1)*counts(2), counts(1) /)
          startsT = (/ starts(1), starts(3), starts(2) /)
          countsT = (/ counts(1), counts(3), counts(2) /)
          err = nf90mpi_put_var_all(ncid, XZY_id, buf, startsT, &
                                    countsT, strides, imap)
          call check(err, 'In nf90mpi_put_var_all XZY: ')

          ! write the transposed variable:  XYZ -> YXZ
          imap    = (/ counts(1), 1_MPI_OFFSET_KIND, counts(1)*counts(2) /)
          startsT = (/ starts(2), starts(1), starts(3) /)
          countsT = (/ counts(2), counts(1), counts(3) /)
          err = nf90mpi_put_var_all(ncid, YXZ_id, buf, startsT, &
                                    countsT, strides, imap)
          call check(err, 'In nf90mpi_put_var_all YZX: ')

          ! write the transposed variable:  XYZ -> YZX
          imap    = (/ counts(1), counts(1)*counts(2), 1_MPI_OFFSET_KIND /)
          startsT = (/ starts(2), starts(3), starts(1) /)
          countsT = (/ counts(2), counts(3), counts(1) /)
          err = nf90mpi_put_var_all(ncid, YZX_id, buf, startsT, &
                                    countsT, strides, imap)
          call check(err, 'In nf90mpi_put_var_all YZX: ')

          ! write the transposed variable:  XYZ -> ZXY
          imap    = (/ counts(1)*counts(2), 1_MPI_OFFSET_KIND, counts(1) /)
          startsT = (/ starts(3), starts(1), starts(2) /)
          countsT = (/ counts(3), counts(1), counts(2) /)
          err = nf90mpi_put_var_all(ncid, ZXY_id, buf, startsT, &
                                    countsT, strides, imap)
          call check(err, 'In nf90mpi_put_var_all ZXY: ')

          ! write the transposed variable:  XYZ -> ZYX
          imap    = (/ counts(1)*counts(2), counts(1), 1_MPI_OFFSET_KIND /)
          startsT = (/ starts(3), starts(2), starts(1) /)
          countsT = (/ counts(3), counts(2), counts(1) /)
          err = nf90mpi_put_var_all(ncid, ZYX_id, buf, startsT, &
                                    countsT, strides, imap)
          call check(err, 'In nf90mpi_put_var_all ZYX: ')

          ! close the file
          err = nf90mpi_close(ncid)
          call check(err, 'In nf90mpi_close: ')

          ! check if there is any PnetCDF internal malloc residue
 998      format(A,I13,A)
          err = nf90mpi_inq_malloc_size(malloc_size)
          if (err == NF90_NOERR) then
              call MPI_Reduce(malloc_size, sum_size, 1, MPI_OFFSET, &
                              MPI_SUM, 0, MPI_COMM_WORLD, err)
              if (rank .EQ. 0 .AND. sum_size .GT. 0_MPI_OFFSET_KIND) print 998, &
                  'heap memory allocated by PnetCDF internally has ', &
                  sum_size/1048576, ' MiB yet to be freed'
          endif

 999      call MPI_Finalize(err)
          ! call EXIT(0) ! EXIT() is a GNU extension
      end program main


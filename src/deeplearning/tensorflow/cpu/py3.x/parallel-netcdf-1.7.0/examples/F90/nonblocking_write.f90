!
!  Copyright (C) 2012, Northwestern University and Argonne National Laboratory
!  See COPYRIGHT notice in top-level directory.
!
! $Id: nonblocking_write.f90 2325 2016-02-28 07:49:13Z wkliao $

!    This example is the Fortran 90 version of nonblocking_write.c
!    It creates a netcdf file in CD-5 format and writes a number of
!    3D integer non-record variables.
!    Usage: (for example)
!    To compile:
!        mpif90 -O2 nonblocking_write.f90 -o nonblocking_write -lpnetcdf
!    To run:
!        mpiexec -n num_processes ./nonblocking_write [filename] [len]
!    where len decides the size of each local array, which is
!    len x len x len. Each non-record variable is of size
!          len*len*len * nprocs * sizeof(int)
!    All variables are partitioned among all processes in a 3D
!    block-block-block fashion.
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

          integer NDIMS, NUM_VARS
          PARAMETER(NDIMS=3, NUM_VARS=10)

          character(LEN=256) filename, cmd, str
          integer i, cmode, err, ierr, get_args, info
          integer rank, nprocs, len, bufsize, ncid
          integer psizes(NDIMS), dimids(NDIMS), varids(NUM_VARS)
          integer req(NUM_VARS), st(NUM_VARS)
          integer(kind=MPI_OFFSET_KIND) gsizes(NDIMS)
          integer(kind=MPI_OFFSET_KIND) starts(NDIMS), counts(NDIMS)
          integer(kind=MPI_OFFSET_KIND) bbufsize
          integer(kind=MPI_OFFSET_KIND) malloc_size, sum_size
          logical verbose

          ! define an array of pointers
          type intp
              integer, dimension(:), pointer :: p
          end type intp
          type(intp) buf(NUM_VARS)

          call MPI_Init(err)
          call MPI_Comm_rank(MPI_COMM_WORLD, rank, err)
          call MPI_Comm_size(MPI_COMM_WORLD, nprocs, err)

          ! take filename from command-line argument if there is any
          if (rank .EQ. 0) then
              verbose = .TRUE.
              filename = "testfile.nc"
              len = 10
              ierr = get_args(3, cmd, filename, verbose, len)
          endif
          call MPI_Bcast(ierr, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, err)
          if (ierr .EQ. 0) goto 999

          call MPI_Bcast(verbose, 1, MPI_LOGICAL, 0, MPI_COMM_WORLD, err)
          call MPI_Bcast(filename, 256, MPI_CHARACTER, 0, MPI_COMM_WORLD, err)
          call MPI_Bcast(len, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, err)

          do i=1,NDIMS
             psizes(i) = 0
          enddo

          ! create a block-block-block data partitioning
          call MPI_Dims_create(nprocs, NDIMS, psizes, err);
          starts(1) = mod(rank, psizes(1))
          starts(2) = mod((rank / psizes(2)), psizes(2))
          starts(3) = mod((rank / (psizes(1) * psizes(2))), psizes(3))

          bufsize = 1
          do i=1,NDIMS
              gsizes(i) = len * psizes(i)
              starts(i) = starts(i) * len + 1;
              counts(i) = len;
              bufsize   = bufsize * len;
          enddo

          ! allocate buffer and initialize with rank IDs
          do i=1, NUM_VARS
             allocate(buf(i)%p(bufsize))
             buf(i)%p(:) = rank
          enddo

          ! set an MPI-IO hint to disable file offset alignment for
          ! fix-sized variables
          call MPI_Info_create(info, err)
          call MPI_Info_set(info, "nc_var_align_size", "1", err)

          ! create file, truncate it if exists
          cmode = IOR(NF90_CLOBBER, NF90_64BIT_DATA)
          err = nf90mpi_create(MPI_COMM_WORLD, filename, cmode, &
                               info, ncid)
          call check(err, 'In nf90mpi_create: ')

          call MPI_Info_free(info, err)

          ! define dimensions
          do i=1, NDIMS
             write(str,'(I2)') i
             err = nf90mpi_def_dim(ncid, "x"//trim(ADJUSTL(str)), &
                                   gsizes(i), dimids(i))
             call check(err, 'In nf90mpi_def_dim x'//trim(str))
          enddo

          !define variables
          do i=1, NUM_VARS
             write(str,'(I2)') i
             err = nf90mpi_def_var(ncid, "var"//trim(ADJUSTL(str)), &
                                   NF90_INT, dimids, varids(i))
             call check(err, 'In nf90mpi_def_var var'//trim(str))
          enddo

          ! do not forget to exit define mode
          err = nf90mpi_enddef(ncid)
          call check(err, 'In nf90mpi_enddef: ')

          ! write one variable at a time using iput
          do i=1, NUM_VARS
             write(str,'(I2)') i
             err = nf90mpi_iput_var(ncid, varids(i), buf(i)%p, req(i), &
                                    starts, counts)
             call check(err, 'In nf90mpi_iput_var '//trim(str))
          enddo

          ! wait for the nonblocking I/O to complete
          err = nf90mpi_wait_all(ncid, NUM_VARS, req, st)
          call check(err, 'In nf90mpi_wait_all')

          ! check the status of each nonblocking request
          do i=1, NUM_VARS
             write(str,'(I2)') i
             call check(st(i), 'In nf90mpi_wait_all req '//trim(str))
          enddo

          ! write one variable at a time using bput

          ! bbufsize must be max of data type converted before and after
          bbufsize = bufsize * NUM_VARS * 4  ! 4 is size of integer
          err = nf90mpi_buffer_attach(ncid, bbufsize)
          call check(err, 'In nf90mpi_buffer_attach')

          do i=1, NUM_VARS
             write(str,'(I2)') i
             err = nf90mpi_bput_var(ncid, varids(i), buf(i)%p, req(i), &
                                    starts, counts)
             call check(err, 'In nf90mpi_bput_var '//trim(str))

             ! can safely free up the buffer here
             deallocate(buf(i)%p)
          enddo

          ! wait for the nonblocking I/O to complete
          err = nf90mpi_wait_all(ncid, NUM_VARS, req, st)
          call check(err, 'In nf90mpi_wait_all')

          ! check the status of each nonblocking request
          do i=1, NUM_VARS
             write(str,'(I2)') i
             call check(st(i), 'In nf90mpi_wait_all req '//trim(str))
          enddo

          ! detach the temporary buffer
          err = nf90mpi_buffer_detach(ncid)
          call check(err, 'In nf90mpi_buffer_detach')

          ! close the file
          err = nf90mpi_close(ncid);
          call check(err, 'In nf90mpi_close')

          ! check if there is any PnetCDF internal malloc residue
 998      format(A,I13,A)
          err = nf90mpi_inq_malloc_size(malloc_size)
          if (err == NF90_NOERR) then
              call MPI_Reduce(malloc_size, sum_size, 1, MPI_OFFSET, &
                              MPI_SUM, 0, MPI_COMM_WORLD, err)
              if (rank .EQ. 0 .AND. sum_size .GT. 0_MPI_OFFSET_KIND) print 998, &
                  'heap memory allocated by PnetCDF internally has ',  &
                  sum_size/1048576, ' MiB yet to be freed'
          endif

 999      call MPI_Finalize(err)
          ! call EXIT(0) ! EXIT() is a GNU extension

      end program main


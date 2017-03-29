!
!  Copyright (C) 2012, Northwestern University and Argonne National Laboratory
!  See COPYRIGHT notice in top-level directory.
!
! $Id: varn_real.f90 2205 2015-11-28 20:41:50Z wkliao $

!
! This example shows how to use a single call of nf90mpi_put_varn_all()
! to write a sequence of one-element requests with arbitrary array indices.
!
! The compile and run commands are given below, together with an ncmpidump of
! the output file.
!
!    % mpif90 -O2 -o varn_real varn_real.f90 -lpnetcdf
!    % mpiexec -n 4 ./varn_real /pvfs2/wkliao/testfile.nc
!    % ncmpidump /pvfs2/wkliao/testfile.nc
!    netcdf testfile {
!    // file format: CDF-5 (big variables)
!    dimensions:
!             Y = 4 ;
!             X = 10 ;
!    variables:
!             float var(Y, X) ;
!    data:
!
!     var =
!       3, 3, 3, 1, 1, 0, 0, 2, 1, 1,
!       0, 2, 2, 2, 3, 1, 1, 2, 2, 2,
!       1, 1, 2, 3, 3, 3, 0, 0, 1, 1,
!       0, 0, 0, 2, 1, 1, 1, 3, 3, 3 ;
!    }
!

      subroutine check(err, message)
          use mpi
          use pnetcdf
          implicit none
          integer err
          character(len=*) message
          character(len=128) msg

          ! It is a good idea to check returned value for possible error
          if (err .NE. NF90_NOERR) then
              write(6,*) message, trim(nf90mpi_strerror(err))
              msg = '*** TESTING F90 varn_real.f90 for varn API '
              call pass_fail(1, msg)
              ! call MPI_Abort(MPI_COMM_WORLD, -1, err)
          end if
      end subroutine check

      program main
          use mpi
          use pnetcdf
          implicit none

          integer NDIMS
          PARAMETER(NDIMS=2)

          character(LEN=256) filename, cmd, msg
          integer rank, nprocs, err, ierr, num_reqs, get_args
          integer ncid, cmode, varid, dimid(2), y, x, i, j, nerrs
          integer, allocatable :: req_lens(:)
          real, allocatable :: buffer(:), buffer2D(:,:)
          real oneReal
          integer(kind=MPI_OFFSET_KIND) NY, NX
          integer(kind=MPI_OFFSET_KIND) w_len, w_req_len
          integer(kind=MPI_OFFSET_KIND), allocatable :: starts(:,:)
          integer(kind=MPI_OFFSET_KIND), allocatable :: counts(:,:)
          integer(kind=MPI_OFFSET_KIND) malloc_size, sum_size

          NY = 4
          NX = 10

          call MPI_Init(ierr)
          call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
          call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)

          ! take filename from command-line argument if there is any
          if (rank .EQ. 0) then
              filename = "testfile.nc"
              err = get_args(cmd, filename)
          endif
          call MPI_Bcast(err, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
          if (err .EQ. 0) goto 999
          
          call MPI_Bcast(filename, 256, MPI_CHARACTER, 0, MPI_COMM_WORLD, ierr)

          nerrs = 0

          if (.FALSE. .AND. nprocs .NE. 4 .AND. rank .EQ. 0) &
              print*,'Warning: ',trim(cmd),' is intended to run on ', &
                     '4 processes'

          ! create file, truncate it if exists
          cmode = IOR(NF90_CLOBBER, NF90_64BIT_DATA)
          err = nf90mpi_create(MPI_COMM_WORLD, filename, cmode, &
                             MPI_INFO_NULL, ncid)
          call check(err, 'In nf90mpi_create: ')

          ! create a global array of size NY * NX */
          err = nf90mpi_def_dim(ncid, "Y", NY, dimid(2))
          call check(err, 'In nf90mpi_def_dim Y: ')
          err = nf90mpi_def_dim(ncid, "X", NX, dimid(1))
          call check(err, 'In nf90mpi_def_dim X: ')
          err = nf90mpi_def_var(ncid, "var", NF90_FLOAT, dimid, varid)
          call check(err, 'In nf90mpi_def_var var: ')
          err = nf90mpi_enddef(ncid)
          call check(err, 'In nf90mpi_enddef: ')

          ! pick arbitrary numbers of requests for 4 processes
          num_reqs = 0
          if (rank .EQ.  0) then
              num_reqs = 8
          elseif (rank .EQ. 1) then
              num_reqs = 13
          elseif (rank .EQ. 2) then
              num_reqs = 9
          elseif (rank .EQ. 3) then
              num_reqs = 10
          endif

          ! Note that in Fortran, array indices start with 1
          ALLOCATE(starts(NDIMS, num_reqs+1))
          ALLOCATE(counts(NDIMS, num_reqs))

          ! assign arbitrary starts
          y=2
          x=1
          if (rank .EQ. 0) then
              starts(y, 1) = 1; starts(x, 1) = 6
              starts(y, 2) = 2; starts(x, 2) = 1
              starts(y, 3) = 3; starts(x, 3) = 7
              starts(y, 4) = 4; starts(x, 4) = 1
              starts(y, 5) = 1; starts(x, 5) = 7
              starts(y, 6) = 3; starts(x, 6) = 8
              starts(y, 7) = 4; starts(x, 7) = 2
              starts(y, 8) = 4; starts(x, 8) = 3
              ! rank 0 is writing the following locations: ("-" means skip)
              !   -  -  -  -  -  0  0  -  -  - 
              !   0  -  -  -  -  -  -  -  -  - 
              !   -  -  -  -  -  -  0  0  -  - 
              !   0  0  0  -  -  -  -  -  -  - 
          elseif (rank .EQ. 1) then
              starts(y,  1) = 1; starts(x,  1) = 4
              starts(y,  2) = 1; starts(x,  2) = 9
              starts(y,  3) = 2; starts(x,  3) = 6
              starts(y,  4) = 3; starts(x,  4) = 1
              starts(y,  5) = 3; starts(x,  5) = 9
              starts(y,  6) = 4; starts(x,  6) = 5
              starts(y,  7) = 1; starts(x,  7) = 5
              starts(y,  8) = 1; starts(x,  8) = 10
              starts(y,  9) = 2; starts(x,  9) = 7
              starts(y, 10) = 3; starts(x, 10) = 2
              starts(y, 11) = 3; starts(x, 11) = 10
              starts(y, 12) = 4; starts(x, 12) = 6
              starts(y, 13) = 4; starts(x, 13) = 7
              ! rank 1 is writing the following locations: ("-" means skip)
              !   -  -  -  1  1  -  -  -  1  1 
              !   -  -  -  -  -  1  1  -  -  - 
              !   1  1  -  -  -  -  -  -  1  1 
              !   -  -  -  -  1  1  1  -  -  - 
          elseif (rank .EQ. 2) then
              starts(y, 1) = 1; starts(x, 1) = 8
              starts(y, 2) = 2; starts(x, 2) = 2
              starts(y, 3) = 2; starts(x, 3) = 8
              starts(y, 4) = 3; starts(x, 4) = 3
              starts(y, 5) = 4; starts(x, 5) = 4
              starts(y, 6) = 2; starts(x, 6) = 3
              starts(y, 7) = 2; starts(x, 7) = 9
              starts(y, 8) = 2; starts(x, 8) = 4
              starts(y, 9) = 2; starts(x, 9) = 10
              ! rank 2 is writing the following locations: ("-" means skip)
              !   -  -  -  -  -  -  -  2  -  - 
              !   -  2  2  2  -  -  -  2  2  2 
              !   -  -  2  -  -  -  -  -  -  - 
              !   -  -  -  2  -  -  -  -  -  - 
          elseif (rank .EQ. 3) then
              starts(y,  1) = 1; starts(x,  1) = 1
              starts(y,  2) = 2; starts(x,  2) = 5
              starts(y,  3) = 3; starts(x,  3) = 4
              starts(y,  4) = 4; starts(x,  4) = 8
              starts(y,  5) = 1; starts(x,  5) = 2
              starts(y,  6) = 3; starts(x,  6) = 5
              starts(y,  7) = 4; starts(x,  7) = 9
              starts(y,  8) = 1; starts(x,  8) = 3
              starts(y,  9) = 3; starts(x,  9) = 6
              starts(y, 10) = 4; starts(x, 10) = 10
              ! rank 3 is writing the following locations: ("-" means skip)
              !   3  3  3  -  -  -  -  -  -  - 
              !   -  -  -  -  3  -  -  -  -  - 
              !   -  -  -  3  3  3  -  -  -  - 
              !   -  -  -  -  -  -  -  3  3  3 
          endif
          counts = 1

          ALLOCATE(req_lens(num_reqs))

          ! allocate I/O buffer and initialize its contents
          w_len = 0
          do i=1, num_reqs
             w_req_len = 1
             do j=1, NDIMS
                w_req_len = w_req_len * counts(j, i)
             enddo
             req_lens(i) = INT(w_req_len)
             w_len = w_len + w_req_len
          enddo
          ALLOCATE(buffer(w_len))
          ALLOCATE(buffer2D(3, w_len/3+1))
          ! Note on 2D buffer, put_varn will fill in the first w_len
          ! elements in buffer2D, no matter the shape of buffer2D is

          buffer   = rank
          buffer2D = rank

          ! varn write a 2D buffer in memory to a 2D array in file
          err = nf90mpi_put_varn_all(ncid, varid, buffer2D, num_reqs, &
                                     starts, counts)
          call check(err, 'In nf90mpi_put_varn_all: ')

          ! varn write a 1D buffer in memory to a 2D array in file
          err = nf90mpi_put_varn_all(ncid, varid, buffer, num_reqs, &
                                     starts, counts)
          call check(err, 'In nf90mpi_put_varn_all: ')

          if (nprocs .GT. 4) call MPI_Barrier(MPI_COMM_WORLD, err)

          ! read a scalar back and check the content
          oneReal = -1.0  ! a scalar
          if (rank .GE. 4) starts = 1_MPI_OFFSET_KIND
          err = nf90mpi_get_varn_all(ncid, varid, oneReal, starts)
          call check(err, ' 22 In nf90mpi_get_varn_all: ')

 995      format(A,I2,A,F4.1)
          if (rank .LT. 4 .AND. oneReal .NE. rank) then
              print 995, "Error: expecting OneReal=",rank, &
                         " but got", oneReal
              nerrs = nerrs + 1
          endif

          ! varn read a 2D array in file to a 2D buffer in memory
          buffer2D = -1.0;
          err = nf90mpi_get_varn_all(ncid, varid, buffer2D, num_reqs, &
                                     starts, counts)
          call check(err, 'In nf90mpi_get_varn_all: ')

 996      format(A,I2,A,I2,A,I2,A,F4.1)
          do i=1, INT(w_len/3)
             do j=1, 3
                if (buffer2D(j,i) .NE. rank) then
                    print 996, "Error: expecting buffer2D(",j,",",i,")=", &
                               rank, " but got", buffer2D(j,i)
                    nerrs = nerrs + 1
                endif
             enddo
          enddo

          ! varn read a 2D array in file to a 1D buffer in memory
          buffer = -1.0;
          err = nf90mpi_get_varn_all(ncid, varid, buffer, num_reqs, &
                                     starts, counts)
          call check(err, 'In nf90mpi_get_varn_all: ')

 997      format(A,I2,A,I2,A,F4.1)
          do i=1, INT(w_len)
             if (buffer(i) .NE. rank) then
                 print 997, "Error: expecting buffer(",i,")=",rank, &
                            " but got", buffer(i)
                 nerrs = nerrs + 1
             endif
          enddo

          err = nf90mpi_close(ncid);
          call check(err, 'In nf90mpi_close: ')

          DEALLOCATE(req_lens);
          DEALLOCATE(buffer);
          DEALLOCATE(buffer2D);
          DEALLOCATE(starts);
          DEALLOCATE(counts);

          ! check if there is any PnetCDF internal malloc residue
 998      format(A,I13,A)
          err = nf90mpi_inq_malloc_size(malloc_size)
          if (err == NF90_NOERR) then
              call MPI_Reduce(malloc_size, sum_size, 1, MPI_OFFSET, &
                              MPI_SUM, 0, MPI_COMM_WORLD, ierr)
              if (rank .EQ. 0 .AND. sum_size .GT. 0_MPI_OFFSET_KIND) print 998, &
                  'heap memory allocated by PnetCDF internally has ',  &
                  sum_size/1048576, ' MiB yet to be freed'
          endif

          msg = '*** TESTING F90 '//trim(cmd)//' for varn API '
          if (rank .eq. 0) call pass_fail(nerrs, msg)

 999      call MPI_Finalize(ierr)

      end program


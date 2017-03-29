!
!   Copyright (C) 2014, Northwestern University and Argonne National Laboratory
!   See COPYRIGHT notice in top-level directory.
!
! $Id: vard_int.f90 2239 2015-12-18 18:21:09Z wkliao $

!
! This example shows how to use the vard API nf90mpi_put_vard() and
! nf90mpi_get_vard() to write and read 2D record and fixed-size
! variables.
!
!    To compile:
!        mpif90 -O2 vard_int.f90 -o vard_int -lpnetcdf
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
          integer err, nprocs, rank, i, j, ierr, get_args, dummy
          integer cmode, ncid, varid0, varid1, dimid(2)
          integer(kind=MPI_OFFSET_KIND) NX, NY
          integer(kind=MPI_OFFSET_KIND) start(2), count(2), bufcount
          PARAMETER(NX=5, NY=2)
          integer buf(NX, NY)
          integer buftype, rec_filetype, fix_filetype
          integer array_of_sizes(2), array_of_subsizes(2)
          integer array_of_starts(2), blocklengths(2)
          integer(kind=MPI_OFFSET_KIND) len, malloc_size, sum_size, recsize
          integer(kind=MPI_ADDRESS_KIND) disps(2)
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
          cmode = NF90_CLOBBER
          err = nf90mpi_create(MPI_COMM_WORLD, filename, cmode, &
                               MPI_INFO_NULL, ncid)
          call check(err, 'In nf90mpi_create: ')

          ! define 2 dimensions
          err = nf90mpi_def_dim(ncid, "RECV_DIM", NF90MPI_UNLIMITED, dimid(2))
          call check(err, 'In nf90mpi_def_dim RECV_DIM: ')
          len = NX * nprocs
          err = nf90mpi_def_dim(ncid, "X", len, dimid(1))
          call check(err, 'In nf90mpi_def_dim X: ')

          ! define 2D record variables of integer type
          err = nf90mpi_def_var(ncid, "rec_var", NF90_INT, dimid, varid0)
          call check(err, 'In nf90mpi_def_var: rec_var ')

          ! define 2D fixed-size variable of integer type
          err = nf90mpi_def_dim(ncid, "FIX_DIM", 2_MPI_OFFSET_KIND, dimid(2))
          call check(err, 'In nf90mpi_def_dim RECV_DIM: ')
          err = nf90mpi_def_var(ncid, "fix_var", NF90_INT, dimid, varid1)
          call check(err, 'In nf90mpi_def_var: fix_var ')

          ! do not forget to exit define mode
          err = nf90mpi_enddef(ncid)
          call check(err, 'In nf90mpi_enddef: ')

          ! create a file type for the record variable */
          err = nf90mpi_inq_recsize(ncid, recsize)
          call check(err, 'In nf90mpi_inq_recsize: ')
          blocklengths(1) = INT(count(1))
          blocklengths(2) = INT(count(1))
          disps(1) = start(1)*4
          disps(2) = recsize + start(1)*4
          call MPI_Type_create_hindexed(2, blocklengths, disps, &
                                        MPI_INTEGER, rec_filetype, err)
          call MPI_Type_commit(rec_filetype, err)

          ! create a file type for the fixed-size variable
          array_of_sizes(1) = INT(NX*nprocs)
          array_of_sizes(2) = 2
          array_of_subsizes(1) = INT(count(1))
          array_of_subsizes(2) = INT(count(2))
          array_of_starts(1) = INT(start(1))
          array_of_starts(2) = INT(start(2))
          call MPI_Type_create_subarray(2, array_of_sizes, array_of_subsizes, &
               array_of_starts, MPI_ORDER_FORTRAN, MPI_INTEGER, fix_filetype,&
               err)
          call MPI_Type_commit(fix_filetype, err)

          bufcount = count(1) * count(2)
          buftype = MPI_INTEGER

          ! write the record variable */
          err = nf90mpi_put_vard_all(ncid, varid0, rec_filetype, buf, &
                                     bufcount, buftype)
          call check(err, 'In nf90mpi_put_vard_all: ')

          ! check if the number of records changed to 2
          err = nf90mpi_inquire(ncid, unlimitedDimId=dimid(2))
          call check(err, 'In nf90mpi_inquire: ')
          err = nf90mpi_inquire_dimension(ncid, dimid(2), len=len)
          call check(err, 'In nf90mpi_inquire_dimension: ')
          if (len .NE. 2) then
              print*, 'Error: number of records should be 2 but got ', &
                       len
          endif

          ! write the fixed-size variable
          err = nf90mpi_put_vard_all(ncid, varid1, fix_filetype, buf, &
                                     bufcount, buftype)
          call check(err, 'In nf90mpi_put_vard_all: ')

          ! close the file
          err = nf90mpi_close(ncid)
          call check(err, 'In nf90mpi_close: ')

          ! open the same file and read back for validate */
          err = nf90mpi_open(MPI_COMM_WORLD, filename, NF90_NOWRITE, &
                             MPI_INFO_NULL, ncid)
          call check(err, 'In nf90mpi_open: ')

          err = nf90mpi_inq_varid(ncid, "rec_var", varid0)
          call check(err, 'In nf90mpi_inq_varid rec_var: ')
          err = nf90mpi_inq_varid(ncid, "fix_var", varid1)
          call check(err, 'In nf90mpi_inq_varid fix_var: ')

          ! PnetCDF start() argument starts with 1
          start = start + 1

          ! read back record variable
          err = nf90mpi_get_vard_all(ncid, varid0, rec_filetype, buf, &
                                     bufcount, buftype)
          call check(err, 'In nf90mpi_get_vard_all: ')

          ! read back fixed-size variable
          err = nf90mpi_get_vard_all(ncid, varid1, fix_filetype, buf, &
                                     bufcount, buftype)
          call check(err, 'In nf90mpi_get_vard_all: ')

          ! close the file
          err = nf90mpi_close(ncid)
          call check(err, 'In nf90mpi_close: ')

          call MPI_Type_free(rec_filetype, err)
          call MPI_Type_free(fix_filetype, err)

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


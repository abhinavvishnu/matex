!
!   Copyright (C) 2014, Northwestern University and Argonne National Laboratory
!   See COPYRIGHT notice in top-level directory.
!
! $Id: inq_num_varsf.f90 2205 2015-11-28 20:41:50Z wkliao $

!
! This program tests if one can get the number of record variables and
! fixed-size variables correctly. It first defines some number of
! fixed-size and record variables and then calls the APIs
!     ncmpi_inq_num_rec_vars() and ncmpi_inq_num_fix_vars()
! to varify if the numbers are correct.
!
! The compile and run commands are given below. This program is to be
! run on one MPI process.
!
!    % mpif90 -g -o inq_num_varsf inq_num_varsf.f90 -lpnetcdf
!
!    % mpiexec -l -n 1 inq_num_varsf testfile.nc
!
!    % ncmpidump -h testfile.nc
!    netcdf testfile {
!    // file format: CDF-2 (large file)
!    dimensions:
!            X = 10 ;
!            Y = 2 ;
!            REC_DIM = UNLIMITED ; // (0 currently)
!    variables:
!            int REC_VAR_1(REC_DIM) ;
!            int REC_VAR_2(REC_DIM, Y, X) ;
!            int REC_VAR_3(REC_DIM, Y) ;
!            int REC_VAR_4(REC_DIM) ;
!            int FIX_VAR_1(Y, X) ;
!            int FIX_VAR_2(X) ;
!            int FIX_VAR_3(X) ;
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
              write(6,*) trim(message), trim(nf90mpi_strerror(err))
              msg = '*** TESTING F90 inq_num_varsf.f90 for no. record/fixed variables'
              call pass_fail(1, msg)
              call MPI_Abort(MPI_COMM_WORLD, -1, err)
          end if
      end subroutine check

      program main
          use mpi
          use pnetcdf
          implicit none

          character(LEN=256) filename, cmd, msg
          integer err, ierr, nprocs, rank, cmode, ncid, get_args
          integer varid(7), dimid(3), dimid_1D(1), dimid_2D(2)
          integer nerrs, nvars, num_rec_vars, num_fix_vars
          integer(kind=MPI_OFFSET_KIND) malloc_size, sum_size

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

          ! create file, truncate it if exists
          cmode = IOR(NF90_CLOBBER, NF90_64BIT_OFFSET)
          err = nf90mpi_create(MPI_COMM_WORLD, filename, cmode, &
                               MPI_INFO_NULL, ncid)
          call check(err, 'In nf90mpi_create: ')

          ! define dimensions
          err = nf90mpi_def_dim(ncid, "X", 10_MPI_OFFSET_KIND, dimid(1))
          call check(err, 'In nf90mpi_def_dim X: ')
          err = nf90mpi_def_dim(ncid, "Y", 2_MPI_OFFSET_KIND,  dimid(2))
          call check(err, 'In nf90mpi_def_dim Y: ')
          err = nf90mpi_def_dim(ncid, "REC_DIM", NF90MPI_UNLIMITED, dimid(3))
          call check(err, 'In nf90mpi_def_dim REC_DIM: ')

          ! define some record variables
          dimid_1D(1) = dimid(3)
          dimid_2D(1) = dimid(2)
          dimid_2D(2) = dimid(3)

          err = nf90mpi_def_var(ncid, "REC_VAR_1", NF90_INT, dimid_1D, varid(1))
          call check(err, 'In nf90mpi_def_var: REC_VAR_1')
          err = nf90mpi_def_var(ncid, "REC_VAR_2", NF90_INT, dimid,    varid(2))
          call check(err, 'In nf90mpi_def_var: REC_VAR_2')
          err = nf90mpi_def_var(ncid, "REC_VAR_3", NF90_INT, dimid_2D, varid(3))
          call check(err, 'In nf90mpi_def_var: REC_VAR_3')
          err = nf90mpi_def_var(ncid, "REC_VAR_4", NF90_INT, dimid_1D, varid(4))
          call check(err, 'In nf90mpi_def_var: REC_VAR_4')

          ! define some fixed-size variables
          dimid_1D(1) = dimid(1)
          dimid_2D(1) = dimid(1)
          dimid_2D(2) = dimid(2)

          err = nf90mpi_def_var(ncid, "FIX_VAR_1", NF90_INT, dimid_2D, varid(5))
          call check(err, 'In nf90mpi_def_var: FIX_VAR_1')
          err = nf90mpi_def_var(ncid, "FIX_VAR_2", NF90_INT, dimid_1D, varid(6))
          call check(err, 'In nf90mpi_def_var: FIX_VAR_2')
          err = nf90mpi_def_var(ncid, "FIX_VAR_3", NF90_INT, dimid_1D, varid(7))
          call check(err, 'In nf90mpi_def_var: FIX_VAR_3')

          ! do not forget to exit define mode
          err = nf90mpi_enddef(ncid)
          call check(err, 'In nf90mpi_enddef: ')

          ! inquire the numbers of variables (record and fixed-size
          err = nf90mpi_inquire(ncid, nVariables=nvars)
          call check(err, 'In nf90mpi_inquire: ')
          err = nf90mpi_inq_num_rec_vars(ncid, num_rec_vars)
          call check(err, 'In nf90mpi_inq_num_rec_vars: ')
          err = nf90mpi_inq_num_fix_vars(ncid, num_fix_vars)
          call check(err, 'In nf90mpi_inq_num_fix_vars: ')

          ! check if the numbers of variables are expected
          nerrs = 0
          if (nvars .NE. 7) then
              write(6,*) "Error: expecting 7 number of variables defined, but got ", nvars
              nerrs = nerrs + 1
          endif
          if (num_rec_vars .NE. 4) then
              write(6,*) "Error: expecting 4 number of recond variables defined, but got ", nvars
              nerrs = nerrs + 1
          endif
          if (num_fix_vars .NE. 3) then
              write(6,*) "Error: expecting 3 number of fixed-size variables defined, but got ", nvars
              nerrs = nerrs + 1
          endif

          err = nf90mpi_close(ncid)
          call check(err, 'In nf90mpi_close: ')

          ! check if there is any PnetCDF internal malloc residue
 998      format(A,I13,A)
          err = nfmpi_inq_malloc_size(malloc_size)
          if (err == NF_NOERR) then
              call MPI_Reduce(malloc_size, sum_size, 1, MPI_OFFSET, &
                              MPI_SUM, 0, MPI_COMM_WORLD, ierr)
              if (rank .EQ. 0 .AND. sum_size .GT. 0_MPI_OFFSET_KIND) print 998, &
                  'heap memory allocated by PnetCDF internally has ',  &
                  sum_size/1048576, ' MiB yet to be freed'
          endif

          msg = '*** TESTING F90 '//trim(cmd)//' for no. record/fixed variables'
          if (rank .eq. 0) call pass_fail(nerrs, msg)

 999      call MPI_Finalize(ierr)
      end program main


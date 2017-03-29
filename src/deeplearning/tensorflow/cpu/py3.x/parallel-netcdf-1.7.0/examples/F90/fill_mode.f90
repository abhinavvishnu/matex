!
!   Copyright (C) 2015, Northwestern University and Argonne National Laboratory
!   See COPYRIGHT notice in top-level directory.
!
! $Id: fill_mode.f90 2245 2015-12-20 18:39:52Z wkliao $

! This example shows how to use
! 1. nf90mpi_set_fill() to enable fill mode
! 2. nf90mpi_def_var_fill() to define the variable's fill value
! 3. nf90mpi_inq_var_fill() to inquire the variable's fill mode
! information
! 4. nf90mpi_put_vara_int_all() to write two 2D 4-byte integer array in
! parallel.
! It first defines a netCDF record variable of size global_nx *
! NFMPI_UNLIMITED
! where
!    global_nx == (nx * number of MPI processes) and
! It then defines a netCDF variable of size global_nx * global_ny where
!    global_nx == (nx * number of MPI processes) and
!    global_ny == ny
! The data partitioning pattern for both variables are a column-wise
! partitioning across all processes. Each process writes a subarray of
! size
! nx * ny. Note the description above follows the Fortran array index
! order.

! Example commands for MPI run and outputs from running ncmpidump on the
! NC file produced by this example program:
!
!    % mpif90 -O2 -o fill_mode fill_mode.f90 -lpnetcdf
!    % mpiexec -n 4 ./fill_mode /pvfs2/wkliao/testfile.nc
!
!    % ncmpidump /pvfs2/wkliao/testfile.nc
!    netcdf testfile {
!    // file format: CDF-5 (big variables)
!    dimensions:
!            REC_DIM = UNLIMITED ; // (2 currently)
!            X = 20 ;
!            Y = 3 ;
!    variables:
!            int rec_var(REC_DIM, X) ;
!                    rec_var:_FillValue = -1 ;
!            int fix_var(Y, X) ;
!                    fix_var:_FillValue = -2147483647 ;
!    data:
!
!     rec_var =
!      0, 0, 0, 0, _, 1, 1, 1, 1, _, 2, 2, 2, 2, _, 3, 3, 3, 3, _,
!      0, 0, 0, 0, _, 1, 1, 1, 1, _, 2, 2, 2, 2, _, 3, 3, 3, 3, _ ;
!
!     fix_var =
!      0, 0, 0, 0, _, 1, 1, 1, 1, _, 2, 2, 2, 2, _, 3, 3, 3, 3, _,
!      0, 0, 0, 0, _, 1, 1, 1, 1, _, 2, 2, 2, 2, _, 3, 3, 3, 3, _,
!      0, 0, 0, 0, _, 1, 1, 1, 1, _, 2, 2, 2, 2, _, 3, 3, 3, 3, _ ;
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
          integer err, nprocs, rank, ierr, get_args, dummy
          integer cmode, ncid, rec_varid, fix_varid, dimid(2)
          integer no_fill, fill_value, old_mode
          integer(kind=MPI_OFFSET_KIND) nx, ny, global_nx, global_ny
          integer(kind=MPI_OFFSET_KIND) starts(2), counts(2)
          integer(kind=MPI_OFFSET_KIND) malloc_size, sum_size
          PARAMETER(nx=5, ny=4)
          integer buf(nx,ny)
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

          ! set parameters
          global_nx = nx * nprocs
          global_ny = ny

          buf = rank;

          ! create file, truncate it if exists
          cmode = IOR(NF90_CLOBBER, NF90_64BIT_DATA)
          err = nf90mpi_create(MPI_COMM_WORLD, filename, cmode, &
                               MPI_INFO_NULL, ncid)
          call check(err, 'In nf90mpi_create: ')

          ! define dimensions x and y
          err = nf90mpi_def_dim(ncid, "REC_DIM", NFMPI_UNLIMITED, dimid(2))
          call check(err, 'In nf90mpi_def_dim REC_DIM: ')
          err = nf90mpi_def_dim(ncid, "X", global_nx, dimid(1))
          call check(err, 'In nf90mpi_def_dim X: ')

          ! define a 2D variable of integer type
          err = nf90mpi_def_var(ncid, "rec_var", NF90_INT, dimid, rec_varid)
          call check(err, 'In nf90mpi_def_var rec_var: ')

          err = nf90mpi_def_dim(ncid, "Y", global_ny, dimid(2))
          call check(err, 'In nf90mpi_def_dim Y: ')

          ! define a 2D variable of integer type
          err = nf90mpi_def_var(ncid, "fix_var", NF90_INT, dimid, fix_varid)
          call check(err, 'In nf90mpi_def_var fix_var: ')

          ! set the fill mode to NF90_FILL for entire file
          err = nf90mpi_set_fill(ncid, NF90_FILL, old_mode)
          call check(err, 'In nf90mpi_set_fill: ')
          if (verbose) then
             if (old_mode .EQ. NF90_FILL) then
                 print*,"The old fill mode is NF90_FILL"
             else
                 print*,"The old fill mode is NF90_NOFILL"
             endif
          endif

          ! set the fill mode back to NF90_NOFILL for entire file
          err = nf90mpi_set_fill(ncid, NF90_NOFILL, old_mode)
          call check(err, 'In nf90mpi_set_fill: ')

          ! set the variable's fill mode to NF90_FILL with default fill
          ! value
          err = nf90mpi_def_var_fill(ncid, fix_varid, 0, NF90_FILL_INT)
          call check(err, 'In nf90mpi_def_var_fill: ')

          ! set a customized fill value -1
          fill_value = -1
          err = nf90mpi_put_att(ncid, rec_varid, "_FillValue", fill_value)
          call check(err, 'In nf90mpi_put_att: ')

          ! do not forget to exit define mode
          err = nf90mpi_enddef(ncid)
          call check(err, 'In nf90mpi_enddef: ')

          ! now we are in data mode

          ! Note that in Fortran, array indices start with 1
          starts(1) = nx * rank + 1
          starts(2) = 1
          counts(1) = nx
          counts(2) = ny

          ! do not write the variable in full
          counts(1) = counts(1) - 1
          err = nf90mpi_put_var_all(ncid, fix_varid, buf, starts, counts)
          call check(err, 'In nf90mpi_put_var_all: ')

          err = nf90mpi_inq_var_fill(ncid, fix_varid, no_fill, fill_value)
          if (no_fill .NE. 0) &
              print*,"Error: expecting no_fill to be 0"
          if (fill_value .NE. NF90_FILL_INT) &
              print*,"Error: expecting no_fill to be ",NF90_FILL_INT, &
                     " but got ", fill_value

          ! fill the 1st record of the record variable
          starts(2) = 1
          err = nf90mpi_fill_var_rec(ncid, rec_varid, starts(2))
          call check(err, 'In nf90mpi_fill_var_rec: ')

          ! write to the 1st record
          counts(2) = 1;
          err = nf90mpi_put_var_all(ncid, rec_varid, buf, starts, counts)
          call check(err, 'In nf90mpi_put_var_all: ')

          ! fill the 2nd record of the record variable
          starts(2) = 2
          err = nf90mpi_fill_var_rec(ncid, rec_varid, starts(2))
          call check(err, 'In nf90mpi_fill_var_rec: ')

          ! write to the 2nd record
          err = nf90mpi_put_var_all(ncid, rec_varid, buf, starts, counts)
          call check(err, 'In nf90mpi_put_var_all: ')

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
                  'heap memory allocated by PnetCDF internally has ',  &
                  sum_size/1048576, ' MiB yet to be freed'
          endif

 999      call MPI_Finalize(err)
          ! call EXIT(0) ! EXIT() is a GNU extension
      end program main


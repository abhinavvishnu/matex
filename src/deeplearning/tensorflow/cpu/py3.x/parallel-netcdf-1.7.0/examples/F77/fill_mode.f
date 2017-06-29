!
!   Copyright (C) 2015, Northwestern University
!   See COPYRIGHT notice in top-level directory.
!
! $Id: fill_mode.f 2245 2015-12-20 18:39:52Z wkliao $

!
! This example shows how to use
! 1. nfmpi_set_fill() to enable fill mode
! 2. nfmpi_def_var_fill() to define the variable's fill value
! 3. nfmpi_inq_var_fill() to inquire the variable's fill mode information
! 4. nfmpi_put_vara_int_all() to write two 2D 4-byte integer array in parallel.
! It first defines a netCDF record variable of size global_nx * NFMPI_UNLIMITED
! where
!    global_nx == (nx * number of MPI processes) and
! It then defines a netCDF variable of size global_nx * global_ny where
!    global_nx == (nx * number of MPI processes) and
!    global_ny == ny
! The data partitioning pattern for both variables are a column-wise
! partitioning across all processes. Each process writes a subarray of size
! nx * ny. Note the description above follows the Fortran array index order.
!
! Example commands for MPI run and outputs from running ncmpidump on the
! NC file produced by this example program:
!
!    % mpif77 -O2 -o fill_mode fill_mode.f -lpnetcdf
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
          implicit none
          include 'mpif.h'
          include 'pnetcdf.inc'
          integer err
          character message*(*)

          ! It is a good idea to check returned value for possible error
          if (err .NE. NF_NOERR) then
              write(6,*) message//' '//nfmpi_strerror(err)
              call MPI_Abort(MPI_COMM_WORLD, -1, err)
          end if
      end ! subroutine check

      program main
          implicit none
          include 'mpif.h'
          include 'pnetcdf.inc'

          character*128 filename, cmd
          integer i, j, err, ierr, nprocs, rank, get_args
          integer cmode, ncid, rec_varid, fix_varid, dimid(2)
          integer no_fill, fill_value, old_mode
          integer*8 nx, ny, global_nx, global_ny, one
          integer*8 starts(2), counts(2)
          PARAMETER(nx=5, ny=3)
          integer buf(nx,ny)
          integer*8 malloc_size, sum_size
          logical verbose
          integer dummy

          call MPI_Init(err)
          call MPI_Comm_rank(MPI_COMM_WORLD, rank, err)
          call MPI_Comm_size(MPI_COMM_WORLD, nprocs, err)

          one = 1
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

          ! set parameters
          global_nx = nx * nprocs
          global_ny = ny

          do i=1, ny
          do j=1, nx
             buf(j,i) = rank
          enddo
          enddo

          ! create file, truncate it if exists
          cmode = IOR(NF_CLOBBER, NF_64BIT_DATA)
          err = nfmpi_create(MPI_COMM_WORLD, filename, cmode,
     +                       MPI_INFO_NULL, ncid)
          call check(err, 'In nfmpi_create: ')

          ! define dimensions x and y
          err = nfmpi_def_dim(ncid, "REC_DIM", NFMPI_UNLIMITED,dimid(2))
          call check(err, 'In nfmpi_def_dim REC_DIM: ')
          err = nfmpi_def_dim(ncid, "X", global_nx, dimid(1))
          call check(err, 'In nfmpi_def_dim X: ')

          ! define a 2D record variable of integer type
          err = nfmpi_def_var(ncid, "rec_var", NF_INT, 2, dimid,
     +                        rec_varid)
          call check(err, 'In nfmpi_def_var: ')

          err = nfmpi_def_dim(ncid, "Y", global_ny, dimid(2))
          call check(err, 'In nfmpi_def_dim Y: ')

          ! define a 2D fixed-size variable of integer type
          err = nfmpi_def_var(ncid, "fix_var", NF_INT, 2, dimid,
     +                        fix_varid)
          call check(err, 'In nfmpi_def_var: ')

          ! set the fill mode to NF_FILL for entire file
          err = nfmpi_set_fill(ncid, NF_FILL, old_mode)
          call check(err, 'In nfmpi_set_fill: ')
          if (verbose) then
             if (old_mode .EQ. NF_FILL) then
                 print*,"The old fill mode is NF_FILL"
             else
                 print*,"The old fill mode is NF_NOFILL"
             endif
          endif

          ! set the fill mode back to NF_NOFILL for entire file
          err = nfmpi_set_fill(ncid, NF_NOFILL, old_mode)
          call check(err, 'In nfmpi_set_fill: ')

          ! set the variable's fill mode to NF_FILL with default fill value
          err = nfmpi_def_var_fill(ncid, fix_varid, 0, NF_FILL_INT)
          call check(err, 'In nfmpi_def_var_fill: ')

          ! set a customized fill value -1
          fill_value = -1
          err = nfmpi_put_att_int(ncid, rec_varid, "_FillValue", NF_INT,
     +                            one, fill_value)
          call check(err, 'In nfmpi_put_att_int: ')

          ! do not forget to exit define mode
          err = nfmpi_enddef(ncid)
          call check(err, 'In nfmpi_enddef: ')

          ! now we are in data mode

          ! Note that in Fortran, array indices start with 1
          starts(1) = nx * rank + 1
          starts(2) = 1
          counts(1) = nx
          counts(2) = ny

          ! do not write the variable in full
          counts(1) = counts(1) - 1
          err = nfmpi_put_vara_int_all(ncid, fix_varid, starts, counts,
     +                                 buf)
          call check(err, 'In nfmpi_put_vara_int_all: ')

          err = nfmpi_inq_var_fill(ncid, fix_varid, no_fill, fill_value)
          if (no_fill .NE. 0)
     +        print*,"Error: expecting no_fill to be 0"
          if (fill_value .NE. NF_FILL_INT)
     +        print*,"Error: expecting no_fill to be ",NF_FILL_INT,
     +               " but got ", fill_value

          ! fill the 1st record of the record variable
          starts(2) = 1
          err = nfmpi_fill_var_rec(ncid, rec_varid, starts(2))
          call check(err, 'In nfmpi_fill_var_rec: ')

          ! write to the 1st record
          counts(2) = 1
          err = nfmpi_put_vara_int_all(ncid, rec_varid, starts, counts,
     +                                 buf)
          call check(err, 'In nfmpi_put_vara_int_all: ')

          ! fill the 2nd record of the record variable
          starts(2) = 2
          err = nfmpi_fill_var_rec(ncid, rec_varid, starts(2))
          call check(err, 'In nfmpi_fill_var_rec: ')

          ! write to the 2nd record
          err = nfmpi_put_vara_int_all(ncid, rec_varid, starts, counts,
     +                                 buf)

          ! close the file
          err = nfmpi_close(ncid)
          call check(err, 'In nfmpi_close: ')

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


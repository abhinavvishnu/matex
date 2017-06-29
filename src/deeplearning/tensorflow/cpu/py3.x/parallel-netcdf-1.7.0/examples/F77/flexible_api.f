!
!   Copyright (C) 2013, Northwestern University
!   See COPYRIGHT notice in top-level directory.
!
! $Id: flexible_api.f 2325 2016-02-28 07:49:13Z wkliao $

!
! This example shows how to use PnetCDF flexible API, nfmpi_put_vara_all()
! to write a 2D 4-byte integer array in parallel.
! It first defines a netCDF variable of size global_nx * global_ny where
!    global_nx == 5 and
!    global_ny == (4 * number of MPI processes).
! The data partitioning pattern is a column-wise partitioning across all
! proceses. Each process writes a subarray of size nx * ny.
! The local buffer has a ghost cell of length 3 surrounding the 2D array
!    integer buf(nx+2*ghost_len, ny+2*ghost_len)
! Note the description above follows the Fortran array index order.
!
! Example commands for MPI run and outputs from running ncmpidump on the
! NC file produced by this example program:
!
!    % mpif77 -O2 -o flexible_api flexible_api.f -lpnetcdf
!    % mpiexec -n 4 ./flexible_api /pvfs2/wkliao/testfile.nc
!
!    % ncmpidump /pvfs2/wkliao/testfile.nc
!    netcdf testfile {
!    // file format: CDF-5 (big variables)
!    dimensions:
!            x = 5 ;
!            y = 16 ;
!    variables:
!            int var(y, x) ;
!    data:
!
!     var =
!      0, 0, 0, 0, 0,
!      0, 0, 0, 0, 0,
!      0, 0, 0, 0, 0,
!      0, 0, 0, 0, 0,
!      1, 1, 1, 1, 1,
!      1, 1, 1, 1, 1,
!      1, 1, 1, 1, 1,
!      1, 1, 1, 1, 1,
!      2, 2, 2, 2, 2,
!      2, 2, 2, 2, 2,
!      2, 2, 2, 2, 2,
!      2, 2, 2, 2, 2,
!      3, 3, 3, 3, 3,
!      3, 3, 3, 3, 3,
!      3, 3, 3, 3, 3,
!      3, 3, 3, 3, 3 ;
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
          end if
      end ! subroutine check

      program main
          implicit none
          include "mpif.h"
          include "pnetcdf.inc"

          character*128 filename, cmd
          integer err, ierr, nprocs, rank, i, j, ghost_len, get_args
          integer cmode, ncid, varid, dimid(2)
          integer*8 nx, ny, global_nx, global_ny
          integer*8 starts(2), counts(2), nTypes
          PARAMETER(nx=5, ny=4, ghost_len=3)
          integer buf(nx+2*ghost_len, ny+2*ghost_len)
          integer subarray
          integer array_of_sizes(2), array_of_subsizes(2)
          integer array_of_starts(2)
          integer*8 malloc_size, sum_size
          logical verbose
          integer dummy

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

          ! set parameters
          global_nx = nx
          global_ny = ny * nprocs

          ! first initialize the entire buffer to -1
          do i=1, ny+2*ghost_len 
          do j=1, nx+2*ghost_len
             buf(j,i) = -1
          enddo
          enddo
          ! assign values for non-ghost cells
          do j=ghost_len+1, ny+ghost_len
             do i=ghost_len+1, nx+ghost_len
                 buf(i, j) = rank
             enddo
          enddo

          ! define an MPI datatype using MPI_Type_create_subarray()
          array_of_sizes(1)    = nx + 2*ghost_len
          array_of_sizes(2)    = ny + 2*ghost_len
          array_of_subsizes(1) = nx
          array_of_subsizes(2) = ny
          array_of_starts(1)   = ghost_len  ! MPI start index starts with 0
          array_of_starts(2)   = ghost_len
          call MPI_Type_create_subarray(2, array_of_sizes,
     +         array_of_subsizes, array_of_starts, MPI_ORDER_FORTRAN,
     +         MPI_INTEGER, subarray, err)
          call MPI_Type_commit(subarray, err)

          ! create file, truncate it if exists
          cmode = IOR(NF_CLOBBER, NF_64BIT_DATA)
          err = nfmpi_create(MPI_COMM_WORLD, filename, cmode,
     +                       MPI_INFO_NULL, ncid)
          call check(err, 'In nfmpi_create: ')

          ! define dimensions x and y
          err = nfmpi_def_dim(ncid, "y", global_ny, dimid(2))
          call check(err, 'In nfmpi_def_dim y: ')
          err = nfmpi_def_dim(ncid, "x", global_nx, dimid(1))
          call check(err, 'In nfmpi_def_dim x: ')

          ! define a 2D variable of integer type
          err = nfmpi_def_var(ncid, "var", NF_INT, 2, dimid, varid)
          call check(err, 'In nfmpi_def_var: ')

          ! do not forget to exit define mode
          err = nfmpi_enddef(ncid)
          call check(err, 'In nfmpi_enddef: ')

          ! now we are in data mode

          ! Note that in Fortran, array indices start with 1
          starts(1) = 1
          starts(2) = ny * rank + 1
          counts(1) = nx
          counts(2) = ny
          nTypes    = 1

          ! In Fortran, subarray buffer type can also use array index ranges
          ! for example in this case
          !    buf(1+ghost_len:nx+ghost_len, 1+ghost_len:ny+ghost_len)
          ! However, this does not work for nonblocking APIs because
          ! wait/wait_all is called separately from the nonblocking calls
          ! Fortran the subarray indexing is lost in the wait call
          err = nfmpi_put_vara_all(ncid, varid, starts, counts, buf,
     +                             nTypes, subarray)
          call check(err, 'In nfmpi_put_vara_all: ')

          call MPI_Type_free(subarray, err)

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


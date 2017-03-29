!
!  Copyright (C) 2012, Northwestern University and Argonne National Laboratory
!  See COPYRIGHT notice in top-level directory.
!
! $Id: put_varn_real.f 2224 2015-12-16 06:10:36Z wkliao $

!
! This example shows how to use a single call of nfmpi_put_varn_real_all()
! to write a sequence of one-element requests with arbitrary array indices.
!
! The compile and run commands are given below, together with an ncmpidump of
! the output file.
!
!    % mpif77 -O2 -o put_varn_real put_varn_real.f -lpnetcdf
!    % mpiexec -n 4 ./put_varn_real /pvfs2/wkliao/testfile.nc
!    % ncmpidump /pvfs2/wkliao/testfile.nc
!    netcdf testfile {
!    // file format: CDF-5 (big variables)
!    dimensions:
!             Y = 4 ;
!             X = 10 ;
!    variables:
!             int var(Y, X) ;
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

          integer NDIMS
          PARAMETER(NDIMS=2)

          character*128 filename, cmd
          integer rank, nprocs, err, num_reqs, ierr, get_args, dummy
          integer i, j, ncid, cmode, varid, dimid(2), y, x
          real buffer(13)
          integer*8 NY, NX
          integer*8 starts(NDIMS, 13)
          integer*8 counts(NDIMS, 13)
          integer*8 malloc_size, sum_size
          logical verbose

          NY = 4
          NX = 10

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

          if (nprocs .NE. 4 .AND. rank .EQ. 0 .AND. verbose)
     +        print*,'Warning: ',cmd,' is intended to run on ',
     +               '4 processes'

          ! create file, truncate it if exists
          cmode = IOR(NF_CLOBBER, NF_64BIT_DATA)
          err = nfmpi_create(MPI_COMM_WORLD, filename, cmode,
     +                       MPI_INFO_NULL, ncid)
          call check(err, 'In nfmpi_create: ')

          ! create a global array of size NY * NX */
          err = nfmpi_def_dim(ncid, "Y", NY, dimid(2))
          call check(err, 'In nfmpi_def_dim Y: ')
          err = nfmpi_def_dim(ncid, "X", NX, dimid(1))
          call check(err, 'In nfmpi_def_dim X: ')
          err = nfmpi_def_var(ncid, "var", NF_FLOAT, NDIMS, dimid,
     +                        varid)
          call check(err, 'In nfmpi_def_var var: ')
          err = nfmpi_enddef(ncid)
          call check(err, 'In nfmpi_enddef: ')

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

          ! assign arbitrary starts
          y=2
          x=1
          if (rank .EQ. 0) then
              starts(y, 1) = 1
              starts(x, 1) = 6
              starts(y, 2) = 2
              starts(x, 2) = 1
              starts(y, 3) = 3
              starts(x, 3) = 7
              starts(y, 4) = 4
              starts(x, 4) = 1
              starts(y, 5) = 1
              starts(x, 5) = 7
              starts(y, 6) = 3
              starts(x, 6) = 8
              starts(y, 7) = 4
              starts(x, 7) = 2
              starts(y, 8) = 4
              starts(x, 8) = 3
              ! rank 0 is writing the following locations: ("-" means skip)
              !   -  -  -  -  -  0  0  -  -  - 
              !   0  -  -  -  -  -  -  -  -  - 
              !   -  -  -  -  -  -  0  0  -  - 
              !   0  0  0  -  -  -  -  -  -  - 
          elseif (rank .EQ. 1) then
              starts(y,  1) = 1
              starts(x,  1) = 4
              starts(y,  2) = 1
              starts(x,  2) = 9
              starts(y,  3) = 2
              starts(x,  3) = 6
              starts(y,  4) = 3
              starts(x,  4) = 1
              starts(y,  5) = 3
              starts(x,  5) = 9
              starts(y,  6) = 4
              starts(x,  6) = 5
              starts(y,  7) = 1
              starts(x,  7) = 5
              starts(y,  8) = 1
              starts(x,  8) = 10
              starts(y,  9) = 2
              starts(x,  9) = 7
              starts(y, 10) = 3
              starts(x, 10) = 2
              starts(y, 11) = 3
              starts(x, 11) = 10
              starts(y, 12) = 4
              starts(x, 12) = 6
              starts(y, 13) = 4
              starts(x, 13) = 7
              ! rank 1 is writing the following locations: ("-" means skip)
              !   -  -  -  1  1  -  -  -  1  1 
              !   -  -  -  -  -  1  1  -  -  - 
              !   1  1  -  -  -  -  -  -  1  1 
              !   -  -  -  -  1  1  1  -  -  - 
          elseif (rank .EQ. 2) then
              starts(y, 1) = 1
              starts(x, 1) = 8
              starts(y, 2) = 2
              starts(x, 2) = 2
              starts(y, 3) = 2
              starts(x, 3) = 8
              starts(y, 4) = 3
              starts(x, 4) = 3
              starts(y, 5) = 4
              starts(x, 5) = 4
              starts(y, 6) = 2
              starts(x, 6) = 3
              starts(y, 7) = 2
              starts(x, 7) = 9
              starts(y, 8) = 2
              starts(x, 8) = 4
              starts(y, 9) = 2
              starts(x, 9) = 10
              ! rank 2 is writing the following locations: ("-" means skip)
              !   -  -  -  -  -  -  -  2  -  - 
              !   -  2  2  2  -  -  -  2  2  2 
              !   -  -  2  -  -  -  -  -  -  - 
              !   -  -  -  2  -  -  -  -  -  - 
          elseif (rank .EQ. 3) then
              starts(y,  1) = 1
              starts(x,  1) = 1
              starts(y,  2) = 2
              starts(x,  2) = 5
              starts(y,  3) = 3
              starts(x,  3) = 4
              starts(y,  4) = 4
              starts(x,  4) = 8
              starts(y,  5) = 1
              starts(x,  5) = 2
              starts(y,  6) = 3
              starts(x,  6) = 5
              starts(y,  7) = 4
              starts(x,  7) = 9
              starts(y,  8) = 1
              starts(x,  8) = 3
              starts(y,  9) = 3
              starts(x,  9) = 6
              starts(y, 10) = 4
              starts(x, 10) = 10
              ! rank 3 is writing the following locations: ("-" means skip)
              !   3  3  3  -  -  -  -  -  -  - 
              !   -  -  -  -  3  -  -  -  -  - 
              !   -  -  -  3  3  3  -  -  -  - 
              !   -  -  -  -  -  -  -  3  3  3 
          endif

          ! allocate I/O buffer and initialize its contents
          do i=1, 13
             buffer(i) = rank
             do j=1, NDIMS
                counts(j,i) = 1
             enddo
          enddo

          ! set the buffer pointers to different offsets to the I/O buffer
          err = nfmpi_put_varn_real_all(ncid, varid, num_reqs, starts,
     +                                  counts, buffer)
          call check(err, 'In nfmpi_put_varn_real_all: ')

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

      end ! program


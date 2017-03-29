!
!   Copyright (C) 2015, Northwestern University and Argonne National
!   Laboratory
!   See COPYRIGHT notice in top-level directory.
!
! $Id: put_parameter.f 2224 2015-12-16 06:10:36Z wkliao $

! This example tests PnetCDF's avoiding in-place Endianness byte swap when
! the user's write buffer is immutable, i.e. defined as PARAMETER.
!
! The compile and run commands are given below, together with an ncmpidump of
! the output file.
!
!    % mpif77 -O2 -o put_parameter put_parameter.f -lpnetcdf
!    % mpiexec -n 4 ./put_parameter /pvfs2/wkliao/testfile.nc
!    % ncmpidump /pvfs2/wkliao/testfile.nc
!    netcdf testfile {
!    // file format: CDF-2 (large file)
!    dimensions:
!            X = 4 ;
!            Y = 4 ;
!    variables:
!            int var1(Y, X) ;
!            int var2(Y, X) ;
!    data:
!    
!     var1 =
!      1, 2, 3, 4,
!      1, 2, 3, 4,
!      1, 2, 3, 4,
!      1, 2, 3, 4 ;
!    
!     var2 =
!      5, 6, 7, 8,
!      5, 6, 7, 8,
!      5, 6, 7, 8,
!      5, 6, 7, 8 ;
!    }
!
!    Note the above dump is in C order
!

       INTEGER FUNCTION XTRIM(STRING)
           CHARACTER*(*) STRING
           INTEGER I, N
           N = LEN(STRING)
           DO I = N, 1, -1
              IF (STRING(I:I) .NE. ' ') GOTO 10
           ENDDO
 10        XTRIM = I
       END ! FUNCTION XTRIM

      subroutine check(err, message)
          implicit none
          include "mpif.h"
          include "pnetcdf.inc"
          integer err, XTRIM
          character*(*) message
          character*128 msg

          ! It is a good idea to check returned value for possible error
          if (err .NE. NF_NOERR) then
              write(6,*) message(1:XTRIM(message)), nfmpi_strerror(err)
              msg = '*** TESTING F77 put_parameter.f for immutable put '
              call pass_fail(1, msg)
              call MPI_Abort(MPI_COMM_WORLD, -1, err)
          end if
      end ! subroutine check

      program main
          implicit none
          include "mpif.h"
          include "pnetcdf.inc"

          integer NX, buffer(4)
          PARAMETER(NX=4)
          data buffer /5,6,7,8/

          character*256 filename, cmd, msg
          integer err, ierr, nprocs, rank, nerrs, get_args, XTRIM
          integer cmode, ncid, varid(2), dimid(2)
          integer*8 len_ll, start(2), count(2)
          integer*8 malloc_size, sum_size

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

          call MPI_Bcast(filename, 256, MPI_CHARACTER, 0,
     +                   MPI_COMM_WORLD, ierr)

          nerrs = 0

          ! create file, truncate it if exists
          cmode = IOR(NF_CLOBBER, NF_64BIT_OFFSET)
          err = nfmpi_create(MPI_COMM_WORLD, filename, cmode,
     +                        MPI_INFO_NULL, ncid)
          call check(err, 'In nfmpi_create: ')

          ! define dimensions x and y
          len_ll = NX
          err = nfmpi_def_dim(ncid, "X", len_ll, dimid(1))
          call check(err, 'In nfmpi_def_dim X: ')

          len_ll = nprocs
          err = nfmpi_def_dim(ncid, "Y", len_ll, dimid(2))
          call check(err, 'In nfmpi_def_dim Y: ')

          ! define a 1D variable of integer type
          err = nfmpi_def_var(ncid, "var1", NF_INT, 2, dimid, varid(1))
          call check(err, 'In nfmpi_def_var: ')

          err = nfmpi_def_var(ncid, "var2", NF_INT, 2, dimid, varid(2))
          call check(err, 'In nfmpi_def_var: ')

          ! do not forget to exit define mode
          err = nfmpi_enddef(ncid)
          call check(err, 'In nfmpi_enddef: ')

          ! now we are in data mode

          start(1) = 1
          start(2) = rank + 1
          count(1) = NX
          count(2) = 1
!
! pgf77 does not like using (/1,2,3,4/) as a function argument
!          err = nfmpi_put_vara_int_all(ncid, varid(1), start, count,
!     +                                 (/1,2,3,4/))
!          call check(err, 'In nfmpi_put_var_int_all: ')
!
          err = nfmpi_put_vara_int_all(ncid, varid(2), start, count,
     +                                 buffer)
          call check(err, 'In nfmpi_put_var_int_all: ')

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

          msg = '*** TESTING F77 '//cmd(1:XTRIM(cmd))//
     +          ' for using immutable write buf '
          if (rank .eq. 0) call pass_fail(nerrs, msg)

 999      call MPI_Finalize(ierr)
      end ! program main


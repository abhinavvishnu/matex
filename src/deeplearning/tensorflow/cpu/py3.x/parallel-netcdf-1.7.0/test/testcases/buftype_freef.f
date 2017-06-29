!
!   Copyright (C) 2015, Northwestern University
!   See COPYRIGHT notice in top-level directory.
!
! $Id: buftype_freef.f 2224 2015-12-16 06:10:36Z wkliao $

! This example tests if PnetCDF duplicates the MPI derived data type supplied
! by the user, when calling the flexible APIs. It tests a PnetCDF bug
! prior to version 1.6.1.
!
! The compile and run commands:
!
!    % mpif77 -O2 -o buftype_freef buftype_freef.f -lpnetcdf
!    % mpiexec -n 4 ./buftype_freef /pvfs2/wkliao/testfile.nc
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
              msg = '*** TESTING F77 buftype_freef.f for flexible API '
              call pass_fail(1, msg)
              call MPI_Abort(MPI_COMM_WORLD, -1, err)
          end if
      end ! subroutine check

      program main
          implicit none
          include "mpif.h"
          include "pnetcdf.inc"

          integer NREQS, XTRIM
          integer*8 NX, NY
          PARAMETER(NREQS=4, NX=4, NY=4)

          character*256 filename, cmd, msg, varname, str
          integer i, err, ierr, nprocs, rank, nerrs, get_args
          integer ncid, ghost
          integer var(64,4), varid(4), dimid(2), req(4), st(4)
          integer buftype(4), gsize(2), subsize(2), a_start(2)
          integer*8 start(2), count(2)
          integer*8 one, malloc_size, sum_size

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
          err = nfmpi_create(MPI_COMM_WORLD, filename, NF_CLOBBER,
     +                        MPI_INFO_NULL, ncid)
          call check(err, 'In nfmpi_create: ')

          ! define dimensions x and y
          err = nfmpi_def_dim(ncid, "X", NX, dimid(1))
          call check(err, 'In nfmpi_def_dim X: ')
          err = nfmpi_def_dim(ncid, "Y", NY*nprocs, dimid(2))
          call check(err, 'In nfmpi_def_dim Y: ')

          ! define 2D variables of integer type
          do i=1, NREQS
             write(str,'(I1)') i
             varname = 'var'//str(1:XTRIM(str))
             err = nfmpi_def_var(ncid,varname,NF_INT,2,dimid,varid(i))
             call check(err, 'In nfmpi_def_var '//
     |                  varname(1:XTRIM(varname))//' : ')
          enddo

          ! do not forget to exit define mode
          err = nfmpi_enddef(ncid)
          call check(err, 'In nfmpi_enddef: ')

          ! Note that in Fortran, array indices start with 1
          start(1) = 1
          start(2) = NY * rank + 1
          count(1) = NX
          count(2) = NY

          do i=1, NREQS
             err = nfmpi_put_vara_int_all(ncid, varid(i), start, count,
     +                                    var(1,i))
             call check(err, 'In nfmpi_put_vara_int_all: ')
          enddo

          ! define an MPI datatype using MPI_Type_create_subarray()
          one        = 1
          ghost      = 2
          gsize(1)   = INT(NX) + 2 * ghost
          gsize(2)   = INT(NY) + 2 * ghost
          subsize(1) = INT(NX)
          subsize(2) = INT(NY)
          a_start(1) = ghost - 1
          a_start(2) = ghost - 1

          do i = 1, NREQS
              call MPI_Type_create_subarray(2, gsize, subsize, a_start,
     +             MPI_ORDER_FORTRAN, MPI_INTEGER, buftype(i), err)
              call MPI_Type_commit(buftype(i), err)

              err = nfmpi_iget_vara(ncid, varid(i), start, count,
     +                              var(1,i), one, buftype(i), req(i))
              call check(err, 'In nfmpi_iget_vara ')
              ! immediately free the data type
              call MPI_Type_free(buftype(i), err)
          enddo

          ! wait for the nonblocking I/O to complete
          err = nfmpi_wait_all(ncid, NREQS, req, st)
          call check(err, 'In nfmpi_wait_all')

          ! check the status of each nonblocking request
          do i=1, NREQS
             write(str,'(I2)') i
             call check(st(i), 'In nfmpi_wait_all req '//
     +                         str(1:XTRIM(str)))
          enddo

          ! close the file
          err = nfmpi_close(ncid)
          call check(err, 'In nfmpi_close')

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
     +          ' for flexible API '
          if (rank .eq. 0) call pass_fail(nerrs, msg)

 999      call MPI_Finalize(ierr)
      end ! program main


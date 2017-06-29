!
!   Copyright (C) 2013, Northwestern University
!   See COPYRIGHT notice in top-level directory.
!
! $Id: varn_intf.f 2224 2015-12-16 06:10:36Z wkliao $

! This example shows how to use a single call of nfmpi_put_varn_int_all() to
! write a sequence of requests with arbitrary array indices and lengths.
! Using nfmpi_put_varn_int_all() can achieve the same effect of HDF5 writing
! a sequence of selected file locations through the following 2 APIs.
!
!   H5Sselect_elements(fid, H5S_SELECT_SET, NUMP, (const hssize_t **)coord);
!   H5Dwrite(dataset, H5T_NATIVE_INT, mid, fid, H5P_DEFAULT, val); 
!
! Note that in nfmpi_put_varn_int_all(), users can write more than one element
! starting at each selected location.
!
! The compile and run commands are given below, together with an ncmpidump of
! the output file.
!
!    % mpif77 -O2 -o varn_int varn_intf.f -lpnetcdf
!    % mpiexec -n 4 ./varn_intf /pvfs2/wkliao/testfile.nc
!    % ncmpidump /pvfs2/wkliao/testfile.nc
!    netcdf testfile {
!    // file format: CDF-5 (big variables)
!    dimensions:
!             X = 4 ;
!             Y = 10 ;
!    variables:
!             int var(Y, X) ;
!    data:
!
!     var =
!      2, 2, 1, 1,
!      2, 2, 0, 0,
!      1, 1, 0, 0,
!      1, 1, 3, 3,
!      3, 3, 2, 2,
!      0, 0, 1, 1,
!      0, 0, 1, 1,
!      2, 2, 0, 0,
!      3, 3, 3, 3,
!      3, 3, 3, 3 ;
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
              msg = '*** TESTING F77 varn_intf.f for varn API '
              call pass_fail(1, msg)
              call MPI_Abort(MPI_COMM_WORLD, -1, err)
          end if
      end ! subroutine check

      program main
          implicit none
          include "mpif.h"
          include "pnetcdf.inc"

          integer NDIMS, XTRIM
          integer*8 NX, NY
          PARAMETER(NDIMS=2, NX=4, NY=10)

          character*256 filename, cmd, msg
          integer i, j, err, ierr, nprocs, rank, nerrs, get_args
          integer cmode, ncid, varid, dimid(NDIMS), num_reqs

          integer*8 w_len, w_req_len
          integer*8 starts(NDIMS, 13)
          integer*8 counts(NDIMS, 13)
          integer*8 malloc_size, sum_size
          integer buffer(13)

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

          if (.FALSE. .AND. nprocs .NE. 4 .AND. rank .EQ. 0)
     +        print*,'Warning: ',cmd(1:XTRIM(cmd)),
     +               ' is intended to run on 4 processes'

          ! create file, truncate it if exists
          cmode = IOR(NF_CLOBBER, NF_64BIT_DATA)
          err = nfmpi_create(MPI_COMM_WORLD, filename, cmode,
     +                        MPI_INFO_NULL, ncid)
          call check(err, 'In nfmpi_create: ')

          ! define dimensions x and y
          err = nfmpi_def_dim(ncid, "X", NX, dimid(1))
          call check(err, 'In nfmpi_def_dim X: ')
          err = nfmpi_def_dim(ncid, "Y", NY, dimid(2))
          call check(err, 'In nfmpi_def_dim Y: ')

          ! define a 2D variable of integer type
          err = nfmpi_def_var(ncid, "var", NF_INT, NDIMS, dimid, varid)
          call check(err, 'In nfmpi_def_var: ')

          ! do not forget to exit define mode
          err = nfmpi_enddef(ncid)
          call check(err, 'In nfmpi_enddef: ')

          ! now we are in data mode

          ! pick arbitrary numbers of requests for 4 processes
          num_reqs = 0
          if (rank .EQ. 0) then
              num_reqs = 3
          elseif (rank .EQ. 1) then
              num_reqs = 3
          elseif (rank .EQ. 2) then
              num_reqs = 3
          elseif (rank .EQ. 3) then
              num_reqs = 3
          endif

          ! assign arbitrary starts and counts
          if (rank .EQ. 0) then
              ! rank 0 is writing the followings: ("-" means skip)
              !        -  -  -  -  -  0  0  -  -  - 
              !        -  -  -  -  -  0  0  -  -  - 
              !        -  0  0  -  -  -  -  0  -  - 
              !        -  0  0  -  -  -  -  0  -  - 
              ! Note this is in Fortran order
              starts(1, 1) = 1
              starts(2, 1) = 6
              counts(1, 1) = 2
              counts(2, 1) = 2
              starts(1, 2) = 3
              starts(2, 2) = 2
              counts(1, 2) = 2
              counts(2, 2) = 2
              starts(1, 3) = 3
              starts(2, 3) = 8
              counts(1, 3) = 2
              counts(2, 3) = 1
          elseif (rank .EQ. 1) then
              ! rank 1 is writing the followings: ("-" means skip)
              !        -  -  1  1  -  -  -  -  -  - 
              !        -  -  1  1  -  -  -  -  -  - 
              !        1  -  -  -  -  1  1  -  -  - 
              !        1  -  -  -  -  1  1  -  -  - 
              ! Note this is in Fortran order
              starts(1, 1) = 1
              starts(2, 1) = 3
              counts(1, 1) = 2
              counts(2, 1) = 2
              starts(1, 2) = 3
              starts(2, 2) = 1
              counts(1, 2) = 2
              counts(2, 2) = 1
              starts(1, 3) = 3
              starts(2, 3) = 6
              counts(1, 3) = 2
              counts(2, 3) = 2
          elseif (rank .EQ. 2) then
              ! rank 2 is writing the followings: ("-" means skip)
              !        2  2  -  -  -  -  -  2  -  - 
              !        2  2  -  -  -  -  -  2  -  - 
              !        -  -  -  -  2  -  -  -  -  - 
              !        -  -  -  -  2  -  -  -  -  - 
              ! Note this is in Fortran order
              starts(1, 1) = 1
              starts(2, 1) = 1
              counts(1, 1) = 2
              counts(2, 1) = 2
              starts(1, 2) = 1
              starts(2, 2) = 8
              counts(1, 2) = 2
              counts(2, 2) = 1
              starts(1, 3) = 3
              starts(2, 3) = 5
              counts(1, 3) = 2
              counts(2, 3) = 1
          elseif (rank .EQ. 3) then
              ! rank 3 is writing the followings: ("-" means skip)
              !        -  -  -  -  3  -  -  -  3  3 
              !        -  -  -  -  3  -  -  -  3  3 
              !        -  -  -  3  -  -  -  -  3  3 
              !        -  -  -  3  -  -  -  -  3  3 
              ! Note this is in Fortran order
              starts(1, 1) = 1
              starts(2, 1) = 5
              counts(1, 1) = 2
              counts(2, 1) = 1
              starts(1, 2) = 1
              starts(2, 2) = 9
              counts(1, 2) = 4
              counts(2, 2) = 2
              starts(1, 3) = 3
              starts(2, 3) = 4
              counts(1, 3) = 2
              counts(2, 3) = 1
          endif
 
          ! w_len is total write length for this process
          w_len = 0
          do i=1, num_reqs
             w_req_len = 1
             do j=1, NDIMS
                w_req_len = w_req_len * counts(j, i)
             enddo
             w_len = w_len + w_req_len
          enddo

          ! initialize buffer contents
          do i=1, 13
             buffer(i) = rank
          enddo

          err = nfmpi_put_varn_int_all(ncid, varid, num_reqs, starts,
     +                                 counts, buffer)
          call check(err, 'In nfmpi_put_varn_int_all: ')

          if (nprocs .GT. 4) call MPI_Barrier(MPI_COMM_WORLD, err)

          ! read back and check the contents
          do i=1, 13
             buffer(i) = -1
          enddo
          err = nfmpi_get_varn_int_all(ncid, varid, num_reqs, starts,
     +                                 counts, buffer)
          call check(err, 'In nfmpi_get_varn_int_all: ')

 997      format(A,I2,A,I2,A,I2)
          do i=1, INT(w_len)
             if (buffer(i) .NE. rank) then
                 print 997, "Error: expecting buffer(",i,")=",rank,
     +                      " but got", buffer(i)
                 nerrs = nerrs + 1
             endif
          enddo

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

          msg = '*** TESTING F77 '//cmd(1:XTRIM(cmd))//' for varn API '
          if (rank .eq. 0) call pass_fail(nerrs, msg)

 999      call MPI_Finalize(ierr)
      end ! program main


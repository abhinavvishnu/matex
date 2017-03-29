!
!  Copyright (C) 2003, Northwestern University and Argonne National Laboratory
!  See COPYRIGHT notice in top-level directory.
!
!  $Id: bigrecords.f 2313 2016-02-01 21:24:42Z wkliao $

!
! this test case came from Annette Koonts at PNNL
!
! test creates a dataset with several (large) record variables, but also a
! smaller 'time' record variable.
!
! on 64 bit systems or on BlueGene (where MPI_AINT is 32 bits), this test will
! read the smaller time variable OK, but on systems with a 32 bit MPI_AINT, the
! time variable will be garbled.

       INTEGER FUNCTION XTRIM(STRING)
           CHARACTER*(*) STRING
           INTEGER I, N
           N = LEN(STRING)
           DO I = N, 1, -1
              IF (STRING(I:I) .NE. ' ') GOTO 10
           ENDDO
 10        XTRIM = I
       END ! FUNCTION XTRIM

      program main
      implicit none
      include "mpif.h"
      include "pnetcdf.inc"


! error status return
      integer  iret
! netCDF id
      integer  ncid
! dimension ids
      integer  time_dim
      integer  cells_dim
      integer  interfaces_dim
      integer  XTRIM

! dimension lengths
      integer*8  cells_len
      integer  interfaces_len

      parameter (cells_len = 41943042)
      parameter (interfaces_len = 26)

! variable ids
      integer  time_id
      integer  interfaces_id
      integer  pressure_id

      integer (kind=MPI_OFFSET_KIND) :: longlen

      integer (kind=MPI_OFFSET_KIND) :: start1d(1)
      integer (kind=MPI_OFFSET_KIND) :: count1d(1)

! rank (number of dimensions) for each variable
      integer  time_rank
      integer  interfaces_rank
      integer  pressure_rank

      parameter (time_rank = 1)
      parameter (interfaces_rank = 1)
      parameter (pressure_rank = 3)

! variable shapes
      integer  time_dims(time_rank)
      integer  interfaces_dims(interfaces_rank)
      integer  pressure_dims(pressure_rank)

! data variables
      real  interfaces(interfaces_len)

      integer  myid, err, ierr, n, get_args
      integer  numprocs

      integer*8 i8_size

      integer*8 time_start(1), time_count(1)

      double precision  time(4)
!      data time /0., 20., 40., 60./

      data interfaces /2685.8359, 671.81, 495.91, 425.10001, 393.42999, 
     + 377.5, 367.59, 360.06, 353.85999, 348.66, 342.5, 336, 328.5, 320,
     + 310, 300, 290, 280, 270, 260, 250, 240, 230, 220, 210, 199.10001/

      character*256 filename, cmd, msg

! attribute vectors
! enter define mode
!      iret = nf_create('pressure_19010101_000000.nc', OR(NF_CLOBBER,NF_64BIT_OFFSET), ncid)

      call MPI_INIT(ierr)
      call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
      call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)

      if (myid .EQ. 0) then
          filename = "testfile.nc"
          err = get_args(cmd, filename)
      endif
      call MPI_Bcast(err, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
      if (err .EQ. 0) goto 999

      call MPI_Bcast(filename, 256, MPI_CHARACTER, 0, MPI_COMM_WORLD,
     +               ierr)

      iret = nfmpi_create( MPI_COMM_WORLD, filename,
     +                       IOR(NF_CLOBBER,NF_64BIT_DATA), 
     +                       MPI_INFO_NULL, ncid)

        call check_err(cmd,"nfmpi_create(): ", iret)
  
! define dimensions

        iret = nfmpi_def_dim(ncid, 'time', NFMPI_UNLIMITED, time_dim)
        call check_err(cmd,"nfmpi_def_dim(): time ", iret)

        i8_size = 41943042

        iret = nfmpi_def_dim(ncid, 'cells', i8_size, cells_dim)
        call check_err(cmd,"nfmpi_def_dim(): cells ", iret)

        i8_size = 26
        iret = nfmpi_def_dim(ncid, 'interfaces', 
     +                       i8_size, interfaces_dim)
        call check_err(cmd,"nfmpi_def_dim(): interfaces ", iret)
! define variables
        time_dims(1) = time_dim

        iret = nfmpi_def_var(ncid, 'time', NF_DOUBLE, 
     +                       time_rank, time_dims, 
     +                       time_id)
        call check_err(cmd,"nfmpi_def_var(): time ", iret)
        interfaces_dims(1) = interfaces_dim

        iret = nfmpi_def_var(ncid, 'interfaces', NF_REAL, 
     +                       interfaces_rank, 
     +                       interfaces_dims, interfaces_id)
        call check_err(cmd,"nfmpi_def_var(): interfaces ", iret)

        pressure_dims(3) = time_dim
        pressure_dims(2) = cells_dim
        pressure_dims(1) = interfaces_dim
        iret = nfmpi_def_var(ncid, 
     +                     'pressure', 
     +                     NF_REAL, 
     +                     pressure_rank, 
     +                     pressure_dims, 
     +                     pressure_id)

        call check_err(cmd,"nfmpi_def_var(): pressure ", iret)
! assign attributes

        longlen = 4
        iret = nfmpi_put_att_text(ncid, time_id, 'long_name', 
     +                            longlen, 'Time')
        call check_err(cmd,"nfmpi_put_att_text(): long_name ", iret)
        longlen = 21
        iret = nfmpi_put_att_text(ncid, time_id, 'units', 
     +                            longlen, 
     +                           'days since 1901-01-01')
        call check_err(cmd,"nfmpi_put_att_text(): units ", iret)

        longlen = 41
        iret = nfmpi_put_att_text(ncid, interfaces_id, 'long_name', 
     +                            longlen,
     +                     'Vertical interfaces, in terms of pressure')
        call check_err(cmd,"nfmpi_put_att_text(): long_name ", iret)

        longlen = 2
        iret = nfmpi_put_att_text(ncid, interfaces_id, 'units', 
     +                            longlen, 'Pa')
        call check_err(cmd,"nfmpi_put_att_text(): units ", iret)

        longlen = 8
        iret = nfmpi_put_att_text(ncid, pressure_id, 'long_name', 
     +                            longlen, 
     1                       'Pressure')
        call check_err(cmd,"nfmpi_put_att_text(): ", iret)

        longlen = 2
        iret = nfmpi_put_att_text(ncid, pressure_id, 'units', 
     +                            longlen, 'Pa')
        call check_err(cmd,"nfmpi_put_att_text(): units ", iret)

! leave define mode
        iret = nfmpi_enddef(ncid)
        call check_err(cmd,"nfmpi_enddef(): ", iret)

        start1d(1) = 1
        count1d(1) = 26
        if (myid .GT. 0) count1d = 0

! store interfaces
        iret = nfmpi_put_vara_real_all(ncid, interfaces_id,
     +                                 start1d, count1d, interfaces)
        call check_err(cmd,"nfmpi_put_vara_real_all(): ", iret)

        time(1) = 0.0
        time(2) = 20.0
        time(3) = 40.0
        time(4) = 60.0
      
! this test is tricky because it writes out the time variable one at a time.
! This element-at-a-time workload does not actually exercise the tricky 32 bit
! MPI_AINT problem, so the issue only shows up at read time.

        do n = 1, 4
          time_start(1) = n
          if(myid .eq. 0) then
            time_count(1) = 1
          else
            time_count(1) = 0
          endif

          iret = nfmpi_put_vara_double_all(ncid, time_id, 
     +                                     time_start, time_count,
     +                                     time(n))
          call check_err(cmd,"nfmpi_put_vara_double_all(): ", iret)
        enddo

        iret = nfmpi_close(ncid)

        call MPI_Barrier (MPI_COMM_WORLD, iret)

! todo: insert code to re-open dataset, read time variable all at onece
! 
      iret = nfmpi_open ( MPI_COMM_SELF,
     +                   filename,
     +                   IOR(NF_CLOBBER,NF_64BIT_DATA),
     +                   MPI_INFO_NULL,
     +                   ncid)
      call check_err(cmd,"nfmpi_open(): ", iret)

      iret = nfmpi_inq_varid(ncid, 'time', time_id);
      call check_err(cmd,"nfmpi_inq_varid(): time ", iret)

      ! deliberately want all processes to end up with the full time array
      time_start(1) = 1
      time_count(1) = 4
      iret = nfmpi_get_vara_double_all(ncid, time_id,
     +                               time_start, time_count, time);
      call check_err(cmd,"nfmpi_get_vara_double_all(): ", iret)

      iret = nfmpi_close(ncid)
      call check_err(cmd,"nfmpi_close(): ", iret)

!     write(6,*) "Time array: ", time
!      if ( (time(1) .eq. 0) .and. (time(2) .eq. 20.0)
!     &             .and. (time(3) .eq. 40.0) .and. (time(4) .eq. 60))
!           write(6,*) " No Errors"
!      else
!           write(6,*) "Error: time array was ", time
!      endif

      msg = '*** TESTING F77 '//cmd(1:XTRIM(cmd))//' for NF_64BIT_DATA'
      if (myid .eq. 0) call pass_fail(0, msg)

 999  call MPI_FINALIZE(ierr)
      end ! program main
       
      subroutine writerecs(cmd,ncid,time_id)

      implicit none
      include "mpif.h"
      include "pnetcdf.inc"
       
      character*(*) cmd
! netCDF id
      integer  ncid
! variable ids
      integer  time_id

! error status return
      integer  iret
      integer  n
       
! netCDF dimension sizes for dimensions used with record variables
      integer  cells_len
      parameter (cells_len = 41943042)
      integer  interfaces_len
      parameter (interfaces_len = 26)
       
! rank (number of dimensions) for each variable
      integer  time_rank
      integer  pressure_rank
      parameter (time_rank = 1)
      parameter (pressure_rank = 3)
! starts and counts for array sections of record variables
      integer*8 time_start(1), time_count(1)

! data variables
       
      integer  time_nr
      parameter (time_nr = 4)
       
      integer  pressure_nr
      parameter (pressure_nr = 1)
!      real  pressure(interfaces_len, cells_len, pressure_nr)
       
      double precision  time(time_nr)
      data time /0., 20., 40., 60./
      

!      pressure = NF_FILL_FLOAT

! store time

      do n = 1, 4
        time_start(1) = n
        time_count(1) = 1
        iret = nfmpi_put_vara_double_all(ncid, time_id, 
     +                                   time_start, time_count,
     +                                   time)

        call check_err(cmd,"nfmpi_put_vara_double_all(): ", iret)
      enddo
       
      end ! subroutine writerecs
       
      subroutine check_err(cmd, msg, iret)

      include "pnetcdf.inc"

      character*(*) cmd, msg
      integer iret
      integer  XTRIM

      if (iret .ne. NF_NOERR) then
          print *, msg, nfmpi_strerror(iret)
          msg = '*** TESTING F77 '//cmd(1:XTRIM(cmd))//
     +          ' for NF_64BIT_DATA'
          call pass_fail(1, msg)
          stop
      endif
      end ! subroutine check_err

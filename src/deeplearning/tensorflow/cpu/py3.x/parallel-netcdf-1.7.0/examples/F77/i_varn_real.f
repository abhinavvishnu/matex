!
!   Copyright (C) 2014, Northwestern University
!   See COPYRIGHT notice in top-level directory.
!
! $Id: i_varn_real.f 2325 2016-02-28 07:49:13Z wkliao $

! This example tests nonblocking varn API, nfmpi_iput_varn_real()
! It writes a sequence of requests with arbitrary array indices and
! lengths to four variables of type NF_REAL
!
! The compile and run commands are given below, together with an
! ncmpidump of the output file.
!
! The compile and run commands are given below, together with an ncmpidump of
! the output file.
!
!    % mpif77 -O2 -o i_varn_real i_varn_real.f -lpnetcdf
!    % mpiexec -n 4 ./i_varn_real /pvfs2/wkliao/testfile.nc
!    % ncmpidump /pvfs2/wkliao/testfile.nc
!    netcdf testfile {
!    // file format: CDF-5 (big variables)
!     dimensions:
!              Y = 4 ;
!              X = 10 ;
!     variables:
!             float var0(Y, X) ;
!             float var1(Y, X) ;
!             float var2(Y, X) ;
!             float var3(Y, X) ;
!     data:
!
!     var0 =
!      1, 0, 3, 0,
!      1, 2, 3, 0,
!      1, 2, 2, 0,
!      3, 2, 1, 2,
!      3, 1, 1, 3,
!      0, 3, 1, 3,
!      0, 3, 0, 3,
!      2, 2, 0, 1,
!      3, 2, 3, 1,
!      3, 2, 3, 1 ;
!
!     var1 =
!      2, 1, 0, 1,
!      2, 3, 0, 1,
!      2, 3, 3, 1,
!      0, 3, 2, 3,
!      0, 2, 2, 0,
!      1, 0, 2, 0,
!      1, 0, 1, 0,
!      3, 3, 1, 2,
!      0, 3, 0, 2,
!      0, 3, 0, 2 ;
!
!     var2 =
!      3, 2, 1, 2,
!      3, 0, 1, 2,
!      3, 0, 0, 2,
!      1, 0, 3, 0,
!      1, 3, 3, 1,
!      2, 1, 3, 1,
!      2, 1, 2, 1,
!      0, 0, 2, 3,
!      1, 0, 1, 3,
!      1, 0, 1, 3 ;
!
!     var3 =
!      0, 3, 2, 3,
!      0, 1, 2, 3,
!      0, 1, 1, 3,
!      2, 1, 0, 1,
!      2, 0, 0, 2,
!      3, 2, 0, 2,
!      3, 2, 3, 2,
!      1, 1, 3, 0,
!      2, 1, 2, 0,
!      2, 1, 2, 0 ;
!     }
!
!    Note the above dump is in C order
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
          integer*8 NX, NY
          PARAMETER(NDIMS=2, NX=4, NY=10)

          character*128 filename, cmd
          integer i, j, n, err, ierr, nprocs, rank, get_args
          integer cmode, ncid, varid(4), dimid(NDIMS), nreqs
          integer reqs(4), sts(4), num_segs(4)

          integer*8 start(NDIMS, 6, 4)
          integer*8 count(NDIMS, 6, 4)
          integer*8 malloc_size, sum_size, two
          real buffer(NX*NY,4)
          logical verbose
          integer dummy, info

          call MPI_Init(err)
          call MPI_Comm_rank(MPI_COMM_WORLD, rank, err)
          call MPI_Comm_size(MPI_COMM_WORLD, nprocs, err)

          two = 2
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

          ! set an MPI-IO hint to disable file offset alignment for
          ! fix-sized variables
          call MPI_Info_create(info, err)
          call MPI_Info_set(info, "nc_var_align_size", "1", err)

          ! create file, truncate it if exists
          cmode = IOR(NF_CLOBBER, NF_64BIT_DATA)
          err = nfmpi_create(MPI_COMM_WORLD, filename, cmode,
     +                       info, ncid)
          call check(err, 'In nfmpi_create: ')

          call MPI_Info_free(info, err)

          ! define dimensions x and y
          err = nfmpi_def_dim(ncid, "Y", NY, dimid(2))
          call check(err, 'In nfmpi_def_dim Y: ')
          err = nfmpi_def_dim(ncid, "X", NX, dimid(1))
          call check(err, 'In nfmpi_def_dim X: ')

          ! define 4 2D variables of real type
          err = nfmpi_def_var(ncid, "var0", NF_REAL, two,
     +                        dimid,varid(1))
          call check(err, 'In nfmpi_def_var var0: ')
          err = nfmpi_def_var(ncid, "var1", NF_REAL, two,
     +                        dimid,varid(2))
          call check(err, 'In nfmpi_def_var var1: ')
          err = nfmpi_def_var(ncid, "var2", NF_REAL, two,
     +                        dimid,varid(3))
          call check(err, 'In nfmpi_def_var var2: ')
          err = nfmpi_def_var(ncid, "var3", NF_REAL, two,
     +                        dimid,varid(4))
          call check(err, 'In nfmpi_def_var var3: ')

          ! do not forget to exit define mode
          err = nfmpi_enddef(ncid)
          call check(err, 'In nfmpi_enddef: ')

          if (rank .GE. 4) goto 123

          ! now we are in data mode
          n = rank+1
          num_segs(n) = 4 ! number of segments for this request
          start(1,1,n)=1
          start(2,1,n)=6
          count(1,1,n)=1
          count(2,1,n)=2
          start(1,2,n)=2
          start(2,2,n)=1
          count(1,2,n)=1
          count(2,2,n)=1
          start(1,3,n)=3
          start(2,3,n)=7
          count(1,3,n)=1
          count(2,3,n)=2
          start(1,4,n)=4
          start(2,4,n)=1
          count(1,4,n)=1
          count(2,4,n)=3
          ! start(:,:,n) n_count(:,:,n) indicate the following:
          ! ("-" means skip)
          !   -  -  -  -  -  X  X  -  -  -
          !   X  -  -  -  -  -  -  -  -  -
          !   -  -  -  -  -  -  X  X  -  -
          !   X  X  X  -  -  -  -  -  -  -

          n = mod(rank+1, 4) + 1
          num_segs(n) = 6 ! number of segments for this request
          start(1,1,n)=1
          start(2,1,n)=4
          count(1,1,n)=1
          count(2,1,n)=2
          start(1,2,n)=1
          start(2,2,n)=9
          count(1,2,n)=1
          count(2,2,n)=2
          start(1,3,n)=2
          start(2,3,n)=6
          count(1,3,n)=1
          count(2,3,n)=2
          start(1,4,n)=3
          start(2,4,n)=1
          count(1,4,n)=1
          count(2,4,n)=2
          start(1,5,n)=3
          start(2,5,n)=9
          count(1,5,n)=1
          count(2,5,n)=2
          start(1,6,n)=4
          start(2,6,n)=5
          count(1,6,n)=1
          count(2,6,n)=3
          ! start(:,:,n) n_count(:,:,n) indicate the following:
          !   -  -  -  X  X  -  -  -  X  X
          !   -  -  -  -  -  X  X  -  -  -
          !   X  X  -  -  -  -  -  -  X  X
          !   -  -  -  -  X  X  X  -  -  -

          n = mod(rank+2, 4) + 1
          num_segs(n) = 5 ! number of segments for this request
          start(1,1,n)=1
          start(2,1,n)=8
          count(1,1,n)=1
          count(2,1,n)=1
          start(1,2,n)=2
          start(2,2,n)=2
          count(1,2,n)=1
          count(2,2,n)=3
          start(1,3,n)=2
          start(2,3,n)=8
          count(1,3,n)=1
          count(2,3,n)=3
          start(1,4,n)=3
          start(2,4,n)=3
          count(1,4,n)=1
          count(2,4,n)=1
          start(1,5,n)=4
          start(2,5,n)=4
          count(1,5,n)=1
          count(2,5,n)=1
          ! start(:,:,n) n_count(:,:,n) indicate the following:
          !   -  -  -  -  -  -  -  X  -  -
          !   -  X  X  X  -  -  -  X  X  X
          !   -  -  X  -  -  -  -  -  -  -
          !   -  -  -  X  -  -  -  -  -  -

          n = mod(rank+3, 4) + 1
          num_segs(n) = 4 ! number of segments for this request
          start(1,1,n)=1
          start(2,1,n)=1
          count(1,1,n)=1
          count(2,1,n)=3
          start(1,2,n)=2
          start(2,2,n)=5
          count(1,2,n)=1
          count(2,2,n)=1
          start(1,3,n)=3
          start(2,3,n)=4
          count(1,3,n)=1
          count(2,3,n)=3
          start(1,4,n)=4
          start(2,4,n)=8
          count(1,4,n)=1
          count(2,4,n)=3
          ! start(:,:,n) n_count(:,:,n) indicate the following:
          !   X  X  X  -  -  -  -  -  -  -
          !   -  -  -  -  X  -  -  -  -  -
          !   -  -  -  X  X  X  -  -  -  -
          !   -  -  -  -  -  -  -  X  X  X

          ! only rank 0, 1, 2, and 3 do I/O:
          ! each of ranks 0 to 3 write 4 nonblocking requests
 123      nreqs = 4
          if (rank .GE. 4) nreqs = 0

          ! initialize buffer contents
          do i=1, 4
          do j=1, NX*NY
             buffer(j,i) = rank
          enddo
          enddo

          do i=1, nreqs
             err = nfmpi_iput_varn_real(ncid, varid(i), num_segs(i),
     +                                  start(1,1,i), count(1,1,i),
     +                                  buffer(1,i), reqs(i))
             call check(err, 'In nfmpi_iput_varn_real: ')
          enddo
          err = nfmpi_wait_all(ncid, nreqs, reqs, sts)
          call check(err, 'In nfmpi_wait_all: ')

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
     +            sum_size, ' B yet to be freed'
          endif

 999      call MPI_Finalize(err)
          ! call EXIT(0) ! EXIT() is a GNU extension
      end ! program main



!=============================================================================
!
! Note that this code was adapted from fanc/pntf_test.F, which was written by:
! John Tannahill, LLNL
!
! This a multi-variable write test on Fortran.
!
! This code writes the array, tt(k)(j)(i), into the file 'testfile.nc'. It
! then reads the array from the file, and compares it with the original
! values.
! 
! i=longitude, j=latitude, k=level
!
! $Id: mcoll_testf.f90 2131 2015-09-25 22:33:12Z wkliao $
!
!=============================================================================

      program Mcoll_Testf

      use mpi
      use pnetcdf
      implicit none

!     -----------------------
!     Parameter declarations.
!     -----------------------

      integer NREADS, NWRITES 
      parameter (NREADS = 5, NWRITES = 5 )
      ! number of read samples
      ! number of write samples
      INTEGER(KIND=MPI_OFFSET_KIND) TOTSIZ_3D(3) ! global sizes of 3D field

!     ----------------------
!     Variable declarations.
!     ----------------------

      logical reorder
      logical isperiodic(3)
      integer comm_cart                   ! Cartesian communicator
      integer err, ierr, get_args
      INTEGER(KIND=MPI_OFFSET_KIND) istart, jstart, kstart      ! offsets of 3D field
      INTEGER(KIND=MPI_OFFSET_KIND) locsiz
      integer mype                        ! rank in comm_cart
      integer totpes                      ! total number of PEs
              
      INTEGER(KIND=MPI_OFFSET_KIND) locsiz_3d(3)  ! local sizes of 3D fields
      integer pe_coords(3)                ! Cartesian PE coords
      integer numpes(3)                   ! number of PEs along axes;
                                          !   determined by MPI where a
                                          !   zero is specified
      integer rank, Write_File
      character(len=256) :: filename, cmd, msg

      real*4  filsiz
      real*4  rdt_l(2)
      real*4  wrt_g(2)
      real*4  wrt_l(2)
              
      real*4  wrates_g(2)
      real*4  wrates_l(2)

      data reorder / .false. /
      data isperiodic / .false., .false., .false. /
      data numpes / 1, 1, 0 /
!      data TOTSIZ_3D / 256, 256, 256 /
      data TOTSIZ_3D / 8, 8, 8 /

!     ----------------
!     Begin execution.
!     ----------------

      call MPI_Init (ierr)
      call MPI_Comm_Size(MPI_COMM_WORLD, totpes, ierr)
      call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)

      if (rank .EQ. 0) then
          filename = 'testfile.nc'
          err = get_args(cmd, filename)
      endif
      call MPI_Bcast(err, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
      if (err .EQ. 0) goto 999

      call MPI_Bcast(filename, 256, MPI_CHARACTER, 0, MPI_COMM_WORLD, ierr)

      call MPI_Dims_Create (totpes, 3, numpes, ierr)

      call MPI_Cart_Create(MPI_COMM_WORLD, 3, numpes, isperiodic, &
                           reorder, comm_cart, ierr)

      call MPI_Comm_Rank (comm_cart, mype, ierr)

      call MPI_Cart_Coords (comm_cart, mype, 3, pe_coords, ierr)

      rdt_l(1) = 1.0e38
      rdt_l(2) = 1.0e38
      wrt_l(1) = 1.0e38
      wrt_l(2) = 1.0e38
!      rdt_l(:) = Huge (rdt_l)   ! initialize for timing
!      wrt_l(:) = Huge (wrt_l)

!     ----------------------------------------
!     Determine local size for tt (locsiz_3d).
!     ----------------------------------------

!     ===============
      call Find_Locnx(TOTSIZ_3D(1), pe_coords(1), numpes(1), &
                      locsiz_3d(1), istart)
      call Find_Locnx(TOTSIZ_3D(2), pe_coords(2), numpes(2), &
                      locsiz_3d(2), jstart)
      call Find_Locnx(TOTSIZ_3D(3), pe_coords(3), numpes(3), &
                      locsiz_3d(3), kstart)
!     ===============

!     -------------------------------
!     Compute file size in 1d6 bytes.
!     -------------------------------

      filsiz = real((TOTSIZ_3D(1) * TOTSIZ_3D(2) * TOTSIZ_3D(3)) * &
               1.0d-6 * 4.0d0)

!     -------------------------------------
!     Print data decomposition information.
!     -------------------------------------

!      if (mype == 0) Write (6,900)

      call MPI_Barrier (comm_cart, ierr)

!      Write (6, 902)
!     &  mype, pe_coords(1), pe_coords(2), pe_coords(3),
!     &  TOTSIZ_3D(1), TOTSIZ_3D(2), TOTSIZ_3D(3),
!     &  locsiz_3d(1), locsiz_3d(2), locsiz_3d(3),
!     &  kstart, jstart, istart

! 900  format ("mype  pe_coords    totsiz_3d         locsiz_3d       ",&
!              "kstart,jstart,istart")
! 902  format (i3,3x,i2,1x,i2,1x,i2,2x,i4,1x,i4,1x,i4,4x,i4,1x,i4,1x, &
!              i4,3x,i6,1x,i6,1x,i6)

!     -------------------------
!     Write and then read back.
!     -------------------------

      locsiz = locsiz_3d(1) * locsiz_3d(2) * locsiz_3d(3)

!     ===============
      ierr = Write_File(filename, NWRITES, comm_cart, &
                      istart, jstart, kstart, locsiz, locsiz_3d,  &
                      TOTSIZ_3D, wrt_l)
      if (ierr .NE. NF90_NOERR) then
          write(6,*) trim(nf90mpi_strerror(ierr))
          goto 999
      endif
!!!   Write (6,*) wrt_l(1), wrt_l(2)

!     ----------------------------
!     Compute and print I/O rates.
!     ----------------------------

      wrates_l(1) = filsiz / wrt_l(2)               ! write rate
      wrates_l(2) = filsiz / (wrt_l(1) + wrt_l(2))  ! effective write rate

      call MPI_Allreduce(wrates_l, wrates_g, 2, MPI_REAL, MPI_MIN, &
                         comm_cart, ierr)
      call MPI_Allreduce(wrt_l,    wrt_g,    2, MPI_REAL, MPI_MAX, &
                         comm_cart, ierr)

!      if (mype == 0) then
!        Write (6,905) filsiz
!        Write (6,910) wrates_g(1), wrates_g(2)
!      end if 

! 905  format ("File size: ", e10.3, " MB")
! 910  format ("    Write: ", f9.3, " MB/s  (eff., ", f9.3, " MB/s)")
! 915  format ("    Read : ", f9.3, " MB/s  (eff., ", f9.3, " MB/s)")
! 920  format ("Total number PEs: ", i4)
! 922  format (e11.3, e11.3, f9.3, e11.3, e11.3, f9.3)

      call MPI_Comm_Free (comm_cart, ierr)

      msg = '*** TESTING F90 '//trim(cmd)//' for nf90mpi_iput_var API'
      if (rank .EQ. 0) call pass_fail(0, msg)

 999  call MPI_Finalize(ierr)

      end program Mcoll_Testf

!     ------------


      integer function Write_File(filename, nwrites, comm_cart, &
                            istart, jstart, kstart, locsiz,  &
                            locsiz_3d, totsiz_3d, wrt_l)

      use mpi
      use pnetcdf
      implicit none

!     ----------------------
!     Argument declarations.
!     ----------------------

      character (len=*) filename
      integer nwrites
      integer comm_cart
      INTEGER(KIND=MPI_OFFSET_KIND) istart, jstart, kstart
      INTEGER(KIND=MPI_OFFSET_KIND) locsiz
      INTEGER(KIND=MPI_OFFSET_KIND) locsiz_3d(3)
      INTEGER(KIND=MPI_OFFSET_KIND) totsiz_3d(3)
      real*4  wrt_l(2)

!     ----------------------------
!     Local variable declarations.
!     ----------------------------

      integer ierr, info
      integer lon_id, lat_id, lev_id
      integer ncid
      integer nw
      integer tt1_id
      integer req(nwrites)
      integer stat(nwrites)
      INTEGER(KIND=MPI_OFFSET_KIND) start_3d(3)
      INTEGER(KIND=MPI_OFFSET_KIND) count_3d(3)
      integer dim_id(3)
      double precision  t1, t2, t3
      integer max_loc_size
      parameter( max_loc_size = 20000000 )
      ! real*4  tt1(max_loc_size)   ! Need tt(locsiz)
      real*4, dimension(locsiz_3d(1), locsiz_3d(2), locsiz_3d(3)) :: tt1

      if (locsiz .gt. MAX_LOC_SIZE) then
         print *, 'locsiz = ', locsiz, ' larger than MAX_LOC_SIZE'
         stop
      endif
!     ----------------
!     Begin execution.
!     ----------------

!      start_3d(1:3) = (/ kstart, jstart, istart /)
!      count_3d(:)   = locsiz_3d(:)
      start_3d(1) = istart
      start_3d(2) = jstart
      start_3d(3) = kstart
      count_3d(1) = locsiz_3d(1)
      count_3d(2) = locsiz_3d(2)
      count_3d(3) = locsiz_3d(3)

!       ==============
        call Get_Field(istart, jstart, kstart, locsiz_3d, &
                       totsiz_3d, tt1)

        call MPI_Barrier (comm_cart, ierr)
        t1 = MPI_Wtime ( )

!       =================
        call MPI_Info_create(info, ierr)
        ! call MPI_Info_set(info, "romio_pvfs2_posix_write","enable",ierr)

        Write_File = nf90mpi_create(comm_cart, filename, NF90_CLOBBER, &
                                    info, ncid)
        if (Write_File .NE. NF90_NOERR) return

        call MPI_Info_free(info, ierr)

!       ==================
        Write_File = nf90mpi_def_dim(ncid, "level",     totsiz_3d(1)*nwrites, &
                             lon_id)
        if (Write_File .NE. NF90_NOERR) return
        Write_File = nf90mpi_def_dim(ncid, "latitude",  totsiz_3d(2), lat_id)
        if (Write_File .NE. NF90_NOERR) return
        Write_File = nf90mpi_def_dim(ncid, "longitude", totsiz_3d(3), lev_id)
        if (Write_File .NE. NF90_NOERR) return
!       ==================

        dim_id(1) = lon_id
        dim_id(2) = lat_id
        dim_id(3) = lev_id

!       ==================
        Write_File = nf90mpi_def_var(ncid, "tt1", NF90_REAL, dim_id, tt1_id)
        if (Write_File .NE. NF90_NOERR) return

!       =================
        Write_File = nf90mpi_enddef (ncid)
        if (Write_File .NE. NF90_NOERR) return
!       =================

        t2 = MPI_Wtime ( )

      do nw = 1, nwrites
         Write_File = nf90mpi_iput_var(ncid, tt1_id, tt1, req(nw), &
                                 start_3d, count_3d)
        if (Write_File .NE. NF90_NOERR) return

         start_3d(1) = start_3d(1) + count_3d(1)
      end do

      Write_File = nf90mpi_wait_all(ncid, nwrites, req, stat)
      if (Write_File .NE. NF90_NOERR) return

!       ================
      Write_File = nf90mpi_close (ncid)
      if (Write_File .NE. NF90_NOERR) return
!       ================

! 900  format ("mynod:", i1, " reqid : ", i1)

      call MPI_Barrier (comm_cart, ierr)
      t3 = MPI_Wtime ( )

      if (t2 - t1 < wrt_l(1)) wrt_l(1) = real(t2 - t1)
      if (t3 - t2 < wrt_l(2)) wrt_l(2) = real(t3 - t2)

      end

!     ------------

      subroutine Find_Locnx(nx, mype, totpes, locnx, ibegin)

      use mpi
      implicit none

!     ----------------------
!     Argument declarations.
!     ----------------------

      INTEGER(KIND=MPI_OFFSET_KIND) nx
      integer mype
      integer totpes
      INTEGER(KIND=MPI_OFFSET_KIND) locnx
      INTEGER(KIND=MPI_OFFSET_KIND) ibegin

!     ----------------------------
!     Local variable declarations.
!     ----------------------------

      INTEGER(KIND=MPI_OFFSET_KIND) iremain

!     ----------------
!     Begin execution.
!     ----------------

      locnx = nx / totpes

      iremain = nx - (totpes * locnx)

      if (mype < iremain) locnx = locnx + 1

      ibegin = mype * (nx / totpes) + iremain + 1

      if (mype < iremain) ibegin = ibegin + (mype - iremain)

      end

!     ------------

      subroutine Get_Field(istart, jstart, kstart, locsiz_3d, &
                           totsiz_3d, tt)

      use mpi
      implicit none

!     ----------------------
!     Argument declarations.
!     ----------------------

      INTEGER(KIND=MPI_OFFSET_KIND) istart, jstart, kstart
      INTEGER(KIND=MPI_OFFSET_KIND) locsiz_3d(3)
      INTEGER(KIND=MPI_OFFSET_KIND) totsiz_3d(3)
      real*4, dimension(locsiz_3d(1),locsiz_3d(2),locsiz_3d(3)) ::tt

!     ----------------------------
!     Local variable declarations.
!     ----------------------------

      integer ii, jj, kk
      integer ind

!     ----------------
!     Begin execution.
!     ----------------

      ind = 1

      do kk = 1, int(locsiz_3d(3))
        do jj = 1, int(locsiz_3d(2))
          do ii = 1, int(locsiz_3d(1))

             tt(ii,jj,kk) = real( &
               (istart-1 +(ii - 1) + 1 + totsiz_3d(3)*(jstart-1 +  &
                       (jj - 1) + totsiz_3d(2)*(kstart-1 +  &
                       (kk-1)))) * 1.0d-3)
             ind = ind + 1

          end do
        end do
      end do

      end

!     ------------

      subroutine Compare_Vec(comm_cart, locsiz, tt, buf)

      use mpi
      implicit none

!     ----------------------
!     Argument declarations.
!     ----------------------

      integer comm_cart
      INTEGER(KIND=MPI_OFFSET_KIND) locsiz
      real*4  tt (locsiz)
      real*4  buf(locsiz)

!     ----------------------------
!     Local variable declarations.
!     ----------------------------

      integer ierr
      integer ii
      real*4  delmax(1), delmin(1), delta
      real*4  diff
      real*4  wr(5)
      real*4  ws(5)

!     ----------------
!     Begin execution.
!     ----------------

      ws(1) = 0.0d0      ! diff
      ws(2) = 0.0d0      ! sumsq
      ws(3) = real(locsiz)     ! locsiz
      ws(4) = 0.0d0      ! delmax
      ws(5) = 1.0d38     ! Huge (ws)  ! delmin

      do ii = 1, int(locsiz)
        delta = (tt(ii) - buf(ii)) * (tt(ii) - buf(ii))
        ws(1) = ws(1) + delta
        ws(2) = ws(2) + tt(ii) * tt(ii)
        if (delta > ws(4)) ws(4) = delta
        if (delta < ws(5)) ws(5) = delta
      end do

      call MPI_Allreduce(ws,    wr,     3, MPI_REAL, MPI_SUM, &
                         comm_cart, ierr)
      call MPI_Allreduce(ws(4), delmax, 1, MPI_REAL, MPI_MAX, &
                         comm_cart, ierr)
      call MPI_Allreduce(ws(5), delmin(1), 1, MPI_REAL, MPI_MIN, &
                         comm_cart, ierr)

      diff      = Sqrt (wr(1) / wr(2))         ! normalized error
      delmax(1) = Sqrt (wr(3) * delmax(1)/wr(2))  ! normalized max difference
      delmin(1) = Sqrt (wr(3) * delmin(1)/wr(2))  ! normalized min difference

      end


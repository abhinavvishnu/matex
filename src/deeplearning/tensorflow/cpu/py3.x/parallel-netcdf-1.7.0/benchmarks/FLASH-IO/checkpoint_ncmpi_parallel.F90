#define MDIM 3
! 3-d problem */
#if N_DIM == 3 
#define NDIM  3
#define NGID 15
! 2-d problem */
#elif N_DIM == 2
#define NDIM  2
#define NGID 9
! 1-d problem */
#else
#define NDIM 1
#define NGID 5
#endif

      subroutine check(err, message)
          use mpi
          use pnetcdf
          implicit none

          integer err
          character(len=*) message

          ! It is a good idea to check returned value for possible error
          write(6,*) trim(message), trim(nfmpi_strerror(err))
          call MPI_Abort(MPI_COMM_WORLD, -1, err)
      end subroutine check

      subroutine write_header_info(nvar_out, ncid, file_creation_time, &
                                   flash_version, total_blocks, time, &
                                   nsteps, nzones_block, unk_labels, &
                                   varid)
          use mpi
          use pnetcdf
          implicit none

          integer nvar_out                    ! num vars to store
          integer ncid                        ! file handle
          character(len=*) file_creation_time ! time and date stamp
          character(len=*) flash_version      ! FLASH version num
          integer total_blocks                ! total # of blocks
          double precision time               ! simulation time
          integer nsteps                      ! total # of timestep
          integer nzones_block(3)             ! nxb, nyb, nzb
          character(len=4) unk_labels(*)      ! unknown labels
          integer varid(*)                    ! output: var ids

          ! local variables
          integer i, k, err
          integer dim_tot_blocks, dim_nxb, dim_nyb, dim_nzb
          integer dim_NGID, dim_NDIM, dim_2
          integer dimids(4)
          character(len=5) record_label
          integer(kind=MPI_OFFSET_KIND) i8NDIM, i8NGID, i8total_blocks
          integer(kind=MPI_OFFSET_KIND) i8nzones_block(3), string_size
          integer atotal_blocks(1), ansteps(1)
          double precision atime(1)

          i8NDIM = NDIM
          i8NGID = NGID
          i8total_blocks = total_blocks
          i8nzones_block(:) = nzones_block(:)
          atotal_blocks(1) = total_blocks
          ansteps(1) = nsteps

          ! to avoid inconsistent header metadata warning from PnetCDF
          atime(1) = time
          call MPI_Bcast(atime, 1, MPI_DOUBLE_PRECISION, 0, MPI_COMM_WORLD, err)

          err = nfmpi_def_dim(ncid, "dim_tot_blocks", i8total_blocks, dim_tot_blocks)
          if (err .NE. NF_NOERR) call check(err, "nfmpi_def_dim: dim_tot_blocks")
          err = nfmpi_def_dim(ncid, "dim_nxb", i8nzones_block(1), dim_nxb)
          if (err .NE. NF_NOERR) call check(err, "nfmpi_def_dim: dim_nxb")
          err = nfmpi_def_dim(ncid, "dim_nyb", i8nzones_block(2), dim_nyb)
          if (err .NE. NF_NOERR) call check(err, "nfmpi_def_dim: dim_nyb")
          err = nfmpi_def_dim(ncid, "dim_nzb", i8nzones_block(3), dim_nzb)
          if (err .NE. NF_NOERR) call check(err, "nfmpi_def_dim: dim_nzb")
          err = nfmpi_def_dim(ncid, "dim_NGID", i8NGID, dim_NGID)
          if (err .NE. NF_NOERR) call check(err, "nfmpi_def_dim: dim_NGID")
          err = nfmpi_def_dim(ncid, "dim_NDIM", i8NDIM, dim_NDIM)
          if (err .NE. NF_NOERR) call check(err, "nfmpi_def_dim: dim_NDIM")
          err = nfmpi_def_dim(ncid, "dim_2", 2_MPI_OFFSET_KIND, dim_2)
          if (err .NE. NF_NOERR) call check(err, "nfmpi_def_dim: dim_2")

          dimids(1) = dim_tot_blocks

          ! define var for refinement level
          err = nfmpi_def_var(ncid, "lrefine", NF_INT, 1, dimids, varid(1))
          if (err .NE. NF_NOERR) call check(err, "nfmpi_def_var: lrefine")

          ! define var for nodetype
          err = nfmpi_def_var(ncid, "nodetype", NF_INT, 1, dimids, varid(2))
          if (err .NE. NF_NOERR) call check(err, "nfmpi_def_var: nodetype")

          ! define var for global id
          dimids(1) = dim_NGID
          dimids(2) = dim_tot_blocks
          err = nfmpi_def_var(ncid, "gid", NF_INT, 2, dimids, varid(3))
          if (err .NE. NF_NOERR) call check(err, "nfmpi_def_var: grid")

          ! define var for grid coordinates
          dimids(1) = dim_NDIM
          dimids(2) = dim_tot_blocks
          err = nfmpi_def_var(ncid, "coordinates", NF_DOUBLE, 2, dimids, varid(4))
          if (err .NE. NF_NOERR) call check(err, "nfmpi_def_var: coordinates")

          ! define var for grid block size
          dimids(1) = dim_NDIM
          dimids(2) = dim_tot_blocks
          err = nfmpi_def_var(ncid, "blocksize", NF_DOUBLE, 2, dimids, varid(5))
          if (err .NE. NF_NOERR) call check(err, "nfmpi_def_var: blocksize")

          ! define var for grid bounding box
          dimids(1) = dim_2
          dimids(2) = dim_NDIM
          dimids(3) = dim_tot_blocks
          err = nfmpi_def_var(ncid, "bndbox", NF_DOUBLE, 3, dimids, varid(6))
          if (err .NE. NF_NOERR) call check(err, "nfmpi_def_var: bndbox")

          ! define var for unknown array
          dimids(1) = dim_nxb    
          dimids(2) = dim_nyb
          dimids(3) = dim_nzb
          dimids(4) = dim_tot_blocks

          do i=1, nvar_out
              record_label = unk_labels(i)
              do k=1, 4
                 if (record_label(k:k) .EQ. ' ') record_label(k:k) = '_'
              enddo
              record_label(5:5) = char(0)  ! string terminate char
              err = nfmpi_def_var(ncid, record_label, NF_DOUBLE, 4, dimids, varid(i+6))
              if (err .NE. NF_NOERR) call check(err, "nfmpi_def_var: record_label")
          enddo

          string_size = LEN_TRIM(file_creation_time)
          err = nfmpi_put_att_text(ncid, NF_GLOBAL, "file_creation_time", string_size, file_creation_time)
          if (err .NE. NF_NOERR) call check(err, "nfmpi_put_att_text: file_creation_time")
          string_size = LEN_TRIM(flash_version)
          err = nfmpi_put_att_text(ncid, NF_GLOBAL, "flash_version",  string_size, flash_version)
          if (err .NE. NF_NOERR) call check(err, "nfmpi_put_att_text: flash_version")
          err = nfmpi_put_att_int(ncid, NF_GLOBAL, "total_blocks",  NF_INT, 1_MPI_OFFSET_KIND, atotal_blocks)
          if (err .NE. NF_NOERR) call check(err, "nfmpi_put_att_int: total_blocks")
          err = nfmpi_put_att_int(ncid, NF_GLOBAL, "nsteps", NF_INT, 1_MPI_OFFSET_KIND, ansteps)
          if (err .NE. NF_NOERR) call check(err, "nfmpi_put_att_int: nsteps")
          err = nfmpi_put_att_int(ncid, NF_GLOBAL, "nxb", NF_INT, 1_MPI_OFFSET_KIND, nzones_block(1))
          if (err .NE. NF_NOERR) call check(err, "nfmpi_put_att_int: nxb")
          err = nfmpi_put_att_int(ncid, NF_GLOBAL, "nyb", NF_INT, 1_MPI_OFFSET_KIND, nzones_block(2))
          if (err .NE. NF_NOERR) call check(err, "nfmpi_put_att_int: nyb")
          err = nfmpi_put_att_int(ncid, NF_GLOBAL, "nzb", NF_INT, 1_MPI_OFFSET_KIND, nzones_block(3))
          if (err .NE. NF_NOERR) call check(err, "nfmpi_put_att_int: nzb")
          err = nfmpi_put_att_double(ncid, NF_GLOBAL, "time", NF_DOUBLE, 1_MPI_OFFSET_KIND, atime)
          if (err .NE. NF_NOERR) call check(err, "nfmpi_put_att_double: time")
      
          err = nfmpi_enddef(ncid)
          if (err .NE. NF_NOERR) call check(err, "nfmpi_enddef")
      end subroutine write_header_info

!----------------------------------------------------------------------------
! subroutine checkpoint_wr
!----------------------------------------------------------------------------

      double precision function checkpoint_wr_ncmpi_par (filenum, simtime)
!
! Do parallel i/o using Parallel netCDF
!
! MZ -- 2-20-00
! Jianwei -- 11/15/02
!
! This version of checkpoint uses Parallel netCDF to store the PARAMESH data.
! The IO is done in parallel -- no copying of the data to a single processor
! to do the writing is performed.  
!
! Parallel netCDF uses MPI-IO (via ROMIO) to support parallel IO. Each
! processor must open the file, define the dataspaces for each netCDF
! variables.  
!
! A single record for each of the PARAMESH data structures is created.  A
! processor only writes to a subset of this record.  Each record has a 
! dimension with length = tot_blocks.  The offset of a processor into this 
! dimension is computed by looking at the total number of blocks that are
! below the current processor.
!
! In this version of the checkpoint, each variable is given its own 
! record -- this makes it easier to change the variable list in the
! future without disturbing the format of the file.
!
! The include file -- ncmpi_flash.h is used for the C routines and mirrors
! the necessary data from physicaldata.fh
!
! written for netCDF 3.0
!
!---------------------------------------------------------------------------

      use pnetcdf

#include "common.fh"

      integer filenum
      double precision simtime
      
      integer block_no
      integer i, j
      integer ngid 
      integer err

      integer n_to_left(0:16383)  ! must extend from 0 to NumPEs-1

! 2-20-00 -- we don't need to allocate more space than necessary
!      integer gid(mfaces+1+mchild,maxblocks_tr)
      integer gid(nfaces+1+nchild,maxblocks_tr)

      integer tot_blocks

      save gid, n_to_left

      integer nzones_block(3)

! create a character variable to hold the string representation of the block
! number.  Note this is set to be 4 characters long (i.e. max = 9999).  
      character*4  fnum_string
      character*80 filename
      character*8 str

! create a temporary array to hold the 4 character variable names
! this will include those defined in definitions.fh and network_common.fh
      character*4 unklabels(nvar)

! storage for the current date and time
        character date_string*40

      character(len=4) :: ionam(ionmax), record_label

      integer varid(6+nvar)

      integer global_offset

      character (len=40) :: flash_release
      double precision unk_buf(1,nxb,nyb,nzb,maxblocks)
#ifdef TIMERS
      double precision time_start, time_io
#endif
      double precision coord_buf(ndim,lnblocks)
      double precision bs_buf(ndim,lnblocks)
      double precision bb_buf(2,ndim,lnblocks)

      integer ncid, cmode, file_info
      integer(kind=MPI_OFFSET_KIND) starts(5), counts(4), put_size
      integer gsizes(5), subsizes(5), buftype, reqs(nvar+6), stats(nvar+6)

!-----------------------------------------------------------------------------
! compute the total number of blocks left of a given processor number
!-----------------------------------------------------------------------------
      chk_t(1) = MPI_Wtime()

! use an allgather routine here 
      call MPI_Allgather(lnblocks, 1,MPI_INTEGER, & 
                         n_to_left,1,MPI_INTEGER, & 
                         MPI_COMM_WORLD,err)

      
! compute the total number of blocks
      tot_blocks = 0

      do i = 0,NumPEs-1
         tot_blocks = tot_blocks + n_to_left(i)
      end do
      
! compute the number of procssors to the left of a processor
      do i = NumPEs-1,1,-1
         n_to_left(i) = n_to_left(i-1)
      end do

      n_to_left(0) = 0
      do i = 2,NumPEs-1
         n_to_left(i) = n_to_left(i) + n_to_left(i-1)
      end do


!-----------------------------------------------------------------------------
! compute the global id -- this is a single array which stores the 
! neighbor block numbers, the parent, and the children of a given block
!-----------------------------------------------------------------------------
      do block_no = 1,lnblocks

         ngid = 0

! loop over the faces and store the neighbors
         do j = 1,nfaces
            ngid = ngid + 1

! if the neighbor exists, then store the block number of the neighbor
! -- take into account the number of blocks below the processor that the
! neighbor is on, so the block number is global
            if (neigh(1,j,block_no).gt.0) then
               gid(ngid,block_no) = neigh(1,j,block_no) +  & 
                    n_to_left(neigh(2,j,block_no))
            else

! the neighbor is either a physical boundary or does not exist at that 
! level of refinement
               gid(ngid,block_no) = neigh(1,j,block_no)
            end if
         end do
        
! store the parent of the current block
         ngid = ngid + 1
         if (parent(1,block_no).gt.0) then
            gid(ngid,block_no) = parent(1,block_no) +  & 
                 n_to_left(parent(2,block_no))
         else
            gid(ngid,block_no) = parent(1,block_no)
         end if
         
! store the children of the current block
         do j = 1,nchild
            ngid = ngid + 1
            if (child(1,j,block_no).gt.0) then
               gid(ngid,block_no) = child(1,j,block_no) +  & 
                    n_to_left(child(2,j,block_no))
            else
               gid(ngid,block_no) = child(1,j,block_no)
            end if
         end do

      end do

!-----------------------------------------------------------------------------
! open the netCDF file
!-----------------------------------------------------------------------------
      write (fnum_string, '(i4.4)') filenum
      filename = trim(basenm) // 'ncmpi_chk_'//fnum_string//'.nc'

      ! set up MPI I/O hints for performance enhancement
      file_info = MPI_INFO_NULL
      call MPI_Info_create(file_info, err)

      ! set some ROMIO hints
      ! call MPI_Info_set(file_info, 'romio_no_indep_rw', 'true', err)

      ! disable file offset alignment for fix-sized variables
      call MPI_Info_set(file_info, "nc_var_align_size", "1", err)

      cmode = IOR(NF_CLOBBER, NF_64BIT_DATA)
      err = nfmpi_create(MPI_COMM_WORLD, trim(filename), cmode, &
                         file_info, ncid)
      if (err .NE. NF_NOERR) call check(err, "nfmpi_create")

      call MPI_Info_free(file_info, err)

      err = nfmpi_get_file_info(ncid, info_used)

!-----------------------------------------------------------------------------
! store the scalar information -- # of blocks, simulation time, etc
!-----------------------------------------------------------------------------
      
! get the current time and date
      date_string = 'now'      

! store the number of zones / block in each direction
      nzones_block(1) = nxb
      nzones_block(2) = nyb
      nzones_block(3) = nzb

! get the names of the fluids being followed
      call get_mfluid_property ("short name", ionam)

! merge the two variable lists into one for storage
      unklabels(1:nvar-nuc2)     = varnam(:)
      unklabels(nvar-nuc2+1:nvar) = ionam(:)

#ifdef TIMERS
      time_start = MPI_Wtime()
#endif

      call write_header_info(nvar, & 
                             ncid, & 
                             date_string, &
                             flash_release(), &
                             tot_blocks, & 
                             simtime, & 
                             nstep, & 
                             nzones_block, & 
                             unklabels, &
                             varid)

#ifdef TIMERS
      print *, 'header: MyPE = ', MyPE, ' time = ',  & 
           MPI_Wtime() - time_start
#endif

      global_offset = n_to_left(MyPE)

!-----------------------------------------------------------------------------
! store the tree information
!-----------------------------------------------------------------------------

! store the refinement level
#ifdef TIMERS
      time_start = MPI_Wtime()
#endif

      starts(1) = global_offset+1
      counts(1) = lnblocks
      if (use_nonblocking_io) then
          err = nfmpi_iput_vara_int(ncid, varid(1), starts, counts, lrefine, reqs(1))
          if (err .NE. NF_NOERR) call check(err, "nfmpi_iput_vara_int: lrefine")
      else
          err = nfmpi_put_vara_int_all(ncid, varid(1), starts, counts, lrefine)
          if (err .NE. NF_NOERR) call check(err, "nfmpi_put_vara_int_all: lrefine")
      endif

#ifdef TIMERS
      print *, 'lrefine: MyPE = ', MyPE, ' time = ',  & 
           MPI_Wtime() - time_start
#endif

! store the nodetype
#ifdef TIMERS
      time_start = MPI_Wtime()
#endif      

      if (use_nonblocking_io) then
          err = nfmpi_iput_vara_int(ncid, varid(2), starts, counts, nodetype, reqs(2))
          if (err .NE. NF_NOERR) call check(err, "nfmpi_put_vara_int: nodetype")
      else
          err = nfmpi_put_vara_int_all(ncid, varid(2), starts, counts, nodetype)
          if (err .NE. NF_NOERR) call check(err, "nfmpi_put_vara_int_all: nodetype")
      endif

#ifdef TIMERS
      print *, 'nodetype: MyPE = ', MyPE, ' time = ',  & 
           MPI_Wtime() - time_start
#endif

! store the global id
#ifdef TIMERS
      time_start = MPI_Wtime()
#endif      

      starts(1) = 1
      starts(2) = global_offset+1
      counts(1) = NGID
      counts(2) = lnblocks
      if (use_nonblocking_io) then
          err = nfmpi_iput_vara_int(ncid, varid(3), starts, counts, gid, reqs(3))
          if (err .NE. NF_NOERR) call check(err, "nfmpi_iput_vara_int: gid")
      else
          err = nfmpi_put_vara_int_all(ncid, varid(3), starts, counts, gid)
          if (err .NE. NF_NOERR) call check(err, "nfmpi_put_vara_int_all: gid")
      endif

#ifdef TIMERS
      print *, 'gid: MyPE = ', MyPE, ' time = ',  & 
           MPI_Wtime() - time_start
#endif

!-----------------------------------------------------------------------------
! store the grid information
!-----------------------------------------------------------------------------
      
! store the coordinates
#ifdef TIMERS
      time_start = MPI_Wtime()
#endif
      coord_buf(1:ndim,1:lnblocks) = coord(1:ndim,1:lnblocks)

      starts(1) = 1
      starts(2) = global_offset+1
      counts(1) = NDIM
      counts(2) = lnblocks
      if (use_nonblocking_io) then
          err = nfmpi_iput_vara_double(ncid, varid(4), starts, counts, coord_buf, reqs(4))
          if (err .NE. NF_NOERR) call check(err, "nfmpi_iput_vara_double: coord")
      else
          err = nfmpi_put_vara_double_all(ncid, varid(4), starts, counts, coord_buf)
          if (err .NE. NF_NOERR) call check(err, "nfmpi_put_vara_double_all: coord")
      endif

#ifdef TIMERS
      print *, 'coord: MyPE = ', MyPE, ' time = ',  & 
           MPI_Wtime() - time_start
#endif

! store the block size
#ifdef TIMERS
      time_start = MPI_Wtime()
#endif
      bs_buf(1:ndim,1:lnblocks) = size(1:ndim,1:lnblocks)

      starts(1) = 1
      starts(2) = global_offset+1
      counts(1) = NDIM
      counts(2) = lnblocks
      if (use_nonblocking_io) then
          err = nfmpi_iput_vara_double(ncid, varid(5), starts, counts, bs_buf, reqs(5))
          if (err .NE. NF_NOERR) call check(err, "nfmpi_iput_vara_double: size")
      else
          err = nfmpi_put_vara_double_all(ncid, varid(5), starts, counts, bs_buf)
          if (err .NE. NF_NOERR) call check(err, "nfmpi_put_vara_double_all: size")
      endif

#ifdef TIMERS
      print *, 'size: MyPE = ', MyPE, ' time = ',  & 
           MPI_Wtime() - time_start
#endif

! store the bounding box 
#ifdef TIMERS
      time_start = MPI_Wtime()
#endif      
      bb_buf(1:2,1:ndim,1:lnblocks) = bnd_box(1:2,1:ndim,1:lnblocks)

      starts(1) = 1
      starts(2) = 1
      starts(3) = global_offset+1
      counts(1) = 2
      counts(2) = NDIM
      counts(3) = lnblocks
      if (use_nonblocking_io) then
          err = nfmpi_iput_vara_double(ncid, varid(6), starts, counts, bb_buf, reqs(6))
          if (err .NE. NF_NOERR) call check(err, "nfmpi_iput_vara_double: bnd_box")
      else
          err = nfmpi_put_vara_double_all(ncid, varid(6), starts, counts, bb_buf)
          if (err .NE. NF_NOERR) call check(err, "nfmpi_put_vara_double_all: bnd_box")
      endif

#ifdef TIMERS
      print *, 'bb1: MyPE = ', MyPE, ' time = ',  & 
           MPI_Wtime() - time_start
#endif

      chk_t(2) = MPI_Wtime()
      chk_t(1) = chk_t(2) - chk_t(1)

!-----------------------------------------------------------------------------
! store the unknowns -- here we will pass the entire unk array on each 
! processor.  The HDF 5 memory space functionality will pick just the 
! interior cells to write to disk.
!-----------------------------------------------------------------------------
#ifdef TIMERS
      time_io = 0.e0
      time_start = MPI_Wtime()
#endif

      if (use_nonblocking_io) then
         ! create an MPI derived data type for buffer unk
         gsizes(1) = nvar
         gsizes(2) = iu_bnd - il_bnd + 1
         gsizes(3) = ju_bnd - jl_bnd + 1
         gsizes(4) = ku_bnd - kl_bnd + 1
         gsizes(5) = maxblocks
         subsizes(1) = 1
         subsizes(2) = nxb
         subsizes(3) = nyb
         subsizes(4) = nzb
         subsizes(5) = lnblocks
         starts(1) = 0
         starts(2) = nguard
         starts(3) = nguard*k2d
         starts(4) = nguard*k3d
         starts(5) = 0
         call MPI_Type_create_subarray(5, gsizes, subsizes, starts, &
                                       MPI_ORDER_FORTRAN, &
                                       MPI_DOUBLE_PRECISION, buftype, &
                                       err)
         call MPI_Type_commit(buftype, err)
      endif

      starts(1) = 1
      starts(2) = 1
      starts(3) = 1
      starts(4) = global_offset+1
      counts(1) = nxb
      counts(2) = nyb
      counts(3) = nzb
      counts(4) = lnblocks

      do i = 1, nvar
         record_label = unklabels(i)

         if (.NOT. use_nonblocking_io) then
            ! when using nonblocking flexible API, we don't even need unk_buf
            unk_buf(1, 1:nxb, 1:nyb, 1:nzb, :) =        &
                unk(i, nguard+1     : nguard+nxb,       & 
                       nguard*k2d+1 : nguard*k2d+nyb,   & 
                       nguard*k3d+1 : nguard*k3d+nzb, :)
         endif

         if (use_nonblocking_io) then
            err = nfmpi_iput_vara(ncid, varid(6+i), starts, counts, &
                                  unk(i, 1, 1, 1, 1), 1_MPI_OFFSET_KIND, buftype, reqs(i+6))
            if (err .NE. NF_NOERR) &
                call check(err, "nfmpi_iput_vara: unknowns")
         else
            err = nfmpi_put_vara_double_all(ncid, varid(6+i), starts, counts, unk_buf)
            if (err .NE. NF_NOERR) &
                call check(err, "nfmpi_put_vara_double_all: unknowns")
         endif
      enddo

      if (use_nonblocking_io) then
          ! wait for the nonblocking I/O to complete
          err = nfmpi_wait_all(ncid, nvar+6, reqs, stats)
          if (err .NE. NF_NOERR) &
              call check(err, "nfmpi_wait_all: unknowns")

          ! check the status of each nonblocking request
          do i=1, nvar+6
             write(str,'(I2)') i
             if (stats(i) .NE. NF_NOERR) &
                 call check(stats(i), 'In nfmpi_wait_all req '//trim(str))
          enddo
          call MPI_Type_free(buftype, err)
      endif

#ifdef TIMERS
      time_io = time_io + (MPI_Wtime() - time_start)
      print *, 'unk: MyPE = ', MyPE, ' time = ', time_io
#endif

!-----------------------------------------------------------------------------
! close the file
!-----------------------------------------------------------------------------

      chk_t(3) = MPI_Wtime()
      chk_t(2) = chk_t(3) - chk_t(2)

      err = nfmpi_inq_put_size(ncid, put_size)
      if (err .NE. NF_NOERR) call check(err, "nfmpi_inq_put_size")

      err = nfmpi_close(ncid);
      if (err .NE. NF_NOERR) call check(err, "nfmpi_close")

      chk_t(3) = MPI_Wtime() - chk_t(3)

      checkpoint_wr_ncmpi_par = put_size

      return
      end




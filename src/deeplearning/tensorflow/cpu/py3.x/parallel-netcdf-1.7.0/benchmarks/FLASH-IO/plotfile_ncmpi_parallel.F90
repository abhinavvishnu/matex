#define MDIM 3
! 3-d problem */
#if N_DIM == 3 
#define NDIM  3
#define NGID 15
#define ik2d 1
#define ik3d 1
! 2-d problem */
#elif N_DIM == 2
#define NDIM  2
#define NGID 9
#define ik2d 1
#define ik3d 0
! 1-d problem */
#else
#define NDIM 1
#define NGID 5
#define ik2d 0
#define ik3d 0
#endif


      subroutine write_header_info_sp(nvar_out, ncid, file_creation_time, &
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
          real    time                        ! simulation time
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
          err = nfmpi_def_var(ncid, "coordinates", NF_FLOAT, 2, dimids, varid(4))
          if (err .NE. NF_NOERR) call check(err, "nfmpi_def_var: coordinates")

          ! define var for grid block size
          dimids(1) = dim_NDIM
          dimids(2) = dim_tot_blocks
          err = nfmpi_def_var(ncid, "blocksize", NF_FLOAT, 2, dimids, varid(5))
          if (err .NE. NF_NOERR) call check(err, "nfmpi_def_var: blocksize")

          ! define var for grid bounding box
          dimids(1) = dim_2
          dimids(2) = dim_NDIM
          dimids(3) = dim_tot_blocks
          err = nfmpi_def_var(ncid, "bndbox", NF_FLOAT, 3, dimids, varid(6))
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
              record_label(5:5) = char(0)
              err = nfmpi_def_var(ncid, record_label, NF_FLOAT, 4, dimids, varid(i+6))
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
      end subroutine write_header_info_sp

!----------------------------------------------------------------------------
! subroutine plotfile
!----------------------------------------------------------------------------

      double precision function plotfile_ncmpi_par(filenum, simtime, corners)
!
! plotfile using parallel i/o using Parallel netCDF
!
! MZ -- 4-29-00
! Jianwei -- 11/15/02
!
! This version of the plotfile routine is based on the Parallel netCDF
! checkpoint.  The IO is done in parallel -- no copying of the data to 
! a single processor to do the writing is performed.  
!
! This is the SINGLE PRECISION version of the plotfile -- temporary
! storage is used to recast a variable (for every zone/block) into
! single precision before passing it onto the SP version of the C netCDF
! write routines.
! 
! The data for all blocks is recast and written together.  This makes the
! amount of data that is written very large, which should perform better
! on the parallel filesystems.  The overhead for storing an entire 
! variable (with corners) is small, <~ 1%.
!
! Parallel netCDF uses MPI-IO (via ROMIO) to support parallel IO.  Each 
! processor must open the file, define the dataspaces for each netCDF variable.
!
! A single record for each of the PARAMESH data structures is created.  A
! processor only writes to a subset of this record.  Each record has a 
! dimension with length = tot_blocks.  The offset of a processor into this 
! dimension is computed by looking at the total number of blocks that are
! below the current processor.
!
! In this version of the plotfile, each variable is given its own 
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
      real simtime
      
      integer block_no
      integer i, j, k, ivar, i_store, j_store, k_store
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

! set the number of variables we are going to write out
      integer, parameter ::  num_out = 4

! create a temporary array to hold the 4 character variable names
! this will include those defined in definitions.fh and network_common.fh
      character*4 unklabels(nvar), sunklabels(num_out)

! storage for the current date and time
      character date_string*40

      character(len=4) :: ionam(ionmax), record_label
      
      integer varid(6+num_out)

      integer global_offset


! hold pointers to the location in unk of the variables we are writing out
      integer iout(num_out)

! allocate storage to hold a single variable information
! this should only be a small memory overhead
      integer, parameter :: single = SELECTED_REAL_KIND(p=6)
      real (kind=single) :: unkt_crn(1,nxb+1,nyb+k2d,nzb+k3d,maxblocks)
      real (kind=single) :: unkt(1,nxb,nyb,nzb,maxblocks)

! allocate storage to hold the coordinate information and bounding box
! information
      real (kind=single) :: coord_single(mdim,maxblocks_tr)
      real (kind=single) :: blk_sz_single(mdim,maxblocks_tr)
      real (kind=single) :: bnd_single(2,mdim,maxblocks_tr)
      real (kind=single) :: sp_var1, sp_var2

      integer, parameter :: release_len = 40
      character (len=release_len) :: flash_release

      logical corners

      integer ncid, cmode, file_info, reqs(num_out+6), stats(num_out+6)
      integer(kind=MPI_OFFSET_KIND) starts(4), counts(4), put_size, buf_size

      if (corners) then
         corner_t(1) = MPI_Wtime()
      else
         nocorner_t(1) = MPI_Wtime()
      endif

!-----------------------------------------------------------------------------
! set the variables we are going to store
!-----------------------------------------------------------------------------
      iout(1) = idens
      iout(2) = itemp
      iout(3) = ipres

! store the first abundance
      iout(4) = inuc_begin


!-----------------------------------------------------------------------------
! compute the total number of blocks left of a given processor number
!-----------------------------------------------------------------------------

! use an allgather routine here to get the number of blocks on each proc.
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

      if (corners) then
         filename = trim(basenm)//'ncmpi_plt_crn_'//fnum_string//'.nc'
      else
         filename = trim(basenm)//'ncmpi_plt_cnt_'//fnum_string//'.nc'
      endif

      ! set up MPI I/O hints for performance enhancement
      file_info = MPI_INFO_NULL
      call MPI_Info_create(file_info, err)

      ! use some ROMIO hints
      ! call MPI_Info_set(file_info, 'romio_no_indep_rw', 'true', err)

      ! disable file offset alignment for fix-sized variables
      call MPI_Info_set(file_info, "nc_var_align_size", "1", err)

      cmode = IOR(NF_CLOBBER, NF_64BIT_DATA)
      err = nfmpi_create(MPI_COMM_WORLD, trim(filename), cmode, &
                         file_info, ncid)
      if (err .NE. NF_NOERR) call check(err, "nfmpi_create")

      call MPI_Info_free(file_info, err)

!-----------------------------------------------------------------------------
! store the scalar information -- # of blocks, simulation time, etc
!-----------------------------------------------------------------------------
        date_string = 'now'      

! store the number of zones / block in each direction
      if (corners) then
         nzones_block(1) = nxb+1
         nzones_block(2) = nyb+k2d
         nzones_block(3) = nzb+k3d
      else
         nzones_block(1) = nxb
         nzones_block(2) = nyb
         nzones_block(3) = nzb
      endif

! get the names of the fluids being followed
      call get_mfluid_property ("short name", ionam)

! merge the two variable lists into one for storage
      unklabels(1:nvar-ionmax)     = varnam(:)
      unklabels(nvar-ionmax+1:nvar) = ionam(:)

! get the subset of the variable labels, corresponding to what we are storing
      do i = 1, num_out
         sunklabels(i) = unklabels(iout(i))
      enddo

      sp_var1 = real(simtime, kind = single)
      sp_var2 = real(dt, kind = single)

      call write_header_info_sp(num_out, &
                                ncid, &
                                date_string, &
                                flash_release(), &
                                tot_blocks, &
                                sp_var1, &
                                nstep, &
                                nzones_block, &
                                sunklabels, &
                                varid)

      global_offset = n_to_left(MyPE)
!-----------------------------------------------------------------------------
! store the tree information
!-----------------------------------------------------------------------------

! store the refinement level
      starts(1) = global_offset+1
      counts(1) = lnblocks
      if (use_nonblocking_io) then
          err = nfmpi_iput_vara_int(ncid, varid(1), starts, counts, lrefine, reqs(1))
          if (err .NE. NF_NOERR) call check(err, "nfmpi_iput_vara_int: lrefine sp")
      else
          err = nfmpi_put_vara_int_all(ncid, varid(1), starts, counts, lrefine)
          if (err .NE. NF_NOERR) call check(err, "nfmpi_put_vara_int_all: lrefine sp")
      endif

! store the nodetype
      if (use_nonblocking_io) then
          err = nfmpi_iput_vara_int(ncid, varid(2), starts, counts, nodetype, reqs(2))
          if (err .NE. NF_NOERR) call check(err, "nfmpi_iput_vara_int: nodetype sp")
      else
          err = nfmpi_put_vara_int_all(ncid, varid(2), starts, counts, nodetype)
          if (err .NE. NF_NOERR) call check(err, "nfmpi_put_vara_int_all: nodetype sp")
      endif

! store the global id
      starts(1) = 1
      starts(2) = global_offset+1
      counts(1) = NGID
      counts(2) = lnblocks
      if (use_nonblocking_io) then
          err = nfmpi_iput_vara_int(ncid, varid(3), starts, counts, gid, reqs(3))
          if (err .NE. NF_NOERR) call check(err, "nfmpi_iput_vara_int: gid sp")
      else
          err = nfmpi_put_vara_int_all(ncid, varid(3), starts, counts, gid)
          if (err .NE. NF_NOERR) call check(err, "nfmpi_put_vara_int_all: gid sp")
      endif

!-----------------------------------------------------------------------------
! store the grid information
!-----------------------------------------------------------------------------
      
! store the coordinates
      do block_no = 1, lnblocks
         coord_single(:,block_no) = real(coord(:,block_no), kind = single)
      enddo
      starts(1) = 1
      starts(2) = global_offset+1
      counts(1) = NDIM
      counts(2) = lnblocks
      if (use_nonblocking_io) then
          err = nfmpi_iput_vara_real(ncid, varid(4), starts, counts, coord_single, reqs(4))
          if (err .NE. NF_NOERR) call check(err, "nfmpi_iput_vara_real: coord sp")
      else
          err = nfmpi_put_vara_real_all(ncid, varid(4), starts, counts, coord_single)
          if (err .NE. NF_NOERR) call check(err, "nfmpi_put_vara_read_all: coord sp")
      endif

! store the block size
      do block_no = 1, lnblocks
         blk_sz_single(:,block_no) = real(size(:,block_no), kind = single)
      enddo

      starts(1) = 1
      starts(2) = global_offset+1
      counts(1) = NDIM
      counts(2) = lnblocks
      if (use_nonblocking_io) then
          err = nfmpi_iput_vara_real(ncid, varid(5), starts, counts, blk_sz_single, reqs(5))
          if (err .NE. NF_NOERR) call check(err, "nfmpi_iput_vara_real: size sp")
      else
          err = nfmpi_put_vara_real_all(ncid, varid(5), starts, counts, blk_sz_single)
          if (err .NE. NF_NOERR) call check(err, "nfmpi_put_vara_real_all: size sp")
      endif

! store the bounding box

      do block_no = 1, lnblocks
         bnd_single(:,:,block_no) =  & 
              real(bnd_box(:,:,block_no), kind = single)
      enddo

      starts(1) = 1
      starts(2) = 1
      starts(3) = global_offset+1
      counts(1) = 2
      counts(2) = NDIM
      counts(3) = lnblocks
      if (use_nonblocking_io) then
          err = nfmpi_iput_vara_real(ncid, varid(6), starts, counts, bnd_single, reqs(6))
          if (err .NE. NF_NOERR) call check(err, "nfmpi_iput_vara_real: bnd_box")
      else
          err = nfmpi_put_vara_real_all(ncid, varid(6), starts, counts, bnd_single)
          if (err .NE. NF_NOERR) call check(err, "nfmpi_put_vara_real_all: bnd_box")
      endif

      if (use_nonblocking_io) then
          ! calculate attach buffer size for using buffered PnetCDF APIs 
          buf_size = (nxb+1) * (nyb+k2d) * (nzb+k3d) * maxblocks
          buf_size = buf_size + nxb * nyb * nzb * maxblocks
          buf_size = buf_size * num_out * 4
          err = nfmpi_buffer_attach(ncid, buf_size)
      endif

!-----------------------------------------------------------------------------
! store the unknowns -- here we will pass the entire unk array on each 
! processor.  The HDF 5 memory space functionality will pick just the 
! interior cells to write to disk.
!-----------------------------------------------------------------------------

      if (corners) then
         corner_t(2) = MPI_Wtime()
         corner_t(1) = corner_t(2) - corner_t(1)
      else
         nocorner_t(2) = MPI_Wtime()
         nocorner_t(1) = nocorner_t(2) - nocorner_t(1)
      endif

      do ivar = 1, num_out
         record_label = unklabels(iout(ivar))

! put the data at the corners if necessary
         if (corners) then

! interpolate only the variable we are storing to the corners            

! ** Important, the limits of the unkt_crn array do not include 
!    guard cells, so we need to map the interior of the unk array
!    into the unkt_crn array.
            do block_no = 1, lnblocks
            
               do k = nguard*k3d+1,nguard*k3d+nzb+k3d
                  k_store = k - nguard*k3d

                  do j = nguard*k2d+1,nguard*k2d+nyb+k2d
                     j_store = j - nguard*k2d
                     
                     do i = nguard+1,nguard+nxb+1
                        i_store = i - nguard

#if N_DIM == 2
                        unkt_crn(1,i_store,j_store,k_store,block_no) =  & 
                             real( & 
                            .25*(unk(iout(ivar),i-1,j,  k,block_no) + & 
                                 unk(iout(ivar),i  ,j,  k,block_no) + & 
                                 unk(iout(ivar),i  ,j-1,k,block_no) + & 
                                 unk(iout(ivar),i-1,j-1,k,block_no)), & 
                             kind = single)
#endif
#if N_DIM == 3
                        unkt_crn(1,i_store,j_store,k_store,block_no) =  & 
                             real( & 
                            .125*(unk(iout(ivar),i-1,j  ,k  ,block_no) + & 
                                  unk(iout(ivar),i  ,j  ,k  ,block_no) + & 
                                  unk(iout(ivar),i  ,j-1,k  ,block_no) + & 
                                  unk(iout(ivar),i-1,j-1,k  ,block_no) + & 
                                  unk(iout(ivar),i-1,j  ,k-1,block_no) + & 
                                  unk(iout(ivar),i  ,j  ,k-1,block_no) + & 
                                  unk(iout(ivar),i  ,j-1,k-1,block_no) + & 
                                  unk(iout(ivar),i-1,j-1,k-1,block_no)), & 
                             kind = single)
#endif
                        
                     end do
                  end do
               end do
               
            enddo


! we now have the data at the corners, in a 4-byte real array

            starts(1) = 1
            starts(2) = 1
            starts(3) = 1
            starts(4) = global_offset+1
            counts(1) = nxb+1
            counts(2) = nyb+ik2d
            counts(3) = nzb+ik3d
            counts(4) = lnblocks
            if (use_nonblocking_io) then
                err = nfmpi_bput_vara_real(ncid, varid(6+ivar), starts, counts, unkt_crn, reqs(ivar+6))
                if (err .NE. NF_NOERR) call check(err, "nfmpi_bput_vara_real: unknowns sp")
            else
                err = nfmpi_put_vara_real_all(ncid, varid(6+ivar), starts, counts, unkt_crn)
                if (err .NE. NF_NOERR) call check(err, "nfmpi_put_vara_real_all: unknowns sp")
            endif

         else

            unkt(1,:,:,:,:) = real(unk(iout(ivar), & 
                                       nguard+1:nguard+nxb, & 
                                       nguard*k2d+1:nguard*k2d+nyb, & 
                                       nguard*k3d+1:nguard*k3d+nzb,:), & 
                                   kind = single)

            starts(1) = 1
            starts(2) = 1
            starts(3) = 1
            starts(4) = global_offset+1
            counts(1) = nxb
            counts(2) = nyb
            counts(3) = nzb
            counts(4) = lnblocks
            if (use_nonblocking_io) then
                err = nfmpi_bput_vara_real(ncid, varid(6+ivar), starts, counts, unkt, reqs(ivar+6))
                if (err .NE. NF_NOERR) call check(err, "nfmpi_bput_vara_real: unknowns sp")
            else
                err = nfmpi_put_vara_real_all(ncid, varid(6+ivar), starts, counts, unkt)
                if (err .NE. NF_NOERR) call check(err, "nfmpi_put_vara_real_all: unknowns sp")
            endif
         endif

      enddo

      ! wait for all nonblocking requests to complete
      if (use_nonblocking_io) then
          ! wait for the nonblocking I/O to complete
          err = nfmpi_wait_all(ncid, num_out+6, reqs, stats)
          if (err .NE. NF_NOERR) &
              call check(err, "(sp) nfmpi_wait_all: ")

          ! check the status of each nonblocking request
          do i=1, num_out+6
             write(str,'(I2)') i
             if (stats(i) .NE. NF_NOERR) &
                 call check(stats(i), '(sp) nfmpi_wait_all req '//trim(str))
          enddo

          ! detach the temporary buffer
          err = nfmpi_buffer_detach(ncid)
          if (err .NE. NF_NOERR) &
              call check(err, "(sp) nfmpi_buffer_detach: ")
      endif

!-----------------------------------------------------------------------------
! close the file
!-----------------------------------------------------------------------------

      if (corners) then
         corner_t(3) = MPI_Wtime()
         corner_t(2) = corner_t(3) - corner_t(2)
      else
         nocorner_t(3) = MPI_Wtime()
         nocorner_t(2) = nocorner_t(3) - nocorner_t(2)
      endif

      err = nfmpi_inq_put_size(ncid, put_size)
      if (err .NE. NF_NOERR) &
          call check(err, "(sp) nfmpi_inq_put_size: ")

      err = nfmpi_close(ncid)
      if (err .NE. NF_NOERR) call check(err, "nfmpi_close_file sp")

      if (corners) then
         corner_t(3) = MPI_Wtime() - corner_t(3)
      else
         nocorner_t(3) = MPI_Wtime() - nocorner_t(3)
      endif

      plotfile_ncmpi_par = put_size

      return
      end








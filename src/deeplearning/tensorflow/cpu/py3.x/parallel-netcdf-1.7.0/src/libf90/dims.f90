!
!  Copyright (C) 2013, Northwestern University and Argonne National Laboratory
!  See COPYRIGHT notice in top-level directory.
!
! $Id: dims.f90 1468 2013-10-26 16:53:18Z wkliao $
!
! This file is taken from netcdf_dims.f90 with changes for PnetCDF use
!  
!

  !
  ! Dimension routines:
  !
  ! -----------
  function nf90mpi_def_dim(ncid, name, len, dimid)
    integer,                        intent( in) :: ncid
    character (len=*),              intent( in) :: name
    integer (kind=MPI_OFFSET_KIND), intent( in) :: len
    integer,                        intent(out) :: dimid
    integer                                     :: nf90mpi_def_dim
  
    nf90mpi_def_dim = nfmpi_def_dim(ncid, name, len, dimid)
  end function nf90mpi_def_dim
  ! -----------
  function nf90mpi_inq_dimid(ncid, name, dimid)
    integer,             intent( in) :: ncid
    character (len=*),   intent( in) :: name
    integer,             intent(out) :: dimid
    integer                          :: nf90mpi_inq_dimid
    
    nf90mpi_inq_dimid = nfmpi_inq_dimid(ncid, name, dimid)
  end function nf90mpi_inq_dimid
  ! -----------
  function nf90mpi_rename_dim(ncid, dimid, name)
    integer,             intent( in) :: ncid
    character (len=*),   intent( in) :: name
    integer,             intent( in) :: dimid
    integer                          :: nf90mpi_rename_dim

    nf90mpi_rename_dim = nfmpi_rename_dim(ncid, dimid, name)
  end function nf90mpi_rename_dim
  ! -----------
  function nf90mpi_inquire_dimension(ncid, dimid, name, len)
    integer,                                  intent( in) :: ncid, dimid
    character (len=*), optional,              intent(out) :: name
    integer (kind=MPI_OFFSET_KIND), optional, intent(out) :: len
    integer                                               :: nf90mpi_inquire_dimension
    
    character (len=nf90_max_name)  :: dimName
    integer (kind=MPI_OFFSET_KIND) :: length
    
    nf90mpi_inquire_dimension = nfmpi_inq_dim(ncid, dimid, dimName, length)
    if(present(name)) name = trim(dimName)
    if(present(len )) len  = length
  end function nf90mpi_inquire_dimension
  ! -----------

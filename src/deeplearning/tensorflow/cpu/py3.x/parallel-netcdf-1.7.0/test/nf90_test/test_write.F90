!
!  Copyright (C) 2013, Northwestern University and Argonne National Laboratory
!  See COPYRIGHT notice in top-level directory.
!
! $Id: test_write.F90 2011 2015-02-14 19:48:35Z wkliao $
!

! Test nf90mpi_create
!    For mode in NF90_NOCLOBBER, NF90_CLOBBER do:
!       create netcdf file 'scratch.nc' with no data, close it
!       test that it can be opened, do nf90mpi_inq to check nvars = 0, etc.
!    Try again in NF90_NOCLOBBER mode, check error return
! On exit, delete this file
        subroutine test_nf90mpi_create()
        use pnetcdf
        implicit        none
#include "tests.inc"

        integer clobber         !/* 0 for NF90_NOCLOBBER, 1 for NF90_CLOBBER */
        integer err
        integer ncid
        integer ndims           !/* number of dimensions */
        integer nvars           !/* number of variables */
        integer ngatts          !/* number of global attributes */
        integer recdim          !/* id of unlimited dimension */
        integer flags
        integer nok

        flags = IOR(NF90_NOCLOBBER, extra_flags)
        nok = 0
        do 1, clobber = 0, 1
            err = nf90mpi_create(comm, scratch, flags,  info, &
                               ncid)
            if (err .ne. NF90_NOERR) then
                call errore('nf90mpi_create: ', err)
            end if
            nok = nok + 1
            err = nf90mpi_close(ncid)
            if (err .ne. NF90_NOERR) then
                call errore('nf90mpi_close: ', err)
            end if
            err = nf90mpi_open(comm, scratch, NF90_NOWRITE, info,  &
                             ncid)
            if (err .ne. NF90_NOERR) then
                call errore('nf90mpi_open: ', err)
            end if
            err = nf90mpi_inquire(ncid, ndims, nvars, ngatts, recdim)
            if (err .ne. NF90_NOERR) then
                call errore('nf90mpi_inquire: ', err)
            else if (ndims .ne. 0) then
                call errori( &
                    'nf90mpi_inquire: wrong number of dimensions returned, ', &
                    ndims)
            else if (nvars .ne. 0) then
                call errori( &
                    'nf90mpi_inquire: wrong number of variables returned, ', &
                    nvars)
            else if (ngatts .ne. 0) then
                call errori( &
                    'nf90mpi_inquire: wrong number of global atts returned, ', &
                    ngatts)
            else if (recdim .ge. 1) then
                call errori( &
                    'nf90mpi_inquire: wrong record dimension ID returned, ', &
                    recdim)
            end if
            err = nf90mpi_close(ncid)
            if (err .ne. NF90_NOERR) then
                call errore('nf90mpi_close: ', err)
            end if

            flags = IOR(NF90_CLOBBER, extra_flags)
1       continue

        flags = IOR(NF90_NOCLOBBER, extra_flags)
        err = nf90mpi_create(comm, scratch, flags,  info, &
                           ncid)
        if (err .ne. NF90_EEXIST) then
            call errore('attempt to overwrite file: ', err)
        end if
        nok = nok + 1
        err = nf90mpi_delete(scratch, info)
        if (err .ne. NF90_NOERR) then
            call errori('delete of scratch file failed: ', err)
        end if
        call print_nok(nok)
        end


! Test nf90mpi_redef 
! (In fact also tests nf90mpi_enddef - called from test_nf90mpi_enddef)
!    BAD_ID
!    attempt redef (error) & enddef on read-only file
!    create file, define dims & vars. 
!    attempt put var (error)
!    attempt redef (error) & enddef.
!    put vars
!    attempt def new dims (error)
!    redef
!    def new dims, vars.
!    put atts
!    enddef
!    put vars
!    close
!    check file: vars & atts
        subroutine test_nf90mpi_redef()
        use pnetcdf
        implicit        none
#include "tests.inc"
        integer         title_len
        parameter       (title_len = 9)

        integer                 ncid            !/* netcdf id */
        integer                 dimid           !/* dimension id */
        integer                 vid             !/* variable id */
        integer                 err, flags
        character*(title_len)   title
        doubleprecision         var
        character*(NF90_MAX_NAME) name
        integer(kind=MPI_OFFSET_KIND)                 start(1)
        integer(kind=MPI_OFFSET_KIND)                 length
        integer                 intlen
        integer                 dimids(1)
        integer                 nok

        nok = 0

        title = 'Not funny'

        ! BAD_ID tests
        err = nf90mpi_redef(BAD_ID)
        if (err .ne. NF90_EBADID) then
            call errore('bad ncid: ', err)
        endif
        nok = nok + 1
        err = nf90mpi_enddef(BAD_ID)
        if (err .ne. NF90_EBADID) then
            call errore('bad ncid: ', err)
        endif
        nok = nok + 1

        ! read-only tests
        err = nf90mpi_open(comm, testfile, NF90_NOWRITE, info, &
                         ncid)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_open: ', err)
        err = nf90mpi_redef(ncid)
        if (err .ne. NF90_EPERM) then
            call errore('nf90mpi_redef in NF90_NOWRITE mode: ', err)
        endif
        nok = nok + 1
        err = nf90mpi_enddef(ncid)
        if (err .ne. NF90_ENOTINDEFINE) then
            call errore('nf90mpi_redef in NF90_NOWRITE mode: ', err)
        endif
        nok = nok + 1
        err = nf90mpi_close(ncid)
        if (err .ne. NF90_NOERR)  &
            call errore('nf90mpi_close: ', err)

!           /* tests using scratch file */
        flags = IOR(NF90_NOCLOBBER, extra_flags)
        err = nf90mpi_create(comm, scratch, flags, info, &
                           ncid)
        if (err .ne. NF90_NOERR) then
            call errore('nf90mpi_create: ', err)
            return
        end if
        call def_dims(ncid)
        call def_vars(ncid)
        call put_atts(ncid)
        err = nf90mpi_inq_varid(ncid, 'd', vid)
        if (err .ne. NF90_NOERR)  &
            call errore('nf90mpi_inq_varid: ', err)
        var = 1.0
!       should not enter indep mode in define mode
        err = nf90mpi_begin_indep_data(ncid)       
        if (err .ne. NF90_EINDEFINE) &
          call errore('nf90mpi_begin_indep_data... in define mode: ', err)
        start = 0
        err = nf90mpi_put_var(ncid, vid, var, start)
        if (err .ne. NF90_EINDEFINE) &
            call errore('nf90mpi_put_var... in define mode: ', err)
        err = nf90mpi_end_indep_data(ncid)
        if (err .ne. NF90_ENOTINDEP) &
          call errore('nf90mpi_end_indep_data... not in indep mode: ', err)
        err = nf90mpi_redef(ncid)
        if (err .ne. NF90_EINDEFINE) then
            call errore('nf90mpi_redef in define mode: ', err)
        endif
        nok = nok + 1
        err = nf90mpi_enddef(ncid)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_enddef: ', err)
        call put_vars(ncid)
        length = 8
        err = nf90mpi_def_dim(ncid, 'abc', length, dimid)
        if (err .ne. NF90_ENOTINDEFINE) &
            call errore('nf90mpi_def_dim in define mode: ', err)
        err = nf90mpi_redef(ncid)
        if (err .ne. NF90_NOERR) then
            call errore('nf90mpi_redef: ', err)
        endif
        nok = nok + 1
        length = 8
        err = nf90mpi_def_dim(ncid, 'abc', length, dimid)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_def_dim: ', err)
        dimids(1) = 0
        err = nf90mpi_def_var(ncid, 'abc', NF90_INT, varid=vid)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_def_var: ', err)
        length = len(title)
        err = nf90mpi_put_att(ncid, NF90_GLOBAL, 'title', &
                              title)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_put_att: ', err)
        err = nf90mpi_enddef(ncid)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_enddef: ', err)
        var = 1.0
        err = nf90mpi_end_indep_data(ncid)
        if (err .ne. NF90_ENOTINDEP) &
          call errore('nf90mpi_end_indep_data: in collective mode: ',err)
        err = nf90mpi_begin_indep_data(ncid)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_begin_indep_data: ', err)
        err = nf90mpi_put_var(ncid, vid, var, start)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_put_var: ', err)
        err = nf90mpi_end_indep_data(ncid)       
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_end_indep_data: ', err)
        err = nf90mpi_close(ncid)
        if (err .ne. NF90_NOERR)  &
            call errore('nf90mpi_close: ', err)

!           /* check scratch file written as expected */
        call check_file(scratch)
        err = nf90mpi_open(comm, scratch, NF90_NOWRITE, &
              info, ncid)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_open: ', err)
        err = nf90mpi_inquire_dimension(ncid, dimid, name, length)
        if (err .ne. NF90_NOERR)  &
            call errore('nf90mpi_inquire_dimension: ', err)
        if (name .ne. "abc") &
            call errori('Unexpected dim name in netCDF ', ncid)
        if (length .ne. 8) then
            intlen = INT(length)
            call errori('Unexpected dim length: ', intlen)
        end if
        err = nf90mpi_begin_indep_data(ncid)
        err = nf90mpi_get_var(ncid, vid, var, start)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_get_var: ', err)
        if (var .ne. 1.0) &
            call errori( &
                'nf90mpi_get_var: unexpected value in netCDF ' &
                , ncid)
        err = nf90mpi_end_indep_data(ncid)
        err = nf90mpi_close(ncid)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_close: ', err)

        err = nf90mpi_delete(scratch, info)
        if (err .ne. NF90_NOERR) &
            call errori('delete failed for netCDF: ', err)
        call print_nok(nok)
        end

! Test nf90mpi_enddef 
! Simply calls test_nf90mpi_redef which tests both nf90mpi_redef & nf90mpi_enddef

        subroutine test_nf90mpi_enddef()
        use pnetcdf
        implicit        none
#include "tests.inc"

        call test_nf90mpi_redef
        end


! Test nf90mpi_sync
!    try with bad handle, check error
!    try in define mode, check error
!    try writing with one handle, reading with another on same netCDF
        subroutine test_nf90mpi_sync()
        use pnetcdf
        implicit        none
#include "tests.inc"

        integer ncidw         !/* netcdf id for writing */
        integer ncidr         !/* netcdf id for reading */
        integer err, flags
        integer nok

        nok = 0
!           /* BAD_ID test */
        err = nf90mpi_sync(BAD_ID)
        if (err .ne. NF90_EBADID) then
            call errore('bad ncid: ', err)
        else
            nok = nok + 1
        endif

!           /* create scratch file & try nf90mpi_sync in define mode */
        flags = IOR(NF90_NOCLOBBER, extra_flags)
        err = nf90mpi_create(comm, scratch, flags, info, &
                           ncidw)
        if (err .ne. NF90_NOERR) then
            call errore('nf90mpi_create: ', err)
            return
        end if
        err = nf90mpi_sync(ncidw)
        if (err .ne. NF90_EINDEFINE) then
            call errore('nf90mpi_sync called in define mode: ', err)
        else
            nok = nok + 1
        endif

!           /* write using same handle */
        call def_dims(ncidw)
        call def_vars(ncidw)
        call put_atts(ncidw)
        err = nf90mpi_enddef(ncidw)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_enddef: ', err)
        call put_vars(ncidw)
        err = nf90mpi_sync(ncidw)
        if (err .ne. NF90_NOERR) then
            call errore('nf90mpi_sync of ncidw failed: ', err)
        else
            nok = nok + 1
        endif

!           /* open another handle, nf90mpi_sync, read (check) */
        err = nf90mpi_open(comm, scratch, NF90_NOWRITE, info, &
                         ncidr)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_open: ', err)
        err = nf90mpi_sync(ncidr)
        if (err .ne. NF90_NOERR) then
            call errore('nf90mpi_sync of ncidr failed: ', err)
        else
            nok = nok + 1
        endif
        call check_dims(ncidr)
        call check_atts(ncidr)
        call check_vars(ncidr)

!           /* close both handles */
        err = nf90mpi_close(ncidr)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_close: ', err)
        err = nf90mpi_close(ncidw)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_close: ', err)

        err = nf90mpi_delete(scratch, info)
        if (err .ne. NF90_NOERR) &
            call errori('delete of scratch file failed: ', err)
        call print_nok(nok)
        end


! Test nf90mpi_abort
!    try with bad handle, check error
!    try in define mode before anything written, check that file was deleted
!    try after nf90mpi_enddef, nf90mpi_redef, define new dims, vars, atts
!    try after writing variable
        subroutine test_nf90mpi_abort()
        use pnetcdf
        implicit        none
#include "tests.inc"

        integer ncid          !/* netcdf id */
        integer err, flags
        integer ndims
        integer nvars
        integer ngatts
        integer recdim
        integer nok

        nok = 0

!           /* BAD_ID test */
        err = nf90mpi_abort(BAD_ID)
        if (err .ne. NF90_EBADID) then
            call errore('bad ncid: status = ', err)
        else
            nok = nok + 1
        endif

!           /* create scratch file & try nf90mpi_abort in define mode */
        flags = IOR(NF90_NOCLOBBER, extra_flags)
        err = nf90mpi_create(comm, scratch, flags, info, &
                           ncid)
        if (err .ne. NF90_NOERR) then
            call errore('nf90mpi_create: ', err)
            return
        end if
        call def_dims(ncid)
        call def_vars(ncid)
        call put_atts(ncid)
        err = nf90mpi_abort(ncid)
        if (err .ne. NF90_NOERR) then
            call errore('nf90mpi_abort of ncid failed: ', err)
        else
            nok = nok + 1
        endif
        err = nf90mpi_close(ncid)    !/* should already be closed */
        if (err .ne. NF90_EBADID) &
            call errore('bad ncid: ', err)
        err = nf90mpi_delete(scratch, info)    !/* should already be deleted */
        if (err .eq. NF90_NOERR) &
            call errori('scratch file should not exist: ', err)

!            create scratch file
!            do nf90mpi_enddef & nf90mpi_redef
!            define new dims, vars, atts
!            try nf90mpi_abort: should restore previous state (no dims, vars, atts)
        flags = IOR(NF90_NOCLOBBER, extra_flags)
        err = nf90mpi_create(comm, scratch, flags, info, &
                           ncid)
        if (err .ne. NF90_NOERR) then
            call errore('nf90mpi_create: ', err)
            return
        end if
        err = nf90mpi_enddef(ncid)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_enddef: ', err)
        err = nf90mpi_redef(ncid)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_redef: ', err)
        call def_dims(ncid)
        call def_vars(ncid)
        call put_atts(ncid)
        err = nf90mpi_abort(ncid)
        if (err .ne. NF90_NOERR) then
            call errore('nf90mpi_abort of ncid failed: ', err)
        else
            nok = nok + 1
        endif
        err = nf90mpi_close(ncid)    !/* should already be closed */
        if (err .ne. NF90_EBADID) &
            call errore('bad ncid: ', err)
        err = nf90mpi_open(comm, scratch, NF90_NOWRITE, info, &
                         ncid)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_open: ', err)
        err = nf90mpi_inquire(ncid, ndims, nvars, ngatts, recdim)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_inquire: ', err)
        if (ndims .ne. 0) &
            call errori('ndims should be ', 0)
        if (nvars .ne. 0) &
            call errori('nvars should be ', 0)
        if (ngatts .ne. 0) &
            call errori('ngatts should be ', 0)
        err = nf90mpi_close (ncid)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_close: ', err)

!           /* try nf90mpi_abort in data mode - should just close */
        flags = IOR(NF90_CLOBBER, extra_flags)
        err = nf90mpi_create(comm, scratch, flags, info, &
                           ncid)
        if (err .ne. NF90_NOERR) then
            call errore('nf90mpi_create: ', err)
            return
        end if
        call def_dims(ncid)
        call def_vars(ncid)
        call put_atts(ncid)
        err = nf90mpi_enddef(ncid)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_enddef: ', err)
        call put_vars(ncid)
        err = nf90mpi_abort(ncid)
        if (err .ne. NF90_NOERR) then
            call errore('nf90mpi_abort of ncid failed: ', err)
        else
            nok = nok + 1
        endif
        err = nf90mpi_close(ncid)       !/* should already be closed */
        if (err .ne. NF90_EBADID) &
            call errore('bad ncid: ', err)
        call check_file(scratch)
        err = nf90mpi_delete(scratch, info)
        if (err .ne. NF90_NOERR) &
            call errori('delete of scratch file failed: ', err)
        call print_nok(nok)
        end


! Test nf90mpi_def_dim
!    try with bad netCDF handle, check error
!    try in data mode, check error
!    check that returned id is one more than previous id
!    try adding same dimension twice, check error
!    try with illegal sizes, check error
!    make sure unlimited size works, shows up in nf90mpi_inq_unlimdim
!    try to define a second unlimited dimension, check error
        subroutine test_nf90mpi_def_dim()
        use pnetcdf
        implicit        none
#include "tests.inc"

        integer ncid
        integer err             !/* status */
        integer i
        integer dimid         !/* dimension id */
        integer(kind=MPI_OFFSET_KIND) length
        integer nok, flags

        nok = 0

!           /* BAD_ID test */
        length = 8
        err = nf90mpi_def_dim(BAD_ID, 'abc', length, dimid)
        if (err .ne. NF90_EBADID) then
            call errore('bad ncid: ', err)
        else
            nok = nok + 1
        endif

!           /* data mode test */
        flags = IOR(NF90_CLOBBER, extra_flags)
        err = nf90mpi_create(comm, scratch, flags, info, &
                           ncid)
        if (err .ne. NF90_NOERR) then
            call errore('nf90mpi_create: ', err)
            return
        end if
        err = nf90mpi_enddef(ncid)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_enddef: ', err)
        length = 8
        err = nf90mpi_def_dim(ncid, 'abc', length, dimid)
        if (err .ne. NF90_ENOTINDEFINE) then
            call errore('bad ncid: ', err)
        else
            nok = nok + 1
        endif

!           /* define-mode tests: unlimited dim */
        err = nf90mpi_redef(ncid)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_redef: ', err)
        err = nf90mpi_def_dim(ncid, dim_name(1), NF90MPI_UNLIMITED, dimid)
        if (err .ne. NF90_NOERR)  then
            call errore('nf90mpi_def_dim: ', err)
        else
            nok = nok + 1
        endif
        if (dimid .ne. 1)  &
            call errori('Unexpected dimid: ', dimid)
        err = nf90mpi_inquire(ncid, unlimitedDimId=dimid)
        if (err .ne. NF90_NOERR)  &
            call errore('nf90mpi_inquire: ', err)
        if (dimid .ne. RECDIM)  &
            call error('Unexpected recdim: ')
        err = nf90mpi_inquire_dimension(ncid, dimid, len=length)
        if (length .ne. 0)  &
            call errori('Unexpected length: ', 0)
        err = nf90mpi_def_dim(ncid, 'abc', NF90MPI_UNLIMITED, dimid)
        if (err .ne. NF90_EUNLIMIT) then
            call errore('2nd unlimited dimension: ', err)
        else
            nok = nok + 1
        endif

!           /* define-mode tests: remaining dims */
        do 1, i = 2, NDIMS
            err = nf90mpi_def_dim(ncid, dim_name(i-1), dim_len(i),  &
                             dimid)
            if (err .ne. NF90_ENAMEINUSE) then
                call errore('duplicate name: ', err)
            else
                nok = nok + 1
            endif
            err = nf90mpi_def_dim(ncid, BAD_NAME, dim_len(i), dimid)
            if (err .ne. NF90_EBADNAME) then
                call errore('bad name: ', err)
            else
                nok = nok + 1
            endif
            length = NF90MPI_UNLIMITED - 1
            err = nf90mpi_def_dim(ncid, dim_name(i), length, &
                             dimid)
            if (err .ne. NF90_EDIMSIZE) then
                call errore('bad size: ', err)
            else
                nok = nok + 1
            endif
            err = nf90mpi_def_dim(ncid, dim_name(i), dim_len(i), dimid)
            if (err .ne. NF90_NOERR)  then
                call errore('nf90mpi_def_dim: ', err)
            else
                nok = nok + 1
            endif
            if (dimid .ne. i)  &
                call errori('Unexpected dimid: ', 0)
1       continue

!           /* Following just to expand unlimited dim */
        call def_vars(ncid)
        err = nf90mpi_enddef(ncid)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_enddef: ', err)
        call put_vars(ncid)

!           /* Check all dims */
        call check_dims(ncid)

        err = nf90mpi_close(ncid)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_close: ', err)
        err = nf90mpi_delete(scratch, info)
        if (err .ne. NF90_NOERR) &
            call errori('delete of scratch file failed: ', err)
        call print_nok(nok)
        end


! Test nf90mpi_rename_dim
!    try with bad netCDF handle, check error
!    check that proper rename worked with nf90mpi_inquire_dimension
!    try renaming to existing dimension name, check error
!    try with bad dimension handle, check error
        subroutine test_nf90mpi_rename_dim()
        use pnetcdf
        implicit        none
#include "tests.inc"

        integer ncid
        integer err             !/* status */
        character*(NF90_MAX_NAME) name
        integer nok, flags

        nok = 0

!           /* BAD_ID test */
        err = nf90mpi_rename_dim(BAD_ID, 1, 'abc')
        if (err .ne. NF90_EBADID) then
            call errore('bad ncid: ', err)
        else
            nok = nok + 1
        endif

!           /* main tests */
        flags = IOR(NF90_NOCLOBBER, extra_flags)
        err = nf90mpi_create(comm, scratch, flags, info, &
                           ncid)
        if (err .ne. NF90_NOERR) then
            call errore('nf90mpi_create: ', err)
            return
        end if
        call def_dims(ncid)
        err = nf90mpi_rename_dim(ncid, BAD_DIMID, 'abc')
        if (err .ne. NF90_EBADDIM) then
            call errore('bad dimid: ', err)
        else
            nok = nok + 1
        endif
        err = nf90mpi_rename_dim(ncid, 3, 'abc')
        if (err .ne. NF90_NOERR) then
            call errore('nf90mpi_rename_dim: ', err)
        else
            nok = nok + 1
        endif
        err = nf90mpi_inquire_dimension(ncid, 3, name)
        if (name .ne. 'abc') &
            call errorc('Unexpected name: ', name)
        err = nf90mpi_rename_dim(ncid, 1, 'abc')
        if (err .ne. NF90_ENAMEINUSE) then
            call errore('duplicate name: ', err)
        else
            nok = nok + 1
        endif

        err = nf90mpi_close(ncid)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_close: ', err)
        err = nf90mpi_delete(scratch, info)
        if (err .ne. NF90_NOERR) &
            call errori('delete of scratch file failed: ', err)
        call print_nok(nok)
        end


! Test nf90mpi_def_var
!    try with bad netCDF handle, check error
!    try with bad name, check error
!    scalar tests:
!      check that proper define worked with nf90mpi_inq_var
!      try redefining an existing variable, check error
!      try with bad datatype, check error
!      try with bad number of dimensions, check error
!      try in data mode, check error
!    check that returned id is one more than previous id
!    try with bad dimension ids, check error
        subroutine test_nf90mpi_def_var()
        use pnetcdf
        implicit        none
#include "tests.inc"

        integer ncid
        integer vid
        integer err             !/* status */
        integer i
        integer ndims
        integer na
        character*(NF90_MAX_NAME) name
        integer dimids(MAX_RANK)
        integer datatype
        integer nok, flags

        nok = 0

!           /* BAD_ID test */
        err = nf90mpi_def_var(BAD_ID, 'abc', NF90_SHORT, varid=vid)
        if (err .ne. NF90_EBADID) then
            call errore('bad ncid: status = ', err)
        else
            nok = nok + 1
        endif

!       scalar tests
        flags = IOR(NF90_NOCLOBBER, extra_flags)
        err = nf90mpi_create(comm, scratch, flags, info, &
                           ncid)
        if (err .ne. NF90_NOERR) then
            call errore('nf90mpi_create: ', err)
            return
        end if
        err = nf90mpi_def_var(ncid, 'abc', NF90_SHORT, varid=vid)
        if (err .ne. NF90_NOERR) then
            call errore('nf90mpi_def_var: ', err)
        else
            nok = nok + 1
        endif
        err = nf90mpi_inquire_variable(ncid, vid, name, datatype, ndims, dimids,  &
                         na)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_inquire_variable: ', err)
        if (name .ne. 'abc') &
            call errorc('Unexpected name: ', name)
        if (datatype .ne. NF90_SHORT) &
            call error('Unexpected datatype')
        if (ndims .ne. 0) &
            call error('Unexpected rank')
        err = nf90mpi_def_var(ncid, BAD_NAME, NF90_SHORT, varid=vid)
        if (err .ne. NF90_EBADNAME) then
            call errore('bad name: ', err)
        else
            nok = nok + 1
        endif
        err = nf90mpi_def_var(ncid, 'abc', NF90_SHORT, varid=vid)
        if (err .ne. NF90_ENAMEINUSE) then
            call errore('duplicate name: ', err)
        else
            nok = nok + 1
        endif
        err = nf90mpi_def_var(ncid, 'ABC', BAD_TYPE, varid=vid)
        if (err .ne. NF90_EBADTYPE) then
            call errore('bad type: ', err)
        else
            nok = nok + 1
        endif
        err = nf90mpi_enddef(ncid)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_enddef: ', err)
        err = nf90mpi_def_var(ncid, 'ABC', NF90_SHORT, varid=vid)
        if (err .ne. NF90_ENOTINDEFINE) then
            call errore('nf90mpi_def_var called in data mode: ', err)
        else
            nok = nok + 1
        endif
        err = nf90mpi_close(ncid)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_close: ', err)
        err = nf90mpi_delete(scratch, info)
        if (err .ne. NF90_NOERR) &
            call errorc('delete of scratch file failed: ', scratch)

!           /* general tests using global vars */
        flags = IOR(NF90_CLOBBER, extra_flags)
        err = nf90mpi_create(comm, scratch, flags, info, &
                           ncid)
        if (err .ne. NF90_NOERR) then
            call errore('nf90mpi_create: ', err)
            return
        end if
        call def_dims(ncid)
        do 1, i = 1, numVars
            err = nf90mpi_def_var(ncid, var_name(i), var_type(i),  &
                             var_dimid(1:var_rank(i),i), vid)
            if (err .ne. NF90_NOERR)  then
                call errore('nf90mpi_def_var: ', err)
            else
                nok = nok + 1
            endif
            if (vid .ne. i) &
                call error('Unexpected varid')
1       continue

!           /* try bad dim ids */
        dimids(1) = BAD_DIMID
        err = nf90mpi_def_var(ncid, 'abc', NF90_SHORT, dimids(1:1), vid)
        if (err .ne. NF90_EBADDIM) then
            call errore('bad dim ids: ', err)
        else
            nok = nok + 1
        endif
        err = nf90mpi_close(ncid)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_close: ', err)

        err = nf90mpi_delete(scratch, info)
        if (err .ne. NF90_NOERR) &
            call errorc('delete of scratch file failed: ', scratch)
        call print_nok(nok)
        end


! Test nf90mpi_rename_var
!    try with bad netCDF handle, check error
!    try with bad variable handle, check error
!    try renaming to existing variable name, check error
!    check that proper rename worked with nf90mpi_inq_varid
!    try in data mode, check error
        subroutine test_nf90mpi_rename_var()
        use pnetcdf
        implicit        none
#include "tests.inc"

        integer ncid
        integer vid
        integer err
        integer i
        character*(NF90_MAX_NAME) name
        integer nok, flags

        nok = 0

        flags = IOR(NF90_NOCLOBBER, extra_flags)
        err = nf90mpi_create(comm, scratch, flags, info, &
                           ncid)
        if (err .ne. NF90_NOERR) then
            call errore('nf90mpi_create: ', err)
            return
        end if
        err = nf90mpi_rename_var(ncid, BAD_VARID, 'newName')
        if (err .ne. NF90_ENOTVAR) then
            call errore('bad var id: ', err)
        else
            nok = nok + 1
        endif
        call def_dims(ncid)
        call def_vars(ncid)

!           /* Prefix "new_" to each name */
        do 1, i = 1, numVars
            err = nf90mpi_rename_var(BAD_ID, i, 'newName')
            if (err .ne. NF90_EBADID) then
                call errore('bad ncid: ', err)
            else
                nok = nok + 1
            endif
            err = nf90mpi_rename_var(ncid, i, var_name(numVars))
            if (err .ne. NF90_ENAMEINUSE) then
                call errore('duplicate name: ', err)
            else
                nok = nok + 1
            endif
            name = 'new_' // var_name(i)
            err = nf90mpi_rename_var(ncid, i, name)
            if (err .ne. NF90_NOERR) then
                call errore('nf90mpi_rename_var: ', err)
            else
                nok = nok + 1
            endif
            err = nf90mpi_inq_varid(ncid, name, vid)
            if (err .ne. NF90_NOERR) &
                call errore('nf90mpi_inq_varid: ', err)
            if (vid .ne. i) &
                call error('Unexpected varid')
1       continue

!           /* Change to data mode */
!           /* Try making names even longer. Then restore original names */
        err = nf90mpi_enddef(ncid)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_enddef: ', err)
        do 2, i = 1, numVars
            name = 'even_longer_' // var_name(i)
            err = nf90mpi_rename_var(ncid, i, name)
            if (err .ne. NF90_ENOTINDEFINE) then
                call errore('longer name in data mode: ', err)
            else
                nok = nok + 1
            endif
            err = nf90mpi_rename_var(ncid, i, var_name(i))
            if (err .ne. NF90_NOERR) then
                call errore('nf90mpi_rename_var: ', err)
            else
                nok = nok + 1
            endif
            err = nf90mpi_inq_varid(ncid, var_name(i), vid)
            if (err .ne. NF90_NOERR) &
                call errore('nf90mpi_inq_varid: ', err)
            if (vid .ne. i) &
                call error('Unexpected varid')
2       continue

        call put_vars(ncid)
        call check_vars(ncid)

        err = nf90mpi_close(ncid)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_close: ', err)

        err = nf90mpi_delete(scratch, info)
        if (err .ne. NF90_NOERR) &
            call errorc('delete of scratch file failed: ', scratch)
        call print_nok(nok)
        end


! Test nf90mpi_copy_att
!    try with bad source or target netCDF handles, check error
!    try with bad source or target variable handle, check error
!    try with nonexisting attribute, check error
!    check that NF90_GLOBAL variable for source or target works
!    check that new attribute put works with target in define mode
!    check that old attribute put works with target in data mode
!    check that changing type and length of an attribute work OK
!    try with same ncid for source and target, different variables
!    try with same ncid for source and target, same variable
        subroutine test_nf90mpi_copy_att()
        use pnetcdf
        implicit        none
#include "tests.inc"
        character*2 ATT_NAME
        integer VARID, NATTS, ATT_LEN

        integer ncid_in
        integer ncid_out
        integer vid
        integer err
        integer i
        integer j
        character*(NF90_MAX_NAME) name    !/* of att */
        integer datatype                !/* of att */
        integer(kind=MPI_OFFSET_KIND) length                  !/* of att */
        character*1     value
        integer nok, flags

        nok = 0
        err = nf90mpi_open(comm, testfile, NF90_NOWRITE, info, &
                         ncid_in)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_open: ', err)
        flags = IOR(NF90_NOCLOBBER, extra_flags)
        err = nf90mpi_create(comm, scratch, flags, info, &
                           ncid_out)
        if (err .ne. NF90_NOERR) then
            call errore('nf90mpi_create: ', err)
            return
        end if
        call def_dims(ncid_out)
        call def_vars(ncid_out)

        do 1, i = 0, numVars
            vid = VARID(i)
            do 2, j = 1, NATTS(i)
                name = ATT_NAME(j,i)
                err = nf90mpi_copy_att(ncid_in, BAD_VARID, name, &
                                      ncid_out, vid)
                if (err .ne. NF90_ENOTVAR) &
                    call errore('bad var id: ', err)
                nok = nok + 1
                err = nf90mpi_copy_att(ncid_in, vid, name, ncid_out,  &
                                  BAD_VARID)
                if (err .ne. NF90_ENOTVAR) &
                    call errore('bad var id: ', err)
                nok = nok + 1
                err = nf90mpi_copy_att(BAD_ID, vid, name,  &
                      ncid_out, vid)
                if (err .ne. NF90_EBADID) &
                    call errore('bad ncid: ', err)
                nok = nok + 1
                err = nf90mpi_copy_att(ncid_in, vid, name,  &
                      BAD_ID, vid)
                if (err .ne. NF90_EBADID) &
                    call errore('bad ncid: ', err)
                nok = nok + 1
                err = nf90mpi_copy_att(ncid_in, vid, 'noSuch', &
                                      ncid_out, vid)
                if (err .ne. NF90_ENOTATT) &
                    call errore('bad attname: ', err)
                nok = nok + 1
                err = nf90mpi_copy_att(ncid_in, vid, name,  &
                       ncid_out, vid)
                if (err .ne. NF90_NOERR) &
                    call errore('nf90mpi_copy_att: ', err)
                nok = nok + 1
                err = nf90mpi_copy_att(ncid_out, vid, name, &
                                      ncid_out, vid)
                if (err .ne. NF90_NOERR) &
                    call errore('source = target: ', err)
                nok = nok + 1
2           continue
1       continue

        err = nf90mpi_close(ncid_in)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_close: ', err)

!           /* Close scratch. Reopen & check attributes */
        err = nf90mpi_close(ncid_out)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_close: ', err)
        err = nf90mpi_open(comm, scratch, NF90_WRITE, info, &
                         ncid_out)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_open: ', err)
        call check_atts(ncid_out)

!           change to define mode
!           define single char. global att. ':a' with value 'A'
!           This will be used as source for following copies
        err = nf90mpi_redef(ncid_out)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_redef: ', err)
        length = 1
        err = nf90mpi_put_att(ncid_out, NF90_GLOBAL, 'a', 'A')
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_put_att: ', err)

!           change to data mode
!           Use scratch as both source & dest.
!           try copy to existing att. change type & decrease length
!           rename 1st existing att of each var (if any) 'a'
!           if this att. exists them copy ':a' to it
        err = nf90mpi_enddef(ncid_out)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_enddef: ', err)
        do 3, i = 1, numVars
            if (NATTS(i) .gt. 0 .and. ATT_LEN(1,i) .gt. 0) then
                err = nf90mpi_rename_att(ncid_out, i,  &
                      att_name(1,i), 'a')
                if (err .ne. NF90_NOERR) &
                    call errore('nf90mpi_rename_att: ', err)
                err = nf90mpi_copy_att(ncid_out, NF90_GLOBAL, 'a', &
                                      ncid_out, i)
                if (err .ne. NF90_NOERR) &
                    call errore('nf90mpi_copy_att: ', err)
                nok = nok + 1
            end if
3       continue
        err = nf90mpi_close(ncid_out)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_close: ', err)

!           /* Reopen & check */
        err = nf90mpi_open(comm, scratch, NF90_WRITE, info, &
                         ncid_out)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_open: ', err)
        do 4, i = 1, numVars
            if (NATTS(i) .gt. 0 .and. ATT_LEN(1,i) .gt. 0) then
                err = nf90mpi_inquire_attribute(ncid_out, i, 'a',  &
                      datatype, length)
                if (err .ne. NF90_NOERR) &
                    call errore('nf90mpi_inquire_attribute: ', err)
                if (datatype .ne. NF90_CHAR) &
                    call error('Unexpected type')
                if (length .ne. 1) &
                    call error('Unexpected length')
                err = nf90mpi_get_att(ncid_out, i, 'a', value)
                if (err .ne. NF90_NOERR) &
                    call errore('nf90mpi_get_att: ', err)
                if (value .ne. 'A') &
                    call error('Unexpected value')
            end if                                                   
4       continue                                                   

        err = nf90mpi_close(ncid_out)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_close: ', err)
        err = nf90mpi_delete(scratch, info)
        if (err .ne. NF90_NOERR) &
            call errorc('delete of scratch file failed', scratch)
        call print_nok(nok)
        end


! Test nf90mpi_rename_att
!    try with bad netCDF handle, check error
!    try with bad variable handle, check error
!    try with nonexisting att name, check error
!    try renaming to existing att name, check error
!    check that proper rename worked with nf90mpi_inq_attid
!    try in data mode, check error
        subroutine test_nf90mpi_rename_att()
        use pnetcdf
        implicit        none
#include "tests.inc"
        character*2 ATT_NAME
        integer VARID, ATT_TYPE, NATTS, ATT_LEN
        double precision hash
        logical equal, inrange

        integer ncid
        integer vid
        integer err, flags
        integer i
        integer j
        integer  k
        integer attnum
        character*(NF90_MAX_NAME) atnam
        character*(NF90_MAX_NAME) name
        character*(NF90_MAX_NAME) oldname
        character*(NF90_MAX_NAME) newname
        integer nok             !/* count of valid comparisons */
        integer datatype
        integer attyp
        integer(kind=MPI_OFFSET_KIND) length
        integer(kind=MPI_OFFSET_KIND) attlength
        integer(kind=MPI_OFFSET_KIND) ndx(1)
        character*(MAX_NELS)    text
        doubleprecision value(MAX_NELS)
        doubleprecision expect

        nok = 0

        flags = IOR(NF90_NOCLOBBER, extra_flags)
        err = nf90mpi_create(comm, scratch, flags, info, &
                           ncid)
        if (err .ne. NF90_NOERR) then
            call errore('nf90mpi_create: ', err)
            return
        end if
        err = nf90mpi_rename_att(ncid, BAD_VARID, 'abc', 'newName')
        if (err .ne. NF90_ENOTVAR) &
            call errore('bad var id: ', err)
        call def_dims(ncid)
        call def_vars(ncid)
        call put_atts(ncid)

        do 1, i = 0, numVars
            vid = VARID(i)
            do 2, j = 1, NATTS(i)
                atnam = ATT_NAME(j,i)
                err = nf90mpi_rename_att(BAD_ID, vid, atnam,  &
                       'newName')
                if (err .ne. NF90_EBADID) &
                    call errore('bad ncid: ', err)
                err = nf90mpi_rename_att(ncid, vid, 'noSuch',  &
                        'newName')
                if (err .ne. NF90_ENOTATT) &
                    call errore('bad attname: ', err)
                newname = 'new_' // trim(atnam)
                err = nf90mpi_rename_att(ncid, vid, atnam, newname)
                if (err .ne. NF90_NOERR) &
                    call errore('nf90mpi_rename_att: ', err)
                err = nf90mpi_inquire_attribute(ncid, vid, newname, attnum=attnum)
                if (err .ne. NF90_NOERR) &
                    call errore('nf90mpi_inquire_attribute: ', err)
                if (attnum .ne. j) &
                    call error('Unexpected attnum')
2           continue
1       continue

!           /* Close. Reopen & check */
        err = nf90mpi_close(ncid)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_close: ', err)
        err = nf90mpi_open(comm, scratch, NF90_WRITE, info,  &
            ncid)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_open: ', err)

        do 3, i = 0, numVars
            vid = VARID(i)
            do 4, j = 1, NATTS(i)
                atnam = ATT_NAME(j,i)
                attyp = ATT_TYPE(j,i)
                attlength = ATT_LEN(j,i)
                newname = 'new_' // trim(atnam)
                err = nf90mpi_inq_attname(ncid, vid, j, name)
                if (err .ne. NF90_NOERR) &
                    call errore('nf90mpi_inq_attname: ', err)
                if (name .ne. newname) &
                    call error('nf90mpi_inq_attname: unexpected name')
                err = nf90mpi_inquire_attribute(ncid, vid, name,  &
                    datatype, length)
                if (err .ne. NF90_NOERR) &
                    call errore('nf90mpi_inquire_attribute: ', err)
                if (datatype .ne. attyp) &
                    call error('nf90mpi_inquire_attribute: unexpected type')
                if (length .ne. attlength) &
                    call error('nf90mpi_inquire_attribute: unexpected length')
                if (datatype .eq. NF90_CHAR) then
                    err = nf90mpi_get_att(ncid, vid, name, text)
                    if (err .ne. NF90_NOERR) &
                        call errore('nf90mpi_get_att: ', err)
                    do 5, k = 1, INT(attlength)
                        ndx(1) = k
                        expect = hash(datatype, -1, ndx)
                        if (ichar(text(k:k)) .ne. expect) then
                            call error( &
                                'nf90mpi_get_att: unexpected value')
                        else
                            nok = nok + 1
                        end if
5                   continue
                else
                    err = nf90mpi_get_att(ncid, vid, name,  &
                           value)
                    if (err .ne. NF90_NOERR) &
                        call errore('nf90mpi_get_att: ', err)
                    do 6, k = 1, INT(attlength)
                        ndx(1) = k
                        expect = hash(datatype, -1, ndx)
                        if (inRange(expect, datatype)) then
                            if (.not. equal(value(k),expect,datatype, &
                                            NF90_DOUBLE)) then
                                call error( &
                              'nf90mpi_get_att: unexpected value')
                            else
                                nok = nok + 1
                            end if
                        end if
6                   continue
                end if
4           continue
3       continue
        call print_nok(nok)

!           /* Now in data mode */
!           /* Try making names even longer. Then restore original names */

        do 7, i = 0, numVars
            vid = VARID(i)
            do 8, j = 1, NATTS(i)
                atnam = ATT_NAME(j,i)
                oldname = 'new_' // trim(atnam)
                newname = 'even_longer_' // trim(atnam)
                err = nf90mpi_rename_att(ncid, vid, oldname, newname)
                if (err .ne. NF90_ENOTINDEFINE) &
                    call errore('longer name in data mode: ', err)
                err = nf90mpi_rename_att(ncid, vid, oldname, atnam)
                if (err .ne. NF90_NOERR) &
                    call errore('nf90mpi_rename_att: ', err)
                err = nf90mpi_inquire_attribute(ncid, vid, atnam, attnum=attnum)
                if (err .ne. NF90_NOERR) &
                    call errore('nf90mpi_inquire_attribute: ', err)
                if (attnum .ne. j) &
                    call error('Unexpected attnum')
8           continue
7       continue

        err = nf90mpi_close(ncid)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_close: ', err)

        err = nf90mpi_delete(scratch, info)
        if (err .ne. NF90_NOERR) &
            call errori('delete of scratch file failed: ', err)
        end


! Test nf90mpi_del_att
!    try with bad netCDF handle, check error
!    try with bad variable handle, check error
!    try with nonexisting att name, check error
!    check that proper delete worked using:
!      nf90mpi_inq_attid, nf90mpi_inq_natts, nf90mpi_inq_varnatts
        subroutine test_nf90mpi_del_att()
        use pnetcdf
        implicit        none
#include "tests.inc"
        character*2 ATT_NAME
        integer VARID, NATTS

        integer ncid
        integer err, flags
        integer i
        integer j
        integer attnum
        integer na
        integer numatts
        integer vid
        character*(NF90_MAX_NAME)  name           !/* of att */
        integer nok             !/* count of valid comparisons */

        nok = 0

        flags = IOR(NF90_NOCLOBBER, extra_flags)
        err = nf90mpi_create(comm, scratch, flags, info, &
                           ncid)
        if (err .ne. NF90_NOERR) then
            call errore('nf90mpi_create: ', err)
            return
        end if
        err = nf90mpi_del_att(ncid, BAD_VARID, 'abc')
        if (err .ne. NF90_ENOTVAR) then
            call errore('bad var id: ', err)
        else
            nok = nok + 1
        endif
        call def_dims(ncid)
        call def_vars(ncid)
        call put_atts(ncid)

        do 1, i = 0, numVars
            vid = VARID(i)
            numatts = NATTS(i)
            do 2, j = 1, numatts
                name = ATT_NAME(j,i)
                err = nf90mpi_del_att(BAD_ID, vid, name)
                if (err .ne. NF90_EBADID) then
                    call errore('bad ncid: ', err)
                else
                    nok = nok + 1
                endif
                err = nf90mpi_del_att(ncid, vid, 'noSuch')
                if (err .ne. NF90_ENOTATT) then
                    call errore('bad attname: ', err)
                else
                    nok = nok + 1
                endif
                err = nf90mpi_del_att(ncid, vid, name)
                if (err .ne. NF90_NOERR) then
                    call errore('nf90mpi_del_att: ', err)
                else
                    nok = nok + 1
                endif
                err = nf90mpi_inquire_attribute(ncid, vid, name, attnum=attnum)
                if (err .ne. NF90_ENOTATT) &
                    call errore('bad attname: ', err)
                if (i .lt. 1) then
                    err = nf90mpi_inquire(ncid, nAttributes=na)
                    if (err .ne. NF90_NOERR) &
                        call errore('nf90mpi_inquire: ', err)
                    if (na .ne. numatts-j) then
                        call errori('natts: expected: ', numatts-j)
                        call errori('natts: got:      ', na)
                    endif
                else
                    err = nf90mpi_inquire_variable(ncid, vid, nAtts=na)
                    if (err .ne. NF90_NOERR) &
                        call errore('nf90mpi_inquire_variable: ', err)
                    if (na .ne. numatts-j) then
                        call errori('natts: expected: ', numatts-j)
                        call errori('natts: got:      ', na)
                    endif
                endif
2           continue
1       continue

!           /* Close. Reopen & check no attributes left */
        err = nf90mpi_close(ncid)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_close: ', err)
        err = nf90mpi_open(comm, scratch, NF90_WRITE,  &
           info, ncid)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_open: ', err)
        err = nf90mpi_inquire(ncid, nAttributes=na)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_inquire: ', err)
        if (na .ne. 0) &
            call errori('natts: expected 0, got ', na)
        do 3, i = 0, numVars
            vid = VARID(i)
            err = nf90mpi_inquire(ncid, nAttributes=na)
            if (err .ne. NF90_NOERR) &
                call errore('nf90mpi_inquire: ', err)
            if (na .ne. 0) &
                call errori('natts: expected 0, got ', na)
3       continue

!           /* restore attributes. change to data mode. try to delete */
        err = nf90mpi_redef(ncid)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_redef: ', err)
        call put_atts(ncid)
        err = nf90mpi_enddef(ncid)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_enddef: ', err)

        do 4, i = 0, numVars
            vid = VARID(i)
            numatts = NATTS(i)
            do 5, j = 1, numatts
                name = ATT_NAME(j,i)
                err = nf90mpi_del_att(ncid, vid, name)
                if (err .ne. NF90_ENOTINDEFINE) then
                    call errore('in data mode: ', err)
                else
                    nok = nok + 1
                endif
5           continue
4       continue

        err = nf90mpi_close(ncid)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_close: ', err)
        err = nf90mpi_delete(scratch, info)
        if (err .ne. NF90_NOERR) &
            call errori('delete of scratch file failed: ', err)
        call print_nok(nok)
        end

! parallel-netcdf doesn't implement set_fill, so i have not
!    parallel-netcdfified this subroutine
! Test nf90mpi_set_fill
!    try with bad netCDF handle, check error
!    try in read-only mode, check error
!    try with bad new_fillmode, check error
!    try in data mode, check error
!    check that proper set to NF90_FILL works for record & non-record variables
!    (note that it is not possible to test NF90_NOFILL mode!)
!    close file & create again for test using attribute _FillValue
        subroutine test_nf90mpi_set_fill()
        use pnetcdf
        implicit none
#include "tests.inc"
        ! character*2 ATT_NAME
        ! integer VARID, ATT_TYPE, NATTS

        integer ncid
        integer vid
        integer err, flags
        integer i
        integer j
        integer old_fillmode
        character*1 text
        doubleprecision value
        doubleprecision fill, fills(1)
        integer(kind=MPI_OFFSET_KIND) index(MAX_RANK)
        integer(kind=MPI_OFFSET_KIND) length
        integer index2indexes
        integer nok             !/* count of valid comparisons */

        value = 0
        nok = 0

!           /* bad ncid */
        err = nf90mpi_set_fill(BAD_ID, NF90_NOFILL, old_fillmode)
        if (err .ne. NF90_EBADID) &
            call errore('bad ncid: ', err)

!           /* try in read-only mode */
        err = nf90mpi_open(comm, testfile, NF90_NOWRITE, info, &
                         ncid)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_open: ', err)
        err = nf90mpi_set_fill(ncid, NF90_NOFILL, old_fillmode)
        if (err .ne. NF90_EPERM) &
            call errore('read-only: ', err)
        err = nf90mpi_close(ncid)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_close: ', err)

!           /* create scratch */
        flags = IOR(NF90_NOCLOBBER, extra_flags)
        err = nf90mpi_create(comm, scratch, flags, info, &
                           ncid)
        if (err .ne. NF90_NOERR) then
            call errore('nf90mpi_create: ', err)
            return
        end if

!           /* BAD_FILLMODE */
        err = nf90mpi_set_fill(ncid, BAD_FILLMODE, old_fillmode)
        if (err .ne. NF90_EINVAL) &
            call errore('bad fillmode: ', err)

!           /* proper calls */
        err = nf90mpi_set_fill(ncid, NF90_FILL, old_fillmode)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_set_fill: ', err)
        if (old_fillmode .ne. NF90_NOFILL) &
            call errori('Unexpected old fill mode: ', old_fillmode)
        err = nf90mpi_set_fill(ncid, NF90_NOFILL, old_fillmode)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_set_fill: ', err)
        if (old_fillmode .ne. NF90_FILL) &
            call errori('Unexpected old fill mode: ', old_fillmode)
        err = nf90mpi_set_fill(ncid, NF90_FILL, old_fillmode)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_set_fill: ', err)

!           /* define dims & vars */
        call def_dims(ncid)
        call def_vars(ncid)

!           /* Change to data mode. Set fillmode again */
        err = nf90mpi_enddef(ncid)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_enddef: ', err)
        err = nf90mpi_set_fill(ncid, NF90_FILL, old_fillmode)
        if (err .ne. NF90_ENOTINDEFINE) &
            call errore('nf90mpi_set_fill: ', err)

!       /* Write record number NRECS to force writing of preceding records */
!       /* Assumes variable cr is char vector with UNLIMITED dimension */
        err = nf90mpi_inq_varid(ncid, 'cr', vid)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_inq_varid: ', err)
        index(1) = NRECS
        text = char(NF90_FILL_CHAR)
        err = nf90mpi_put_var_all(ncid, vid, text, index)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_put_var_all: ', err)

!           /* get all variables & check all values equal default fill */
        do 1, i = 1, numVars
            if (var_dimid(var_rank(i),i) .eq. RECDIM) cycle ! skip record variables

            if (var_type(i) .eq. NF90_CHAR) then
                fill = NF90_FILL_CHAR
            else if (var_type(i) .eq. NF90_BYTE) then
                fill = NF90_FILL_BYTE
            else if (var_type(i) .eq. NF90_SHORT) then
                fill = NF90_FILL_SHORT
            else if (var_type(i) .eq. NF90_INT) then
                fill = NF90_FILL_INT
            else if (var_type(i) .eq. NF90_FLOAT) then
                fill = NF90_FILL_FLOAT
            else if (var_type(i) .eq. NF90_DOUBLE) then
                fill = NF90_FILL_DOUBLE
            else if (var_type(i) .eq. NF90_UBYTE) then
                fill = NF90_FILL_UBYTE
            else if (var_type(i) .eq. NF90_USHORT) then
                fill = NF90_FILL_USHORT
            else if (var_type(i) .eq. NF90_UINT) then
                fill = NF90_FILL_UINT
            else if (var_type(i) .eq. NF90_INT64) then
                fill = NF90_FILL_INT64
            else if (var_type(i) .eq. NF90_UINT64) then
                fill = NF90_FILL_UINT64
            else
                stop 'test_nf90mpi_set_fill(): impossible var_type(i)'
            end if

            do 2, j = 1, var_nels(i)
                err = index2indexes(j, var_rank(i), var_shape(1,i),  &
                                    index)
                if (err .ne. NF90_NOERR) &
                    call error('error in index2indexes()')
                if (var_type(i) .eq. NF90_CHAR) then
                    err = nf90mpi_get_var_all(ncid, i, text, index)
                    if (err .ne. NF90_NOERR) &
                        call errore('nf90mpi_get_var_all failed: ',err)
                    value = ichar(text)
                else
                    err = nf90mpi_get_var_all(ncid, i, value, index)
                    if (err .ne. NF90_NOERR) &
                        call errore &
                                 ('nf90mpi_get_var_all failed: ',err)
                end if
                if (value .ne. fill .and.  &
                    abs((fill - value)/fill) .gt. 1.0e-9) then
                    call errord('Unexpected fill value: ', value)
                else
                    nok = nok + 1
                end if
2           continue
1       continue

!       /* close scratch & create again for test using attribute _FillValue */
        err = nf90mpi_close(ncid)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_close: ', err)
        flags = IOR(NF90_CLOBBER, extra_flags)
        err = nf90mpi_create(comm, scratch, flags, info, &
                           ncid)
        if (err .ne. NF90_NOERR) then
            call errore('nf90mpi_create: ', err)
            return
        end if
        call def_dims(ncid)
        call def_vars(ncid)

!           /* set _FillValue = 42 for all vars */
        fill = 42
        text = char(int(fill))
        length = 1
        do 3, i = 1, numVars
            if (var_dimid(var_rank(i),i) .eq. RECDIM) cycle ! skip record variables

            if (var_type(i) .eq. NF90_CHAR) then
                err = nf90mpi_put_att(ncid, i, '_FillValue', &
                   text)
                if (err .ne. NF90_NOERR) &
                    call errore('nf90mpi_put_att: ', err)
            else
                ! cannot use overloading, as fill's type must match with variable's
                ! err = nf90mpi_put_att(ncid, i, '_FillValue', fill)
                fills(1) = fill
                err = nfmpi_put_att_double(ncid, i, '_FillValue', &
                                           var_type(i),length,fills(1))
                if (err .ne. NF90_NOERR) &
                    call errore('nf90mpi_put_att: ', err)
            end if
3       continue

!           /* data mode. write records */
        err = nf90mpi_enddef(ncid)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_enddef: ', err)
        index(1) = NRECS
        err = nf90mpi_put_var_all(ncid, vid, text, index)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_put_var_all: ', err)

!           /* get all variables & check all values equal 42 */
        do 4, i = 1, numVars
            if (var_dimid(var_rank(i),i) .eq. RECDIM) cycle ! skip record variables

            do 5, j = 1, var_nels(i)
                err = index2indexes(j, var_rank(i), var_shape(1,i),  &
                                    index)
                if (err .ne. NF90_NOERR) &
                    call error('error in index2indexes')
                if (var_type(i) .eq. NF90_CHAR) then
                    err = nf90mpi_get_var_all(ncid, i, text, index)
                    if (err .ne. NF90_NOERR) &
                        call errore('nf90mpi_get_var_all failed: ',err)
                    value = ichar(text)
                else
                    err = nf90mpi_get_var_all(ncid, i, value, index)
                    if (err .ne. NF90_NOERR) &
                        call errore &
                                ('nf90mpi_get_var_all failed: ', err)
                end if
                if (value .ne. fill) then
                    call errord(' Value expected: ', fill)
                    call errord(' Value read:     ', value)
                else
                    nok = nok + 1
                end if
5           continue
4       continue

        err = nf90mpi_close(ncid)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_close: ', err)
        err = nf90mpi_delete(scratch, info)
        if (err .ne. NF90_NOERR) &
            call errori('delete of scratch file failed: ', err)

        call print_nok(nok)
        end

#if 0
! * Test nc_set_default_format
! *    try with bad default format
! *    try with NULL old_formatp
! *    try in data mode, check error
! *    check that proper set to NC_FILL works for record & non-record variables
! *    (note that it is not possible to test NC_NOFILL mode!)
! *    close file & create again for test using attribute _FillValue
      subroutine test_nf90mpi_set_default_format()
        use pnetcdf
      implicit none
#include "tests.inc"
      
      integer ncid
      integer err, flags
      integer i
      integer version
      integer old_format
      integer nf90mpi_get_file_version
      
!     /* bad format */
      err = nf90mpi_set_default_format(3, old_format)
      IF (err .ne. NF90_EINVAL) &
           call errore("bad default format: status = %d", err)
     
!    /* Cycle through available formats. */
      do 1 i=1, 2
         err = nf90mpi_set_default_format(i, old_format)
         if (err .ne. NF90_NOERR)  &
               call errore("setting classic format: status = %d", err)
         flags = IOR(NF90_CLOBBER, extra_flags)
         err = nf90mpi_create(comm, scratch, flags,  &
                      info, ncid)
         if (err .ne. NF90_NOERR)  &
              call errore("bad nf90mpi_create: status = %d", err)
         err = nf90mpi_put_att_text(ncid, NF90_GLOBAL, "testatt",  &
              "blah")
         if (err .ne. NF90_NOERR) call errore("bad put_att: status = %d", err)
         err = nf90mpi_close(ncid)
         if (err .ne. NF90_NOERR) call errore("bad close: status = %d", err)
         err = nf90mpi_get_file_version(scratch, version)
         if (err .ne. NF90_NOERR) call errore("bad file version = %d", err)
         if (version .ne. i) &
              call errore("bad file version = %d", err)
 1    continue

!    /* Remove the left-over file. */
      err = nf90mpi_delete(scratch)
      if (err .ne. NF90_NOERR) call errore("remove failed", err)
      end

#endif

!     This function looks in a file for the netCDF magic number.
      integer function nf90mpi_get_file_version(path, version)
        use pnetcdf
      implicit none
#include "tests.inc"
      
      character*(*) path
      integer version
      character magic*4
      integer ver
      integer f
      parameter (f = 10)

      open(f, file=path, status='OLD', form='UNFORMATTED', &
           access='DIRECT', recl=4)

!     Assume this is not a netcdf file.
      nf90mpi_get_file_version = NF90_ENOTNC
      version = 0

!     Read the magic number, the first 4 bytes of the file.
      read(f, rec=1, err = 1) magic

!     If the first three characters are not "CDF" we're done.
      if (index(magic, 'CDF') .eq. 1) then
         ver = ichar(magic(4:4))
         if (ver .eq. 1) then
            version = 1
            nf90mpi_get_file_version = NF90_NOERR
         elseif (ver .eq. 2) then
            version = 2
            nf90mpi_get_file_version = NF90_NOERR
         endif
      endif

 1    close(f)
      return
      end



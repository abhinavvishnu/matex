!
!  Copyright (C) 2013, Northwestern University and Argonne National Laboratory
!  See COPYRIGHT notice in top-level directory.
!
! $Id: test_read.F90 2296 2016-01-06 21:19:36Z wkliao $
!

! Test nf90mpi_strerror.
!    Try on a bad error status.
!    Test for each defined error status.
!
        subroutine test_nf90mpi_strerror()
        use pnetcdf
        implicit        none
#include "tests.inc"
        integer         number_of_messages
        parameter       (number_of_messages = 27)

        integer         i, msg_len
        integer         status(number_of_messages)
        character*80    message, unknown_err_msg
        character*80    msg(number_of_messages)
        integer         nok

        data    status(1)  / NF90_NOERR/
        data    status(2)  / NF90_EBADID /
        data    status(3)  / NF90_EEXIST /
        data    status(4)  / NF90_EINVAL /
        data    status(5)  / NF90_EPERM /
        data    status(6)  / NF90_ENOTINDEFINE /
        data    status(7)  / NF90_EINDEFINE /
        data    status(8)  / NF90_EINVALCOORDS /
        data    status(9)  / NF90_EMAXDIMS /
        data    status(10) / NF90_ENAMEINUSE /
        data    status(11) / NF90_ENOTATT /
        data    status(12) / NF90_EMAXATTS /
        data    status(13) / NF90_EBADTYPE /
        data    status(14) / NF90_EBADDIM /
        data    status(15) / NF90_EUNLIMPOS /
        data    status(16) / NF90_EMAXVARS /
        data    status(17) / NF90_ENOTVAR /
        data    status(18) / NF90_EGLOBAL /
        data    status(19) / NF90_ENOTNC /
        data    status(20) / NF90_ESTS /
        data    status(21) / NF90_EMAXNAME /
        data    status(22) / NF90_EUNLIMIT /
        data    status(23) / NF90_ENORECVARS /
        data    status(24) / NF90_ECHAR /
        data    status(25) / NF90_EEDGE /
        data    status(26) / NF90_ESTRIDE /
        data    status(27) / NF90_EBADNAME /

        data msg(1)  / 'No error' /
        data msg(2)  / 'NetCDF: Not a valid ID' /
        data msg(3)  / 'NetCDF: File exists && NC_NOCLOBBER' /
        data msg(4)  / 'NetCDF: Invalid argument' /
        data msg(5)  / 'NetCDF: Write to read only' /
        data msg(6)  / 'NetCDF: Operation not allowed in data mode' /
        data msg(7)  / 'NetCDF: Operation not allowed in define mode' /
        data msg(8)  / 'NetCDF: Index exceeds dimension bound' /
        data msg(9)  / 'NetCDF: NC_MAX_DIMS exceeded' /
        data msg(10) / 'NetCDF: String match to name in use' /
        data msg(11) / 'NetCDF: Attribute not found' /
        data msg(12) / 'NetCDF: NC_MAX_ATTRS exceeded' /
        data msg(13) &
        / 'NetCDF: Not a valid data type or _FillValue type mismatch' /
        data msg(14) / 'NetCDF: Invalid dimension ID or name' /
        data msg(15) / 'NetCDF: NC_UNLIMITED in the wrong index' /
        data msg(16) / 'NetCDF: NC_MAX_VARS exceeded' /
        data msg(17) / 'NetCDF: Variable not found' /
        data msg(18) / 'NetCDF: Action prohibited on NC_GLOBAL varid' /
        data msg(19) / 'NetCDF: Unknown file format' /
        data msg(20) / 'NetCDF: In Fortran, string too short' /
        data msg(21) / 'NetCDF: NC_MAX_NAME exceeded' /
        data msg(22) / 'NetCDF: NC_UNLIMITED size already in use' /
        data msg(23) &
        / 'NetCDF: nc_rec op when there are no record vars' /
        data msg(24) &
        /'NetCDF: Attempt to convert between text & numbers'/
        data msg(25) / 'NetCDF: Start+count exceeds dimension bound' /
        data msg(26) / 'NetCDF: Illegal stride' /
        data msg(27) / 'NetCDF: Name contains illegal characters' /

        nok = 0

!       /* Try on a bad error status */
        message = nf90mpi_strerror(-666)!/* should fail */
!       pnetcdf differs from serial netcdf in that we report the error
!       code along with the message.

        unknown_err_msg = "Unknown Error"
        msg_len = LEN(TRIM(unknown_err_msg))
        if (message(1:msg_len) .ne. unknown_err_msg(1:msg_len)) then
            call errorc('nf90mpi_strerror on bad error status returned: ', &
                message)
        else
            nok = nok + 1
        endif

!       /* Try on each legitimate error status */
        do 1, i=1, number_of_messages
            message = nf90mpi_strerror(status(i))
            if (trim(message) .ne. trim(msg(i))) then
                call error('nf90mpi_strerror() should return "'  &
                           // trim(msg(i)) // '"' // ' but got '// &
                           '"' // trim(message) // '"')
            else
                nok = nok + 1
            endif
1       continue
        call print_nok(nok)
        end


! Test nf90mpi_open.
! If in read-only section of tests,
!    Try to open a non-existent netCDF file, check error return.
!    Open a file that is not a netCDF file, check error return.
!    Open a netCDF file with a bad mode argument, check error return.
!    Open a netCDF file with NFMPI_NOWRITE mode, try to write, check error.
!    Try to open a netcdf twice, check whether returned netcdf ids different.
! If in writable section of tests,
!    Open a netCDF file with NFMPI_WRITE mode, write something, close it.
! On exit, any open netCDF files are closed.
        subroutine test_nf90mpi_open()
        use pnetcdf
        implicit        none
#include "tests.inc"
        integer err
        integer ncid
        integer ncid2
        integer         nok, flags

        nok = 0

!       /* Try to open a nonexistent file */
        err = nf90mpi_open(comm, 'tooth-fairy.nc', NF90_NOWRITE, &
                         info, ncid)!/* should fail */

!       On some systems, opening an nonexisting file will actually create the
!       file. In this case, we print the error messages on screen and move on
!       to the next test, instead of aborting the entire test.

        if (err .eq. NF90_NOERR) then
            print*, &
        'opening a nonexistent file expects to fail, but got NF90_NOERR'
        elseif (err .ne. NF90_ENOENT) then
            print*, &
        'opening a nonexistent file expects NF90_ENOENT, but got ',err
        else
!           print*, "Expected error message complaining: "// &
!                   "File tooth-fairy.nc does not exist"
            nok = nok + 1
        endif

!       Open a file that is not a netCDF file. This call should fail
        err = nf90mpi_open(comm, 'test_get.F90', NF90_NOWRITE, &
                           info, ncid)
        if (err .ne. NF90_ENOTNC .and. err .ne. NF90_EOFILE) then
            call errore('nf90mpi_open of non-netCDF file: ', err)
        else
            nok = nok + 1
        endif

!       Open a netCDF file in read-only mode, check that write fails
        err = nf90mpi_open(comm, testfile, NF90_NOWRITE, info, ncid)
        if (err .NE. NF90_NOERR) then
            call errore('nf90mpi_open: ', err)
        else
            nok = nok + 1
        endif
        err = nf90mpi_redef(ncid)    !/* should fail */
        if (err .ne. NF90_EPERM) &
            call error('nf90mpi_redef of read-only file should fail')
!       Opened OK, see if can open again and get a different netCDF ID
        err = nf90mpi_open(comm, testfile, NF90_NOWRITE, info, &
                         ncid2)
        if (err .NE. NF90_NOERR) then
            call errore('nf90mpi_open: ', err)
        else
            err = nf90mpi_close(ncid2)
            nok = nok + 1
        end if
        if (ncid2 .eq. ncid) &
            call error('netCDF IDs for first and second '// &
                       'nf90mpi_open calls should differ')

        if (.not. readonly) then        !/* tests using netCDF scratch file */
            flags = IOR(NF90_NOCLOBBER, extra_flags)
            err = nf90mpi_create(comm, scratch, flags, &
                               info, ncid2)
            if (err .NE. NF90_NOERR) then
                call errore('nf90mpi_create: ', err)
            else 
                err = nf90mpi_close(ncid2)
            end if
            err = nf90mpi_open(comm, scratch, NF90_WRITE, info, &
                             ncid2)
            if (err .NE. NF90_NOERR) then
                call errore('nf90mpi_open: ', err)
            else 
                err = nf90mpi_close(ncid2)
                nok = nok + 1
            end if
            err = nf90mpi_delete(scratch, info)
            if (err .NE. NF90_NOERR)  &
                call errorc('delete of scratch file failed: ', scratch)
        end if

        err = nf90mpi_close(ncid)
        if (err .NE. NF90_NOERR) &
            call errore('nf90mpi_close: ', err)
        call print_nok(nok)
        end


! 
! Test nf90mpi_close.
!    Try to close a netCDF file twice, check whether second close fails.
!    Try on bad handle, check error return.
!    Try in define mode and data mode.
!
        subroutine test_nf90mpi_close()
        use pnetcdf
        implicit        none
#include "tests.inc"
        integer ncid
        integer err
        integer nok, flags

        nok = 0

        err = nf90mpi_open(comm, testfile, NF90_NOWRITE, info, &
                         ncid)
        if (err .NE. NF90_NOERR) &
            call errore('nf90mpi_open: ', err)

!       /* Close a netCDF file twice, second time should fail */
        err = nf90mpi_close(ncid)
        if (err .NE. NF90_NOERR) then
            call errore('nf90mpi_close failed: ', err)
        else
            nok = nok + 1
        endif
        err = nf90mpi_close(ncid)
        if (err .ne. NF90_EBADID) then
            call error('nf90mpi_close of closed file should have failed')
        else
            nok = nok + 1
        endif

!       /* Try with a bad netCDF ID */
        err = nf90mpi_close(BAD_ID)!/* should fail */
        if (err .ne. NF90_EBADID) then
            call errore( &
               'nf90mpi_close with bad netCDF ID returned wrong error: ',  &
               err)
        else
            nok = nok + 1
        endif

!       /* Close in data mode */
        err = nf90mpi_open(comm, testfile, NF90_NOWRITE, info, &
                         ncid)
        if (err .NE. NF90_NOERR) &
            call errore('nf90mpi_open: ', err)
        err = nf90mpi_close(ncid)
        if (err .NE. NF90_NOERR) then
            call errore('nf90mpi_close in data mode failed: ', err)
        else
            nok = nok + 1
        endif

        if (.not. readonly) then        !/* tests using netCDF scratch file */
            flags = IOR(NF90_NOCLOBBER, extra_flags)
            err = nf90mpi_create(comm, scratch, flags, &
                               info, ncid)
            if (err .NE. NF90_NOERR)  &
                call errore('nf90mpi_create: ', err)
            err = nf90mpi_close(ncid)
            if (err .NE. NF90_NOERR) then
                call errore('nf90mpi_close in define mode: ', err)
            else
                nok = nok + 1
            endif
            err = nf90mpi_delete(scratch, info)
            if (err .NE. NF90_NOERR) then
                call errorc('delete of scratch file failed: ',  &
                    scratch)
            else
                nok = nok + 1
            endif
        end if
        call print_nok(nok)
        end


! Test nf90mpi_inq.
!    Try on bad handle, check error return.
!    Try in data mode, check returned values.
!    Try asking for subsets of info.
! If in writable section of tests,
!    Try in define mode, after adding an unlimited dimension, variable.
! On exit, any open netCDF files are closed.
        subroutine test_nf90mpi_inq()
        use pnetcdf
        implicit        none
#include "tests.inc"
        integer ncid
        integer ncid2                   !/* for scratch netCDF dataset */
        integer ndims                   !/* number of dimensions */
        integer nvars                   !/* number of variables */
        integer ngatts                  !/* number of global attributes */
        integer recdim                  !/* id of unlimited dimension */
        integer err
        integer ndims0
        integer nvars0
        integer ngatts0
        integer recdim0
        integer did
        integer vid
        integer(kind=MPI_OFFSET_KIND) length
        integer VDIMS(1)
        integer nok, flags

        nok = 0

        err = nf90mpi_open(comm, testfile, NF90_NOWRITE, info, &
                         ncid)
        if (err .NE. NF90_NOERR) &
            call errore('nf90mpi_open: ', err)
        
!       /* Try on bad handle */
        err = nf90mpi_inquire(BAD_ID, ndims, nvars, ngatts, recdim)
        if (err .ne. NF90_EBADID) then
            call errore('bad ncid: ', err)
        else
            nok = nok + 1
        endif
        
        err = nf90mpi_inquire(ncid, ndims, nvars, ngatts, recdim)
        if (err .ne. NF90_NOERR) then
            call errore('nf90mpi_inquire: ', err)
        else if (ndims .ne. NDIMS) then
            call errori &
            ('nf90mpi_inquire: wrong number of dimensions returned: ', ndims)
        else if (nvars .ne. numVars) then
            call errori &
            ('nf90mpi_inquire: wrong number of variables returned: ', nvars)
        else if (ngatts .ne. numGatts) then
            call errori( &
                'nf90mpi_inquire: wrong number of global atts returned: ', &
                ngatts)
        else if (recdim .ne. RECDIM) then
            call errori &
            ('nf90mpi_inquire: wrong record dimension ID returned: ', recdim)
        else
            nok = nok + 1
        end if

        if (.not. readonly) then  ! tests using netCDF scratch file
            flags = IOR(NF90_NOCLOBBER, extra_flags)
            err = nf90mpi_create(comm, scratch, flags, &
                               info, ncid2)
            if (err .NE. NF90_NOERR) then
                call errore('nf90mpi_create: ', err)
            else                ! add dim, var, gatt, check inq
                err = nf90mpi_enddef(ncid2) ! enter data mode
                err = nf90mpi_inquire(ncid2, ndims0, nvars0,  &
                    ngatts0, recdim0)
                if (err .NE. NF90_NOERR) then
                    call errore('nf90mpi_inquire: ', err)
                else
                    nok = nok + 1
                endif
                err = nf90mpi_redef(ncid2) !/* enter define mode */
!               /* Check that inquire still works in define mode */
                err = nf90mpi_inquire(ncid2, ndims, nvars, ngatts, recdim)
                if (err .NE. NF90_NOERR) then
                    call errore('nf90mpi_inquire in define mode: ', err)
                else if (ndims .ne. ndims0) then
                    call errori &
                    ('nf90mpi_inquire in define mode: ndims wrong, ', ndims)
                else if (nvars .ne. nvars0) then
                    call errori &
                    ('nf90mpi_inquire in define mode: nvars wrong, ', nvars)
                else if (ngatts .ne. ngatts0) then
                    call errori( &
                    'nf90mpi_inquire in define mode: ngatts wrong, ', ngatts)
                    print *, ' expected ', ngatts0
                else if (recdim .ne. recdim0) then
                    call errori &
                    ('nf90mpi_inquire in define mode: recdim wrong, ', recdim)
                else
                    nok = nok + 1
                end if

!               /* Add dim, var, global att */
                length = 1
                err = nf90mpi_def_dim(ncid2, 'inqd', length, did)
                if (err .NE. NF90_NOERR) &
                    call errore('nf90mpi_def_dim: ', err)
                VDIMS = 0
                err = nf90mpi_def_var(ncid2, 'inqv', NF90_FLOAT, varid=vid)
                if (err .NE. NF90_NOERR) &
                    call errore('nf90mpi_def_var: ', err)

                length = len('stuff')
                err = nf90mpi_put_att(ncid2, NF90_GLOBAL, 'inqa',  &
                                      'stuff')
                if (err .NE. NF90_NOERR) &
                    call errore('nf90mpi_put_att: ', err)

!               Make sure nf90mpi_inquire sees the additions while in define mode
                err = nf90mpi_inquire(ncid2, ndims, nvars, ngatts, recdim)
                if (err .NE. NF90_NOERR) then
                    call errore('nf90mpi_inquire in define mode: ', err)
                else if (ndims .ne. ndims0 + 1) then
                    call errori &
                    ('nf90mpi_inquire in define mode: ndims wrong, ', ndims)
                else if (nvars .ne. nvars0 + 1) then
                    call errori &
                    ('nf90mpi_inquire in define mode: nvars wrong, ', nvars)
                else if (ngatts .ne. ngatts0 + 1) then
                    call errori &
                    ('nf90mpi_inquire in define mode: ngatts wrong, ', ngatts)
                    print *, ' expected (added attr)', ngatts0 + 1
                else
                    nok = nok + 1
                end if
                err = nf90mpi_enddef(ncid2)
                if (err .NE. NF90_NOERR) &
                    call errore('nf90mpi_enddef: ', err)

!               Make sure nf90mpi_inquire stills sees additions in data mode
                err = nf90mpi_inquire(ncid2, ndims, nvars, ngatts, recdim)
                if (err .NE. NF90_NOERR) then
                    call errore('nf90mpi_inquire failed in data mode: ',err)
                else if (ndims .ne. ndims0 + 1) then
                    call errori &
                    ('nf90mpi_inquire in define mode: ndims wrong, ', ndims)
                else if (nvars .ne. nvars0 + 1) then
                    call errori &
                    ('nf90mpi_inquire in define mode: nvars wrong, ', nvars)
                else if (ngatts .ne. ngatts0 + 1) then
                    call errori &
                    ('nf90mpi_inquire in define mode: ngatts wrong, ', ngatts)
                else
                    nok = nok + 1
                end if
                err = nf90mpi_close(ncid2)
                err = nf90mpi_delete(scratch, info)
                if (err .NE. NF90_NOERR) &
                    call errorc('delete of scratch file failed: ',  &
                        scratch)
            end if
        end if

        err = nf90mpi_close(ncid)
        if (err .NE. NF90_NOERR) &
            call errore('nf90mpi_close: ', err)
        call print_nok(nok)
        end


        subroutine test_nf90mpi_inq_natts()
        use pnetcdf
        implicit        none
#include "tests.inc"
        integer ncid
        integer ngatts                  ! number of global attributes
        integer err
        integer nok

        nok = 0

        err = nf90mpi_inquire(BAD_ID, nAttributes=ngatts)
        if (err .ne. NF90_EBADID) then
            call errore('bad ncid: ', err)
        else
            nok = nok + 1
        endif
        err = nf90mpi_open(comm, testfile, NF90_NOWRITE, info, &
                         ncid)
        if (err .NE. NF90_NOERR) &
            call errore('nf90mpi_open: ', err)
        err = nf90mpi_inquire(ncid, nAttributes=ngatts)
        if (err .NE. NF90_NOERR) then
            call errore('nf90mpi_inquire: ', err)
        else if (ngatts .ne. numGatts) then
            call errori &
            ('nf90mpi_inquire: wrong number of global atts returned, ', &
              ngatts)
        else
            nok = nok + 1
        end if
        err = nf90mpi_close(ncid)
        if (err .NE. NF90_NOERR) &
            call errore('nf90mpi_close: ', err)
        call print_nok(nok)
        end


        subroutine test_nf90mpi_inq_ndims()
        use pnetcdf
        implicit        none
#include "tests.inc"
        integer ncid
        integer ndims
        integer err
        integer nok

        nok = 0

        err = nf90mpi_inquire(BAD_ID, ndims)
        if (err .ne. NF90_EBADID) then
            call errore('bad ncid: ', err)
        else
            nok = nok + 1
        endif
        err = nf90mpi_open(comm, testfile, NF90_NOWRITE, info, &
                           ncid)
        if (err .NE. NF90_NOERR) &
            call errore('nf90mpi_open: ', err)
        err = nf90mpi_inquire(ncid, ndims)
        if (err .NE. NF90_NOERR) then
            call errore('nf90mpi_inquire: ', err)
        else if (ndims .ne. NDIMS) then
            call errori &
            ('nf90mpi_inquire: wrong number returned, ', ndims)
        else
            nok = nok + 1
        end if
        err = nf90mpi_close(ncid)
        if (err .NE. NF90_NOERR) &
            call errore('nf90mpi_close: ', err)
        call print_nok(nok)
        end


        subroutine test_nf90mpi_inq_nvars()
        use pnetcdf
        implicit        none
#include "tests.inc"
        integer ncid
        integer nvars
        integer err
        integer nok

        nok = 0

        err = nf90mpi_inquire(BAD_ID, nVariables=nvars)
        if (err .ne. NF90_EBADID) then
            call errore('bad ncid: ', err)
        else
            nok = nok + 1
        endif
        err = nf90mpi_open(comm, testfile, NF90_NOWRITE, info, &
                         ncid)
        if (err .NE. NF90_NOERR) &
            call errore('nf90mpi_open: ', err)
        err = nf90mpi_inquire(ncid, nVariables=nvars)
        if (err .NE. NF90_NOERR) then
            call errore('nf90mpi_inquire: ', err)
        else if (nvars .ne. numVars) then
            call errori &
            ('nf90mpi_inquire: wrong number returned, ', nvars)
        else
            nok = nok + 1
        end if
        err = nf90mpi_close(ncid)
        if (err .NE. NF90_NOERR) &
            call errore('nf90mpi_close: ', err)
        call print_nok(nok)
        end


        subroutine test_nf90mpi_inq_unlimdim()
        use pnetcdf
        implicit        none
#include "tests.inc"
        integer ncid
        integer unlimdim
        integer err
        integer nok

        nok = 0

        err = nf90mpi_inquire(BAD_ID, unlimitedDimId=unlimdim)
        if (err .ne. NF90_EBADID) then
            call errore('bad ncid: ', err)
        else
            nok = nok + 1
        endif
        err = nf90mpi_open(comm, testfile, NF90_NOWRITE, info, &
                           ncid)
        if (err .NE. NF90_NOERR) &
            call errore('nf90mpi_open: ', err)
        err = nf90mpi_inquire(ncid, unlimitedDimId=unlimdim)
        if (err .NE. NF90_NOERR) then
            call errore('nf90mpi_inquire: ', err)
        else if (unlimdim .ne. RECDIM) then
            call errori &
            ('nf90mpi_inquire: wrong number returned, ', unlimdim)
            print *, 'expected ', RECDIM
        else
            nok = nok + 1
        end if
        err = nf90mpi_close(ncid)
        if (err .NE. NF90_NOERR) &
            call errore('nf90mpi_close: ', err)
        call print_nok(nok)
        end


        subroutine test_nf90mpi_inq_dimid()
        use pnetcdf
        implicit        none
#include "tests.inc"
        integer ncid
        integer dimid
        integer i
        integer err
        integer nok

        nok = 0

        err = nf90mpi_open(comm, testfile, NF90_NOWRITE, info, &
                         ncid)
        if (err .NE. NF90_NOERR) &
            call errore('nf90mpi_open: ', err)
        err = nf90mpi_inq_dimid(ncid, 'noSuch', dimid)
        if (err .ne. NF90_EBADDIM) then
            call errore('bad dim name: ', err)
        else
            nok = nok + 1
        endif
        do 1, i = 1, NDIMS
            err = nf90mpi_inq_dimid(BAD_ID, dim_name(i), dimid)
            if (err .ne. NF90_EBADID) then
                call errore('bad ncid: ', err)
            else
                nok = nok + 1
            endif
            err = nf90mpi_inq_dimid(ncid, dim_name(i), dimid)
            if (err .NE. NF90_NOERR) then
                call errore('nf90mpi_inq_dimid: ', err)
            else if (dimid .ne. i) then
                call errori('expected ', i)
                call errori('got ', dimid)
            else
                nok = nok + 1
            end if
1       continue
        err = nf90mpi_close(ncid)
        if (err .NE. NF90_NOERR) &
            call errore('nf90mpi_close: ', err)
        call print_nok(nok)
        end


        subroutine test_nf90mpi_inq_dim()
        use pnetcdf
        implicit        none
#include "tests.inc"
        integer ncid
        integer i
        integer err
        character*(NF90_MAX_NAME) name
        integer(kind=MPI_OFFSET_KIND) length
        integer intlen
        integer nok

        nok = 0

        err = nf90mpi_open(comm, testfile, NF90_NOWRITE, info, &
                         ncid)
        if (err .NE. NF90_NOERR) &
            call errore('nf90mpi_open: ', err)
        do 1, i = 1, NDIMS
            err = nf90mpi_inquire_dimension(BAD_ID, i, name, length)
            if (err .ne. NF90_EBADID) then
                call errore('bad ncid: ', err)
            else
                nok = nok + 1
            endif
            err = nf90mpi_inquire_dimension(ncid, BAD_DIMID, name, length)
            if (err .ne. NF90_EBADDIM) then
                call errore('bad dimid: ', err)
            else
                nok = nok + 1
            endif
            err = nf90mpi_inquire_dimension(ncid, i, name, length)
            if (err .NE. NF90_NOERR) then
                call errore('nf90mpi_inquire_dimension: ', err)
            else if (dim_name(i) .ne. name)  then
                call errorc('name unexpected: ', name)
                print *, ' expected ', dim_name(i),' for the ',i,'entry'
            else if (dim_len(i) .ne. length) then
                intlen = INT(length)
                call errori('size unexpected: ', intlen)
            else
                nok = nok + 1
            end if
1       continue
        err = nf90mpi_close(ncid)
        if (err .NE. NF90_NOERR) &
            call errore('nf90mpi_close: ', err)
        call print_nok(nok)
        end


        subroutine test_nf90mpi_inq_dimlen()
        use pnetcdf
        implicit        none
#include "tests.inc"
        integer ncid
        integer i
        integer err
        integer(kind=MPI_OFFSET_KIND) length
        integer intlen
        integer nok

        nok = 0

        err = nf90mpi_open(comm, testfile, NF90_NOWRITE, info, &
                         ncid)
        if (err .NE. NF90_NOERR) &
            call errore('nf90mpi_open: ', err)
        do 1, i = 1, NDIMS
            err = nf90mpi_inquire_dimension(BAD_ID, i, len=length)
            if (err .ne. NF90_EBADID) then
                call errore('bad ncid: ', err)
            else
                nok = nok + 1
            endif
            err = nf90mpi_inquire_dimension(ncid, BAD_DIMID, len=length)
            if (err .ne. NF90_EBADDIM) &
                call errore('bad dimid: ', err)
            err = nf90mpi_inquire_dimension(ncid, i, len=length)
            if (err .NE. NF90_NOERR) then
                call errore('nf90mpi_inquire_dimension: ', err)
            else if (dim_len(i) .ne. length) then
                intlen = INT(length)
                call errori('size unexpected: ', intlen)
                print *, 'expected ', dim_len(i),' for the ',i,'entry'
            else
                nok = nok + 1
            end if
1       continue
        err = nf90mpi_close(ncid)
        if (err .NE. NF90_NOERR) &
            call errore('nf90mpi_close: ', err)
        call print_nok(nok)
        end


        subroutine test_nf90mpi_inq_dimname()
        use pnetcdf
        implicit        none
#include "tests.inc"
        integer ncid
        integer i
        integer err
        character*(NF90_MAX_NAME)  name
        integer nok

        nok = 0

        err = nf90mpi_open(comm, testfile, NF90_NOWRITE, info,  &
                         ncid)
        if (err .NE. NF90_NOERR) &
            call errore('nf90mpi_open: ', err)
        do 1, i = 1, NDIMS
            err = nf90mpi_inquire_dimension(BAD_ID, i, name)
            if (err .ne. NF90_EBADID) then
                call errore('bad ncid: ', err)
            else
                nok = nok + 1
            end if
            err = nf90mpi_inquire_dimension(ncid, BAD_DIMID, name)
            if (err .ne. NF90_EBADDIM) then
                call errore('bad dimid: ', err)
            else
                nok = nok + 1
            end if
            err = nf90mpi_inquire_dimension(ncid, i, name)
            if (err .NE. NF90_NOERR) then
                call errore('nf90mpi_inquire_dimension: ', err)
            else if (dim_name(i) .ne. name)  then
                call errorc('name unexpected: ', name)
            else
                nok = nok + 1
            end if
1       continue
        err = nf90mpi_close(ncid)
        if (err .NE. NF90_NOERR) &
            call errore('nf90mpi_close: ', err)
        call print_nok(nok)
        end


        subroutine test_nf90mpi_inq_varid()
        use pnetcdf
        implicit        none
#include "tests.inc"
        integer ncid
        integer vid
        integer i
        integer err
        integer nok

        nok = 0

        err = nf90mpi_open(comm, testfile, NF90_NOWRITE, info,  &
                         ncid)
        if (err .NE. NF90_NOERR) &
            call errore('nf90mpi_open: ', err)

        err = nf90mpi_inq_varid(ncid, 'noSuch', vid)
        if (err .ne. NF90_ENOTVAR) &
            call errore('bad ncid: ', err)

        do 1, i = 1, numVars
            err = nf90mpi_inq_varid(BAD_ID, var_name(i), vid)
            if (err .ne. NF90_EBADID) then
                call errore('bad ncid: ', err)
            else
                nok = nok + 1
            end if
            err = nf90mpi_inq_varid(ncid, var_name(i), vid)
            if (err .NE. NF90_NOERR) then
                call errore('nf90mpi_inq_varid: ', err)
            else if (vid .ne. i) then
                call errori('varid unexpected: ', vid)
            else
                nok = nok + 1
            endif
1       continue

        err = nf90mpi_close(ncid)
        if (err .NE. NF90_NOERR) &
            call errore('nf90mpi_close: ', err)
        call print_nok(nok)
        end


        subroutine test_nf90mpi_inq_var()
        use pnetcdf
        implicit        none
#include "tests.inc"
        LOGICAL INT_VEC_EQ

        integer ncid
        integer i
        integer err
        character*(NF90_MAX_NAME) name
        integer datatype
        integer ndims
        integer dimids(MAX_RANK)
        integer na
        integer nok

        nok = 0

        err = nf90mpi_open(comm, testfile, NF90_NOWRITE, info,  &
                         ncid)
        if (err .NE. NF90_NOERR) &
            call errore('nf90mpi_open: ', err)
        do 1, i = 1, numVars
            err = nf90mpi_inquire_variable(BAD_ID, i, name, datatype, ndims,  &
                  dimids, na)
            if (err .ne. NF90_EBADID) then
                call errore('bad ncid: ', err)
            else
                nok = nok + 1
            endif
            err = nf90mpi_inquire_variable(ncid,BAD_VARID,name,datatype,ndims, &
                             dimids,na)
            if (err .ne. NF90_ENOTVAR) then
                call errore('bad var id: ', err)
            else
                nok = nok + 1
            endif
            err = nf90mpi_inquire_variable(ncid, i, name, datatype, ndims, dimids,  &
                             na)
            if (err .NE. NF90_NOERR) then
                call errore('nf90mpi_inquire_variable: ', err)
            else if (var_name(i) .ne. name)  then
                call errorc('name unexpected: ', name)
            else if (var_type(i) .ne. datatype) then
                call errori('type unexpected: ', datatype)
            else if (var_rank(i) .ne. ndims) then
                call errori('ndims expected: ', ndims)
            else if (.not.int_vec_eq(var_dimid(1,i),dimids,ndims)) then
                call error('unexpected dimid')
            else if (var_natts(i) .ne. na) then
                call errori('natts unexpected: ', na)
            else
                nok = nok + 1
            end if
1       continue
        err = nf90mpi_close(ncid)
        if (err .NE. NF90_NOERR) &
            call errore('nf90mpi_close: ', err)
        call print_nok(nok)
        end


        subroutine test_nf90mpi_inq_vardimid()
        use pnetcdf
        implicit        none
#include "tests.inc"
        LOGICAL INT_VEC_EQ

        integer ncid
        integer i
        integer err
        integer dimids(MAX_RANK)
        integer nok

        nok = 0

        err = nf90mpi_open(comm, testfile, NF90_NOWRITE, info,  &
                         ncid)
        if (err .NE. NF90_NOERR) &
            call errore('nf90mpi_open: ', err)
        do 1, i = 1, numVars
            err = nf90mpi_inquire_variable(BAD_ID, i, dimids=dimids)
            if (err .ne. NF90_EBADID) then
                call errore('bad ncid: ', err)
            else
                nok = nok + 1
            endif
            err = nf90mpi_inquire_variable(ncid, BAD_VARID, dimids=dimids)
            if (err .ne. NF90_ENOTVAR) then
                call errore('bad var id: ', err)
            else
                nok = nok + 1
            endif
            err = nf90mpi_inquire_variable(ncid, i, dimids=dimids)
            if (err .NE. NF90_NOERR) then
                call errore('nf90mpi_inquire_variable: ', err)
            else if (.not.int_vec_eq(var_dimid(1,i), dimids,  &
                     var_rank(i))) then
                call error('unexpected dimid')
                print *, ' for variable ', i
            else
                nok = nok + 1
            end if
1       continue
        err = nf90mpi_close(ncid)
        if (err .NE. NF90_NOERR) &
            call errore('nf90mpi_close: ', err)
        call print_nok(nok)
        end


        subroutine test_nf90mpi_inq_varname()
        use pnetcdf
        implicit        none
#include "tests.inc"
        integer ncid
        integer i
        integer err
        character*(NF90_MAX_NAME) name
        integer nok

        nok = 0

        err = nf90mpi_open(comm, testfile, NF90_NOWRITE, info,  &
                         ncid)
        if (err .NE. NF90_NOERR) &
            call errore('nf90mpi_open: ', err)
        do 1, i = 1, numVars
            err = nf90mpi_inquire_variable(BAD_ID, i, name)
            if (err .ne. NF90_EBADID) then
                call errore('bad ncid: ', err)
            else
                nok = nok + 1
            endif
            err = nf90mpi_inquire_variable(ncid, BAD_VARID, name)
            if (err .ne. NF90_ENOTVAR) then
                call errore('bad var id: ', err)
            else
                nok = nok + 1
            endif
            err = nf90mpi_inquire_variable(ncid, i, name)
            if (err .NE. NF90_NOERR) then
                call errore('nf90mpi_inquire_variable: ', err)
            else if (var_name(i) .ne. name)  then
                call errorc('name unexpected: ', name)
            else
                nok = nok + 1
            end if
1       continue
        err = nf90mpi_close(ncid)
        if (err .NE. NF90_NOERR) &
            call errore('nf90mpi_close: ', err)
        call print_nok(nok)
        end


        subroutine test_nf90mpi_inq_varnatts()
        use pnetcdf
        implicit        none
#include "tests.inc"
        integer VARID, NATTS

        integer ncid
        integer i
        integer err
        integer na
        integer nok

        nok = 0

        err = nf90mpi_open(comm, testfile, NF90_NOWRITE, info,  &
                         ncid)
        if (err .NE. NF90_NOERR) &
            call errore('nf90mpi_open: ', err)
        do 1, i = 0, numVars    ! start with global attributes
            err = nf90mpi_inquire_variable(BAD_ID, i, nAtts=na)
            if (err .ne. NF90_EBADID) then
                call errore('bad ncid: ', err)
            else
                nok = nok + 1
            end if
            err = nf90mpi_inquire_variable(ncid, BAD_VARID, nAtts=na)
            if (err .ne. NF90_ENOTVAR) then
                call errore('bad var id: ', err)
            else
                nok = nok + 1
            end if
            if (i .eq. 0) then
                err = nf90mpi_inquire(ncid, nAttributes=na)
            else
                err = nf90mpi_inquire_variable(ncid, VARID(i), nAtts=na)
            endif
            if (err .NE. NF90_NOERR) then
                call errore('nf90mpi_inquire_variable: ', err)
            else if (NATTS(i) .ne. na) then ! works for global attributes
                call errori('natts unexpected: ', na)
            else
                nok = nok + 1
            end if
1       continue
        err = nf90mpi_close(ncid)
        if (err .NE. NF90_NOERR) &
            call errore('nf90mpi_close: ', err)
        call print_nok(nok)
        end


        subroutine test_nf90mpi_inq_varndims()
        use pnetcdf
        implicit        none
#include "tests.inc"
        integer ncid
        integer i
        integer err
        integer ndims
        integer nok

        nok = 0

        err = nf90mpi_open(comm, testfile, NF90_NOWRITE, info,  &
                         ncid)
        if (err .NE. NF90_NOERR) &
            call errore('nf90mpi_open: ', err)
        do 1, i = 1, numVars
            err = nf90mpi_inquire_variable(BAD_ID, i, ndims=ndims)
            if (err .ne. NF90_EBADID) then
                call errore('bad ncid: ', err)
            else
                nok = nok + 1
            end if
            err = nf90mpi_inquire_variable(ncid, BAD_VARID, ndims=ndims)
            if (err .ne. NF90_ENOTVAR) then
                call errore('bad var id: ', err)
            else
                nok = nok + 1
            end if
            err = nf90mpi_inquire_variable(ncid, i, ndims=ndims)
            if (err .NE. NF90_NOERR) then
                call errore('nf90mpi_inquire_variable: ', err)
            else if (var_rank(i) .ne. ndims) then
                call errori('ndims unexpected: ', ndims)
            else
                nok = nok + 1
            end if
1       continue
        err = nf90mpi_close(ncid)
        if (err .NE. NF90_NOERR) &
            call errore('nf90mpi_close: ', err)
        call print_nok(nok)
        end


        subroutine test_nf90mpi_inq_vartype()
        use pnetcdf
        implicit        none
#include "tests.inc"
        integer ncid
        integer i
        integer err
        integer datatype
        integer nok

        nok = 0

        err = nf90mpi_open(comm, testfile, NF90_NOWRITE, info,  &
                         ncid)
        if (err .NE. NF90_NOERR) &
            call errore('nf90mpi_open: ', err)
        do 1, i = 1, numVars
            err = nf90mpi_inquire_variable(BAD_ID, i, xtype=datatype)
            if (err .ne. NF90_EBADID) then
                call errore('bad ncid: ', err)
            else
                nok = nok + 1
            endif
            err = nf90mpi_inquire_variable(ncid, BAD_VARID, xtype=datatype)
            if (err .ne. NF90_ENOTVAR) then
                call errore('bad var id: ', err)
            else
                nok = nok + 1
            endif
            err = nf90mpi_inquire_variable(ncid, i, xtype=datatype)
            if (err .NE. NF90_NOERR) then
                call errore('nf90mpi_inquire_variable: ', err)
            else if (var_type(i) .ne. datatype) then
                call errori('type unexpected: ', datatype)
                nok = nok + 1
            end if
1       continue
        err = nf90mpi_close(ncid)
        if (err .NE. NF90_NOERR) &
            call errore('nf90mpi_close: ', err)
        call print_nok(nok)
        end


        subroutine test_nf90mpi_inq_att()
        use pnetcdf
        implicit        none
#include "tests.inc"
        character*2 ATT_NAME
        integer ATT_TYPE, ATT_LEN, NATTS

        integer ncid, i, j, err, type, nok
        integer(kind=MPI_OFFSET_KIND) alen

        nok = 0

        err = nf90mpi_open(comm, testfile, NF90_NOWRITE, info,  &
                         ncid)
        if (err .NE. NF90_NOERR)  &
            call errore('nf90mpi_open: ', err)

        do 1, i = 0, numVars
            ! NF90_GLOBAL is defined to be 0
            do 2, j = 1, NATTS(i)
                err = nf90mpi_inquire_attribute(BAD_ID, i, ATT_NAME(j,i), type, alen)
                if (err .ne. NF90_EBADID) then
                    call errore('bad ncid: ', err)
                else
                    nok = nok + 1
                endif
                err = nf90mpi_inquire_attribute &
                           (ncid, BAD_VARID, ATT_NAME(j,i), type, alen)
                if (err .ne. NF90_ENOTVAR) then
                    call errore('bad var id: ', err)
                else
                    nok = nok + 1
                endif
                err = nf90mpi_inquire_attribute(ncid, i, 'noSuch', type, alen)
                if (err .ne. NF90_ENOTATT) then
                    call errore('Bad attribute name: ', err)
                else
                    nok = nok + 1
                endif
                err = nf90mpi_inquire_attribute(ncid, i, ATT_NAME(j,i), type, alen)
                if (err .NE. NF90_NOERR) then
                    call error(nf90mpi_strerror(err))
                else
                    if (type .ne. ATT_TYPE(j,i)) then
                        print*,'ATT_NAME=',trim(ATT_NAME(j,i))
                        print*,'i=',i,' j=',j,' type=',type,' ATT_TYPE=',ATT_TYPE(j,i)
                    endif
                    if (type .ne. ATT_TYPE(j,i)) &
                        call error('type not that expected')
                    if (alen .ne. ATT_LEN(j,i))  &
                        call error('length not that expected')
                    nok = nok + 1
                end if
2           continue
1       continue

        err = nf90mpi_close(ncid)
        if (err .NE. NF90_NOERR) &
            call errore('nf90mpi_close: ', err)
        call print_nok(nok)
        end


        subroutine test_nf90mpi_inq_attlen()
        use pnetcdf
        implicit        none
#include "tests.inc"
        integer ATT_LEN, NATTS
        character*2 ATT_NAME

        integer ncid
        integer i
        integer j
        integer err
        integer(kind=MPI_OFFSET_KIND) len
        integer nok

        nok = 0

        err = nf90mpi_open(comm, testfile, NF90_NOWRITE, info,  &
                         ncid)
        if (err .NE. NF90_NOERR) &
            call errore('nf90mpi_open: ', err)

        do 1, i = 0, numVars
            err = nf90mpi_inquire_attribute(ncid, i, 'noSuch', len=len)
            if (err .ne. NF90_ENOTATT) then
                call errore('Bad attribute name: ', err)
            else
                nok = nok + 1
            endif
            do 2, j = 1, NATTS(i)
                err = nf90mpi_inquire_attribute(BAD_ID, i,  &
                   ATT_NAME(j,i), len=len)
                if (err .ne. NF90_EBADID) then
                    call errore('bad ncid: ', err)
                else
                    nok = nok + 1
                endif
                err = nf90mpi_inquire_attribute &
                      (ncid, BAD_VARID, ATT_NAME(j,i), len=len)
                if (err .ne. NF90_ENOTVAR) then
                    call errore('bad varid: ', err)
                else
                    nok = nok + 1
                endif
                err = nf90mpi_inquire_attribute(ncid, i, ATT_NAME(j,i), len=len)
                if (err .NE. NF90_NOERR) then
                    call error(nf90mpi_strerror(err))
                else
                    if (len .ne. ATT_LEN(j,i)) &
                        call error('len not that expected')
                    nok = nok + 1
                end if
2           continue
1       continue

        err = nf90mpi_close(ncid)
        if (err .NE. NF90_NOERR) &
            call errore('nf90mpi_close: ', err)
        call print_nok(nok)
        end


        subroutine test_nf90mpi_inq_atttype()
        use pnetcdf
        implicit        none
#include "tests.inc"
        character*2 ATT_NAME
        integer ATT_TYPE, NATTS

        integer ncid
        integer i
        integer j
        integer err
        integer datatype
        integer nok

        nok = 0

        err = nf90mpi_open(comm, testfile, NF90_NOWRITE, info,  &
                         ncid)
        if (err .NE. NF90_NOERR) &
            call errore('nf90mpi_open: ', err)

        do 1, i = 0, numVars
            err = nf90mpi_inquire_attribute(ncid, i, 'noSuch', datatype)
            if (err .ne. NF90_ENOTATT) then
                call errore('Bad attribute name: ', err)
            else
                nok = nok + 1
            endif
            do 2, j = 1, NATTS(i)
                err = nf90mpi_inquire_attribute &
                      (BAD_ID, i, ATT_NAME(j,i), datatype)
                if (err .ne. NF90_EBADID) then
                    call errore('bad ncid: ', err)
                else
                    nok = nok + 1
                endif
                err = nf90mpi_inquire_attribute(ncid, BAD_VARID, ATT_NAME(j,i),  &
                                     datatype)
                if (err .ne. NF90_ENOTVAR) then
                    call errore('bad varid: ', err)
                else
                    nok = nok + 1
                endif
                err = nf90mpi_inquire_attribute &
                      (ncid, i, ATT_NAME(j,i), datatype)
                if (err .NE. NF90_NOERR) then
                    call error(nf90mpi_strerror(err))
                else
                    if (datatype .ne. ATT_TYPE(j,i)) &
                        call error('type not that expected')
                    nok = nok + 1
                end if
2           continue
1       continue

        err = nf90mpi_close(ncid)
        if (err .NE. NF90_NOERR) &
            call errore('nf90mpi_close: ', err)
        call print_nok(nok)
        end


        subroutine test_nf90mpi_inq_attname()
        use pnetcdf
        implicit        none
#include "tests.inc"
        character*2 ATT_NAME
        integer NATTS

        integer ncid
        integer i
        integer j
        integer err
        character*(NF90_MAX_NAME) name
        integer nok

        nok = 0

        err = nf90mpi_open(comm, testfile, NF90_NOWRITE, info,  &
                         ncid)
        if (err .NE. NF90_NOERR) &
            call errore('nf90mpi_open: ', err)

        do 1, i = 0, numVars
            err = nf90mpi_inq_attname(ncid, i, BAD_ATTNUM, name)
            if (err .ne. NF90_ENOTATT) then
                call errore('Bad attribute number: ', err)
            else
                nok = nok + 1
            endif
            err = nf90mpi_inq_attname(ncid, i, NATTS(i)+1, name)
            if (err .ne. NF90_ENOTATT) then
                call errore('Bad attribute number: ', err)
            else
                nok = nok + 1
            endif
            do 2, j = 1, NATTS(i)
                err = nf90mpi_inq_attname(BAD_ID, i, j, name)
                if (err .ne. NF90_EBADID) then
                    call errore('bad ncid: ', err)
                else
                    nok = nok + 1
                endif
                err = nf90mpi_inq_attname(ncid, BAD_VARID, j, name)
                if (err .ne. NF90_ENOTVAR) then
                    call errore('bad var id: ', err)
                else
                    nok = nok + 1
                endif
                err = nf90mpi_inq_attname(ncid, i, j, name)
                if (err .NE. NF90_NOERR) then
                    call error(nf90mpi_strerror(err))
                else
                    if (ATT_NAME(j,i) .ne. name) &
                        call error('name not that expected')
                    nok = nok + 1
                end if
2           continue
1       continue

        err = nf90mpi_close(ncid)
        if (err .NE. NF90_NOERR) &
            call errore('nf90mpi_close: ', err)
        call print_nok(nok)
        end


        subroutine test_nf90mpi_inq_attid()
        use pnetcdf
        implicit        none
#include "tests.inc"
        character*2 ATT_NAME
        integer NATTS

        integer ncid
        integer i
        integer j
        integer err
        integer attnum
        integer nok

        nok = 0

        err = nf90mpi_open(comm, testfile, NF90_NOWRITE, info,  &
                         ncid)
        if (err .NE. NF90_NOERR) &
            call errore('nf90mpi_open: ', err)

        do 1, i = 0, numVars
            err = nf90mpi_inquire_attribute(ncid, i, 'noSuch', attnum=attnum)
            if (err .ne. NF90_ENOTATT) then
                call errore('Bad attribute name: ', err)
            else
                nok = nok + 1
            endif
            do 2, j = 1, NATTS(i)
                err = nf90mpi_inquire_attribute(BAD_ID, i,  &
                      ATT_NAME(j,i), attnum=attnum)
                if (err .ne. NF90_EBADID) then
                    call errore('bad ncid: ', err)
                else
                    nok = nok + 1
                endif
                err = nf90mpi_inquire_attribute(ncid, BAD_VARID, ATT_NAME(j,i),  &
                                   attnum=attnum)
                if (err .ne. NF90_ENOTVAR) then
                    call errore('bad varid: ', err)
                else
                    nok = nok + 1
                endif
                err = nf90mpi_inquire_attribute(ncid, i,  &
                     ATT_NAME(j,i), attnum=attnum)
                if (err .NE. NF90_NOERR) then
                    call error(nf90mpi_strerror(err))
                else
                    if (attnum .ne. j) &
                        call error('attnum not that expected')
                    nok = nok + 1
                end if
2           continue
1       continue

        err = nf90mpi_close(ncid)
        if (err .NE. NF90_NOERR) &
            call errore('nf90mpi_close: ', err)
        call print_nok(nok)
        end

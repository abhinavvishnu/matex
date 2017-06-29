!
!  Copyright (C) 2013, Northwestern University and Argonne National Laboratory
!  See COPYRIGHT notice in top-level directory.
!
! $Id: nf90_test.F90 2131 2015-09-25 22:33:12Z wkliao $
!

!
! Test driver for netCDF-3 interface.  This program performs tests against
! the netCDF-3 specification for all user-level functions in an
! implementation of the netCDF library.
!
! Unless invoked with "-r" (readonly) option, must be invoked from a
! directory in which the invoker has write permission.
!
! Files:
! The read-only tests read files:
!     test.nc (see below)
!     test_get.F (used merely as an example of a non-netCDF file)
! 
! The write tests 
!     read test.nc (see below) 
!     write scratch.nc (deleted after each test)
! 
! The file test.nc is created by running nc_test with the -c (create) option.
!

        subroutine usage()
        use pnetcdf
        implicit        none
#include "tests.inc"

        call error('usage: nf90_test [-hrv] [-n <MAX_NMPT>]')
        call error('       nf90_test [-c]')
        call error('   [-h] Print help' )
        call error('   [-c] Create file test.nc (Do not do tests)' )
        call error('   [-1] test CDF-1 format' )
        call error('   [-2] test CDF-2 format' )
        call error('   [-5] test CDF-5 format' )
        call error('   [-r] Just do read-only tests' )
        call error( &
        '   [-d directory] directory for storing input/output files' )
        call error('   [-v] Verbose mode' )
        call error( &
        '   [-n <max>] max. number of messages per test (Default: 20)')
        end

        subroutine report_test
        implicit        none
        character*128   msg
#include "tests.inc"

        write(msg,"(A,I1)") '*** TESTING F90 '//trim(PROGNAME)// &
                            ' for CDF-', cdf_format
        if (nfailsTotal .ne. 0) then
          write(*,*) trim(PROGNAME)//' expects to see 0 failure ... '//&
                     'Total number of failures: ', nfailsTotal
        endif
        call pass_fail(nfailsTotal, msg)
        end

        subroutine test(name, func)
        use pnetcdf
        implicit        none
        character*(*)   name
        character(len=25) name_str
        integer           name_len
        external        func
#include "tests.inc"

        name_len = LEN_TRIM(name)
        name_str(1:name_len) = name(:)
        name_str(name_len+1:25) = ' '
        if (verbose) write(*, 1, advance="no") name_str
1       format('*** Testing ', a, ' ... ')

        nfails = 0
        call func()
        nfailsTotal = nfailsTotal + nfails
        if ( nfails .ne. 0) then
            print *, ' '
            print *, '  ### ', nfails, ' FAILURES TESTING ', name,  &
                     '! Stop ... ###'
            call report_test
            stop
        end if
        end


#if _CRAYIEEE
! which machines need this?
        subroutine getarg(iarg, carg)
        use pnetcdf
        implicit        none
        integer iarg
        character*(*)   carg
        integer ilen
        integer ierror
        call PXFGETARG(iarg, carg, ilen, ierror)
        end
#endif

        program nf_test
        use pnetcdf
#if defined(VISUAL_CPLUSPLUS)
!       DIGITAL Visual Fortran needs DFLIB for getarg
        USE DFLIB
!       DIGITAL Visual Fortran needs DFPORT for iargc
        USE DFPORT
        implicit        none
#elif defined(NAGf90Fortran)
        USE F90_UNIX_ENV, only : iargc, getarg
        implicit none
#else
        implicit        none
        integer         iargc
#endif
#if defined(__crayx1)
        integer         ipxfargc 
#endif
#include "tests.inc"

        integer         argc
        character*80    arg
        integer         iarg
        integer         iopt
        character*1     opt
        integer         lastopt
        logical         skiparg
        integer         err                       

        external        test_nf90mpi_strerror
        external        test_nf90mpi_open
        external        test_nf90mpi_close
        external        test_nf90mpi_inq
        external        test_nf90mpi_inq_dimid
        external        test_nf90mpi_inq_dim
        external        test_nf90mpi_inq_dimlen
        external        test_nf90mpi_inq_dimname
        external        test_nf90mpi_inq_varid
        external        test_nf90mpi_inq_var
        external        test_nf90mpi_inq_natts
        external        test_nf90mpi_inq_ndims
        external        test_nf90mpi_inq_nvars
        external        test_nf90mpi_inq_unlimdim
        external        test_nf90mpi_inq_vardimid
        external        test_nf90mpi_inq_varname
        external        test_nf90mpi_inq_varnatts
        external        test_nf90mpi_inq_varndims
        external        test_nf90mpi_inq_vartype
        external        test_nf90mpi_get_var1_text
#if defined(NF_INT1_T)
        external        test_nf90mpi_get_var1_int1
#endif
#if defined(NF_INT2_T)
        external        test_nf90mpi_get_var1_int2
#endif
        external        test_nf90mpi_get_var1_int
        external        test_nf90mpi_get_var1_real
        external        test_nf90mpi_get_var1_double
        external        test_nf90mpi_get_var1_int8
        external        test_nf90mpi_get_var_text
#if defined(NF_INT1_T)
        external        test_nf90mpi_get_var_int1
#endif
#if defined(NF_INT2_T)
        external        test_nf90mpi_get_var_int2
#endif
        external        test_nf90mpi_get_var_int
        external        test_nf90mpi_get_var_real
        external        test_nf90mpi_get_var_double
        external        test_nf90mpi_get_var_int8
        external        test_nf90mpi_get_vara_text
#if defined(NF_INT1_T)
        external        test_nf90mpi_get_vara_int1
#endif
#if defined(NF_INT2_T)
        external        test_nf90mpi_get_vara_int2
#endif
        external        test_nf90mpi_get_vara_int
        external        test_nf90mpi_get_vara_real
        external        test_nf90mpi_get_vara_double
        external        test_nf90mpi_get_vara_int8
        external        test_nf90mpi_get_vars_text
#if defined(NF_INT1_T)
        external        test_nf90mpi_get_vars_int1
#endif
#if defined(NF_INT2_T)
        external        test_nf90mpi_get_vars_int2
#endif
        external        test_nf90mpi_get_vars_int
        external        test_nf90mpi_get_vars_real
        external        test_nf90mpi_get_vars_double
        external        test_nf90mpi_get_vars_int8

        external        test_nf90mpi_get_varm_text
#if defined(NF_INT1_T)
        external        test_nf90mpi_get_varm_int1
#endif
#if defined(NF_INT2_T)
        external        test_nf90mpi_get_varm_int2
#endif
        external        test_nf90mpi_get_varm_int
        external        test_nf90mpi_get_varm_real
        external        test_nf90mpi_get_varm_double
        external        test_nf90mpi_get_varm_int8

        external        test_nf90mpi_iget_var1_text
#if defined(NF_INT1_T)
        external        test_nf90mpi_iget_var1_int1
#endif
#if defined(NF_INT2_T)
        external        test_nf90mpi_iget_var1_int2
#endif
        external        test_nf90mpi_iget_var1_int
        external        test_nf90mpi_iget_var1_real
        external        test_nf90mpi_iget_var1_double
        external        test_nf90mpi_iget_var1_int8
        external        test_nf90mpi_iget_var_text
#if defined(NF_INT1_T)
        external        test_nf90mpi_iget_var_int1
#endif
#if defined(NF_INT2_T)
        external        test_nf90mpi_iget_var_int2
#endif
        external        test_nf90mpi_iget_var_int
        external        test_nf90mpi_iget_var_real
        external        test_nf90mpi_iget_var_double
        external        test_nf90mpi_iget_var_int8
        external        test_nf90mpi_iget_vara_text
#if defined(NF_INT1_T)
        external        test_nf90mpi_iget_vara_int1
#endif
#if defined(NF_INT2_T)
        external        test_nf90mpi_iget_vara_int2
#endif
        external        test_nf90mpi_iget_vara_int
        external        test_nf90mpi_iget_vara_real
        external        test_nf90mpi_iget_vara_double
        external        test_nf90mpi_iget_vara_int8
        external        test_nf90mpi_iget_vars_text
#if defined(NF_INT1_T)
        external        test_nf90mpi_iget_vars_int1
#endif
#if defined(NF_INT2_T)
        external        test_nf90mpi_iget_vars_int2
#endif
        external        test_nf90mpi_iget_vars_int
        external        test_nf90mpi_iget_vars_real
        external        test_nf90mpi_iget_vars_double
        external        test_nf90mpi_iget_vars_int8

        external        test_nf90mpi_iget_varm_text
#if defined(NF_INT1_T)
        external        test_nf90mpi_iget_varm_int1
#endif
#if defined(NF_INT2_T)
        external        test_nf90mpi_iget_varm_int2
#endif
        external        test_nf90mpi_iget_varm_int
        external        test_nf90mpi_iget_varm_real
        external        test_nf90mpi_iget_varm_double
        external        test_nf90mpi_iget_varm_int8

        external        test_nf90mpi_get_att_text
#if defined(NF_INT1_T)
        external        test_nf90mpi_get_att_int1
#endif
#if defined(NF_INT2_T)
        external        test_nf90mpi_get_att_int2
#endif
        external        test_nf90mpi_get_att_int
        external        test_nf90mpi_get_att_real
        external        test_nf90mpi_get_att_double
        external        test_nf90mpi_get_att_int8
        external        test_nf90mpi_inq_att
        external        test_nf90mpi_inq_attname
        external        test_nf90mpi_inq_attid
        external        test_nf90mpi_inq_attlen
        external        test_nf90mpi_inq_atttype
        external        test_nf90mpi_create
        external        test_nf90mpi_redef
        external        test_nf90mpi_enddef
        external        test_nf90mpi_sync
        external        test_nf90mpi_abort
        external        test_nf90mpi_def_dim
        external        test_nf90mpi_rename_dim
        external        test_nf90mpi_def_var
        external        test_nf90mpi_put_var1_text
#if defined(NF_INT1_T)
        external        test_nf90mpi_put_var1_int1
#endif
#if defined(NF_INT2_T)
        external        test_nf90mpi_put_var1_int2
#endif
        external        test_nf90mpi_put_var1_int
        external        test_nf90mpi_put_var1_real
        external        test_nf90mpi_put_var1_double
        external        test_nf90mpi_put_var1_int8
        external        test_nf90mpi_put_var_text
#if defined(NF_INT1_T)
        external        test_nf90mpi_put_var_int1
#endif
#if defined(NF_INT2_T)
        external        test_nf90mpi_put_var_int2
#endif
        external        test_nf90mpi_put_var_int
        external        test_nf90mpi_put_var_real
        external        test_nf90mpi_put_var_double
        external        test_nf90mpi_put_var_int8
        external        test_nf90mpi_put_vara_text
#if defined(NF_INT1_T)
        external        test_nf90mpi_put_vara_int1
#endif
#if defined(NF_INT2_T)
        external        test_nf90mpi_put_vara_int2
#endif
        external        test_nf90mpi_put_vara_int
        external        test_nf90mpi_put_vara_real
        external        test_nf90mpi_put_vara_double
        external        test_nf90mpi_put_vara_int8
        external        test_nf90mpi_put_vars_text
#if defined(NF_INT1_T)
        external        test_nf90mpi_put_vars_int1
#endif
#if defined(NF_INT2_T)
        external        test_nf90mpi_put_vars_int2
#endif
        external        test_nf90mpi_put_vars_int
        external        test_nf90mpi_put_vars_real
        external        test_nf90mpi_put_vars_double
        external        test_nf90mpi_put_vars_int8

        external        test_nf90mpi_put_varm_text
#if defined(NF_INT1_T)
        external        test_nf90mpi_put_varm_int1
#endif
#if defined(NF_INT2_T)
        external        test_nf90mpi_put_varm_int2
#endif
        external        test_nf90mpi_put_varm_int
        external        test_nf90mpi_put_varm_real
        external        test_nf90mpi_put_varm_double
        external        test_nf90mpi_put_varm_int8

        external        test_nf90mpi_iput_var1_text
#if defined(NF_INT1_T)
        external        test_nf90mpi_iput_var1_int1
#endif
#if defined(NF_INT2_T)
        external        test_nf90mpi_iput_var1_int2
#endif
        external        test_nf90mpi_iput_var1_int
        external        test_nf90mpi_iput_var1_real
        external        test_nf90mpi_iput_var1_double
        external        test_nf90mpi_iput_var1_int8
        external        test_nf90mpi_iput_var_text
#if defined(NF_INT1_T)
        external        test_nf90mpi_iput_var_int1
#endif
#if defined(NF_INT2_T)
        external        test_nf90mpi_iput_var_int2
#endif
        external        test_nf90mpi_iput_var_int
        external        test_nf90mpi_iput_var_real
        external        test_nf90mpi_iput_var_double
        external        test_nf90mpi_iput_var_int8
        external        test_nf90mpi_iput_vara_text
#if defined(NF_INT1_T)
        external        test_nf90mpi_iput_vara_int1
#endif
#if defined(NF_INT2_T)
        external        test_nf90mpi_iput_vara_int2
#endif
        external        test_nf90mpi_iput_vara_int
        external        test_nf90mpi_iput_vara_real
        external        test_nf90mpi_iput_vara_double
        external        test_nf90mpi_iput_vara_int8
        external        test_nf90mpi_iput_vars_text
#if defined(NF_INT1_T)
        external        test_nf90mpi_iput_vars_int1
#endif
#if defined(NF_INT2_T)
        external        test_nf90mpi_iput_vars_int2
#endif
        external        test_nf90mpi_iput_vars_int
        external        test_nf90mpi_iput_vars_real
        external        test_nf90mpi_iput_vars_double
        external        test_nf90mpi_iput_vars_int8

        external        test_nf90mpi_iput_varm_text
#if defined(NF_INT1_T)
        external        test_nf90mpi_iput_varm_int1
#endif
#if defined(NF_INT2_T)
        external        test_nf90mpi_iput_varm_int2
#endif
        external        test_nf90mpi_iput_varm_int
        external        test_nf90mpi_iput_varm_real
        external        test_nf90mpi_iput_varm_double
        external        test_nf90mpi_iput_varm_int8

        external        test_nf90mpi_rename_var
        external        test_nf90mpi_put_att_text
#if defined(NF_INT1_T)
        external        test_nf90mpi_put_att_int1
#endif
#if defined(NF_INT2_T)
        external        test_nf90mpi_put_att_int2
#endif
        external        test_nf90mpi_put_att_int
        external        test_nf90mpi_put_att_real
        external        test_nf90mpi_put_att_double
        external        test_nf90mpi_put_att_int8
        external        test_nf90mpi_copy_att
        external        test_nf90mpi_rename_att
        external        test_nf90mpi_del_att
        external        test_nf90mpi_set_fill
#if 0
        external        test_nf90mpi_set_default_format
#endif
        external        nc_ignorefpe

        call MPI_INIT(err)                        
        comm = MPI_COMM_WORLD                   

        call nc_ignorefpe(1)

        testfile = 'test.nc'
        scratch = 'scratch.nc'

        nfailsTotal = 0
        call getarg(0, progname)
        create_file = .false.   !/* file test.nc will normally already exist */
        readonly = .false.      !/* assume may write in test dir as default */
        verbose = .false.
        max_nmpt = 20
        skiparg = .false.
        cdf_format = 1
        extra_flags = 0

#if defined(__crayx1)
        argc = ipxfargc()
#else
        argc = iargc()
#endif
        call getarg(0, PROGNAME)

        do 1, iarg = 1, argc
            if (skiparg) then
                skiparg = .false.
            else
                call getarg(iarg, arg)
                if (arg(1:1) .eq. '-') then
                    lastopt = index(arg, ' ') - 1
                    do 2, iopt = 2, lastopt
                        opt = arg(iopt:iopt)
                        if (opt .eq. 'c') then
                            create_file = .true.
                        else if (opt .eq. 'r') then
                            readonly = .true.
                        else if (opt .eq. 'v') then
                            verbose = .true.
                        else if (opt .eq. 'n') then
                            call getarg(iarg+1, arg)
                            ! NOTE: The UNICOS 8 fort77(1) compiler does
                            ! not support list-directed I/O from an internal
                            ! file -- so we use a format specification.
                            read (arg, '(i6)') max_nmpt
                            skiparg = .true.
                            go to 1
                        else if (opt .eq. '1') then
                            cdf_format = 1
                        else if (opt .eq. '2') then
                            cdf_format = 2
                            extra_flags = NF90_64BIT_OFFSET
                        else if (opt .eq. '5') then
                            cdf_format = 5
                            extra_flags = NF90_64BIT_DATA
                        else if (opt .eq. 'd') then
                            call getarg(iarg+1, arg)
                            testfile = trim(arg) // "/test.nc"
                            scratch = trim(arg) // "/scratch.nc"
                            skiparg = .true.
                            go to 1
                        else
                            call usage
                            call ud_exit(1)
                        end if
    2           continue
                else
                    call usage
                    call ud_exit(1)
                end if
            end if
1       continue

        call MPI_Info_create(info, err)
        ! call MPI_Info_set(info, "romio_pvfs2_posix_write", "enable",err)

        numGatts = 6
        numVars  = 136
        numTypes = 6
        if (cdf_format .EQ. 5) then
            numGatts = NGATTS
            numVars  = NVARS
            numTypes = NTYPES
        endif

!       /* Initialize global variables defining test file */
        call init_gvars

        if ( create_file ) then
            call write_file(testfile)
            call MPI_Info_free(info, err)
            if (nfailsTotal .eq. 0) &
                 call ud_exit(0)
            call ud_exit(1)
        end if

!       /* delete any existing scratch netCDF file */
        if ( .not. readonly ) &
            err = nf90mpi_delete(scratch, info)

!       /* Test read-only functions, using pregenerated test-file */
        call test('nf90mpi_strerror', test_nf90mpi_strerror)
        call test('nf90mpi_open', test_nf90mpi_open)
        call test('nf90mpi_close', test_nf90mpi_close)
        call test('nf90mpi_inq', test_nf90mpi_inq)
        call test('nf90mpi_inq_dimid', test_nf90mpi_inq_dimid)
        call test('nf90mpi_inq_dim', test_nf90mpi_inq_dim)
        call test('nf90mpi_inq_dimlen', test_nf90mpi_inq_dimlen)
        call test('nf90mpi_inq_dimname', test_nf90mpi_inq_dimname)
        call test('nf90mpi_inq_varid', test_nf90mpi_inq_varid)
        call test('nf90mpi_inq_var', test_nf90mpi_inq_var)
        call test('nf90mpi_inq_natts', test_nf90mpi_inq_natts)
        call test('nf90mpi_inq_ndims', test_nf90mpi_inq_ndims)
        call test('nf90mpi_inq_nvars', test_nf90mpi_inq_nvars)
        call test('nf90mpi_inq_unlimdim', test_nf90mpi_inq_unlimdim)
        call test('nf90mpi_inq_vardimid', test_nf90mpi_inq_vardimid)
        call test('nf90mpi_inq_varname', test_nf90mpi_inq_varname)
        call test('nf90mpi_inq_varnatts', test_nf90mpi_inq_varnatts)
        call test('nf90mpi_inq_varndims', test_nf90mpi_inq_varndims)
        call test('nf90mpi_inq_vartype', test_nf90mpi_inq_vartype)

        call test('nf90mpi_get_var1_text', test_nf90mpi_get_var1_text)
#if defined(NF_INT1_T)
        call test('nf90mpi_get_var1_int1', test_nf90mpi_get_var1_int1)
#endif
#if defined(NF_INT2_T)
        call test('nf90mpi_get_var1_int2', test_nf90mpi_get_var1_int2)
#endif
        call test('nf90mpi_get_var1_int', test_nf90mpi_get_var1_int)
        call test('nf90mpi_get_var1_real', test_nf90mpi_get_var1_real)
        call test('nf90mpi_get_var1_double', test_nf90mpi_get_var1_double)
        call test('nf90mpi_get_var1_int8', test_nf90mpi_get_var1_int8)

        call test('nf90mpi_get_var_text', test_nf90mpi_get_var_text)
#if defined(NF_INT1_T)
        call test('nf90mpi_get_var_int1', test_nf90mpi_get_var_int1)
#endif
#if defined(NF_INT2_T)
        call test('nf90mpi_get_var_int2', test_nf90mpi_get_var_int2)
#endif
        call test('nf90mpi_get_var_int', test_nf90mpi_get_var_int)
        call test('nf90mpi_get_var_real', test_nf90mpi_get_var_real)
        call test('nf90mpi_get_var_double', test_nf90mpi_get_var_double)
        call test('nf90mpi_get_var_int8', test_nf90mpi_get_var_int8)

        call test('nf90mpi_get_vara_text', test_nf90mpi_get_vara_text)
#if defined(NF_INT1_T)
        call test('nf90mpi_get_vara_int1', test_nf90mpi_get_vara_int1)
#endif
#if defined(NF_INT2_T)
        call test('nf90mpi_get_vara_int2', test_nf90mpi_get_vara_int2)
#endif
        call test('nf90mpi_get_vara_int', test_nf90mpi_get_vara_int)
        call test('nf90mpi_get_vara_real', test_nf90mpi_get_vara_real)
        call test('nf90mpi_get_vara_double', test_nf90mpi_get_vara_double)
        call test('nf90mpi_get_vara_int8', test_nf90mpi_get_vara_int8)

        call test('nf90mpi_get_vars_text', test_nf90mpi_get_vars_text)
#if defined(NF_INT1_T)
        call test('nf90mpi_get_vars_int1', test_nf90mpi_get_vars_int1)
#endif
#if defined(NF_INT2_T)
        call test('nf90mpi_get_vars_int2', test_nf90mpi_get_vars_int2)
#endif
        call test('nf90mpi_get_vars_int', test_nf90mpi_get_vars_int)
        call test('nf90mpi_get_vars_real', test_nf90mpi_get_vars_real)
        call test('nf90mpi_get_vars_double', test_nf90mpi_get_vars_double)
        call test('nf90mpi_get_vars_int8', test_nf90mpi_get_vars_int8)

        call test('nf90mpi_get_varm_text', test_nf90mpi_get_varm_text)
#if defined(NF_INT1_T)
        call test('nf90mpi_get_varm_int1', test_nf90mpi_get_varm_int1)
#endif
#if defined(NF_INT2_T)
        call test('nf90mpi_get_varm_int2', test_nf90mpi_get_varm_int2)
#endif
        call test('nf90mpi_get_varm_int', test_nf90mpi_get_varm_int)
        call test('nf90mpi_get_varm_real', test_nf90mpi_get_varm_real)
        call test('nf90mpi_get_varm_double', test_nf90mpi_get_varm_double)
        call test('nf90mpi_get_varm_int8', test_nf90mpi_get_varm_int8)

        call test('nf90mpi_iget_var1_text', test_nf90mpi_iget_var1_text)
#if defined(NF_INT1_T)
        call test('nf90mpi_iget_var1_int1', test_nf90mpi_iget_var1_int1)
#endif
#if defined(NF_INT2_T)
        call test('nf90mpi_iget_var1_int2', test_nf90mpi_iget_var1_int2)
#endif
        call test('nf90mpi_iget_var1_int', test_nf90mpi_iget_var1_int)
        call test('nf90mpi_iget_var1_real', test_nf90mpi_iget_var1_real)
        call test('nf90mpi_iget_var1_double', test_nf90mpi_iget_var1_double)
        call test('nf90mpi_iget_var1_int8', test_nf90mpi_iget_var1_int8)

        call test('nf90mpi_iget_var_text', test_nf90mpi_iget_var_text)
#if defined(NF_INT1_T)
        call test('nf90mpi_iget_var_int1', test_nf90mpi_iget_var_int1)
#endif
#if defined(NF_INT2_T)
        call test('nf90mpi_iget_var_int2', test_nf90mpi_iget_var_int2)
#endif
        call test('nf90mpi_iget_var_int', test_nf90mpi_iget_var_int)
        call test('nf90mpi_iget_var_real', test_nf90mpi_iget_var_real)
        call test('nf90mpi_iget_var_double', test_nf90mpi_iget_var_double)
        call test('nf90mpi_iget_var_int8', test_nf90mpi_iget_var_int8)

        call test('nf90mpi_iget_vara_text', test_nf90mpi_iget_vara_text)
#if defined(NF_INT1_T)
        call test('nf90mpi_iget_vara_int1', test_nf90mpi_iget_vara_int1)
#endif
#if defined(NF_INT2_T)
        call test('nf90mpi_iget_vara_int2', test_nf90mpi_iget_vara_int2)
#endif
        call test('nf90mpi_iget_vara_int', test_nf90mpi_iget_vara_int)
        call test('nf90mpi_iget_vara_real', test_nf90mpi_iget_vara_real)
        call test('nf90mpi_iget_vara_double', test_nf90mpi_iget_vara_double)
        call test('nf90mpi_iget_vara_int8', test_nf90mpi_iget_vara_int8)

        call test('nf90mpi_iget_vars_text', test_nf90mpi_iget_vars_text)
#if defined(NF_INT1_T)
        call test('nf90mpi_iget_vars_int1', test_nf90mpi_iget_vars_int1)
#endif
#if defined(NF_INT2_T)
        call test('nf90mpi_iget_vars_int2', test_nf90mpi_iget_vars_int2)
#endif
        call test('nf90mpi_iget_vars_int', test_nf90mpi_iget_vars_int)
        call test('nf90mpi_iget_vars_real', test_nf90mpi_iget_vars_real)
        call test('nf90mpi_iget_vars_double', test_nf90mpi_iget_vars_double)
        call test('nf90mpi_iget_vars_int8', test_nf90mpi_iget_vars_int8)

        call test('nf90mpi_iget_varm_text', test_nf90mpi_iget_varm_text)
#if defined(NF_INT1_T)
        call test('nf90mpi_iget_varm_int1', test_nf90mpi_iget_varm_int1)
#endif
#if defined(NF_INT2_T)
        call test('nf90mpi_iget_varm_int2', test_nf90mpi_iget_varm_int2)
#endif
        call test('nf90mpi_iget_varm_int', test_nf90mpi_iget_varm_int)
        call test('nf90mpi_iget_varm_real', test_nf90mpi_iget_varm_real)
        call test('nf90mpi_iget_varm_double', test_nf90mpi_iget_varm_double)
        call test('nf90mpi_iget_varm_int8', test_nf90mpi_iget_varm_int8)

        call test('nf90mpi_get_att_text', test_nf90mpi_get_att_text)
#if defined(NF_INT1_T)
        call test('nf90mpi_get_att_int1', test_nf90mpi_get_att_int1)
#endif
#if defined(NF_INT2_T)
        call test('nf90mpi_get_att_int2', test_nf90mpi_get_att_int2)
#endif
        call test('nf90mpi_get_att_int', test_nf90mpi_get_att_int)
        call test('nf90mpi_get_att_real', test_nf90mpi_get_att_real)
        call test('nf90mpi_get_att_double', test_nf90mpi_get_att_double)
        call test('nf90mpi_get_att_int8', test_nf90mpi_get_att_int8)
        call test('nf90mpi_inq_att', test_nf90mpi_inq_att)
        call test('nf90mpi_inq_attname', test_nf90mpi_inq_attname)
        call test('nf90mpi_inq_attid', test_nf90mpi_inq_attid)
        call test('nf90mpi_inq_attlen', test_nf90mpi_inq_attlen)
        call test('nf90mpi_inq_atttype', test_nf90mpi_inq_atttype)

!           /* Test write functions */
        if (.not. readonly) then
            call test('nf90mpi_create', test_nf90mpi_create)
            call test('nf90mpi_redef', test_nf90mpi_redef)
            call test('nf90mpi_enddef', test_nf90mpi_enddef)
            call test('nf90mpi_sync', test_nf90mpi_sync)
            call test('nf90mpi_abort', test_nf90mpi_abort)
            call test('nf90mpi_def_dim', test_nf90mpi_def_dim)
            call test('nf90mpi_rename_dim', test_nf90mpi_rename_dim)
            call test('nf90mpi_def_var', test_nf90mpi_def_var)
            call test('nf90mpi_put_var1_text', test_nf90mpi_put_var1_text)
#if defined(NF_INT1_T)
            call test('nf90mpi_put_var1_int1', test_nf90mpi_put_var1_int1)
#endif
#if defined(NF_INT2_T)
            call test('nf90mpi_put_var1_int2', test_nf90mpi_put_var1_int2)
#endif
            call test('nf90mpi_put_var1_int', test_nf90mpi_put_var1_int)
            call test('nf90mpi_put_var1_real', test_nf90mpi_put_var1_real)
            call test('nf90mpi_put_var1_double',  &
                       test_nf90mpi_put_var1_double)
            call test('nf90mpi_put_var1_int8', test_nf90mpi_put_var1_int8)
            call test('nf90mpi_put_var_text', test_nf90mpi_put_var_text)
#if defined(NF_INT1_T)
            call test('nf90mpi_put_var_int1', test_nf90mpi_put_var_int1)
#endif
#if defined(NF_INT2_T)
            call test('nf90mpi_put_var_int2', test_nf90mpi_put_var_int2)
#endif
            call test('nf90mpi_put_var_int', test_nf90mpi_put_var_int)
            call test('nf90mpi_put_var_real', test_nf90mpi_put_var_real)
            call test('nf90mpi_put_var_double', &
                       test_nf90mpi_put_var_double)
            call test('nf90mpi_put_var_int8', test_nf90mpi_put_var_int8)
            call test('nf90mpi_put_vara_text', test_nf90mpi_put_vara_text)
#if defined(NF_INT1_T)
            call test('nf90mpi_put_vara_int1', test_nf90mpi_put_vara_int1)
#endif
#if defined(NF_INT2_T)
            call test('nf90mpi_put_vara_int2', test_nf90mpi_put_vara_int2)
#endif
            call test('nf90mpi_put_vara_int', test_nf90mpi_put_vara_int)
            call test('nf90mpi_put_vara_real', test_nf90mpi_put_vara_real)
            call test('nf90mpi_put_vara_double', &
                       test_nf90mpi_put_vara_double)
            call test('nf90mpi_put_vara_int8', test_nf90mpi_put_vara_int8)
            call test('nf90mpi_put_vars_text', test_nf90mpi_put_vars_text)
#if defined(NF_INT1_T)
            call test('nf90mpi_put_vars_int1', test_nf90mpi_put_vars_int1)
#endif
#if defined(NF_INT2_T)
            call test('nf90mpi_put_vars_int2', test_nf90mpi_put_vars_int2)
#endif
            call test('nf90mpi_put_vars_int', test_nf90mpi_put_vars_int)
            call test('nf90mpi_put_vars_real', test_nf90mpi_put_vars_real)
            call test('nf90mpi_put_vars_double', &
                       test_nf90mpi_put_vars_double)
            call test('nf90mpi_put_vars_int8', test_nf90mpi_put_vars_int8)

            call test('nf90mpi_put_varm_text', test_nf90mpi_put_varm_text)
#if defined(NF_INT1_T)
            call test('nf90mpi_put_varm_int1', test_nf90mpi_put_varm_int1)
#endif
#if defined(NF_INT2_T)
            call test('nf90mpi_put_varm_int2', test_nf90mpi_put_varm_int2)
#endif
            call test('nf90mpi_put_varm_int', test_nf90mpi_put_varm_int)
            call test('nf90mpi_put_varm_real', test_nf90mpi_put_varm_real)
            call test('nf90mpi_put_varm_double', &
                       test_nf90mpi_put_varm_double)
            call test('nf90mpi_put_varm_int8', test_nf90mpi_put_varm_int8)

            call test('nf90mpi_iput_var1_text', test_nf90mpi_iput_var1_text)
#if defined(NF_INT1_T)
            call test('nf90mpi_iput_var1_int1', test_nf90mpi_iput_var1_int1)
#endif
#if defined(NF_INT2_T)
            call test('nf90mpi_iput_var1_int2', test_nf90mpi_iput_var1_int2)
#endif
            call test('nf90mpi_iput_var1_int', test_nf90mpi_iput_var1_int)
            call test('nf90mpi_iput_var1_real', test_nf90mpi_iput_var1_real)
            call test('nf90mpi_iput_var1_double',  &
                       test_nf90mpi_iput_var1_double)
            call test('nf90mpi_iput_var1_int8', test_nf90mpi_iput_var1_int8)

            call test('nf90mpi_iput_var_text', test_nf90mpi_iput_var_text)
#if defined(NF_INT1_T)
            call test('nf90mpi_iput_var_int1', test_nf90mpi_iput_var_int1)
#endif
#if defined(NF_INT2_T)
            call test('nf90mpi_iput_var_int2', test_nf90mpi_iput_var_int2)
#endif
            call test('nf90mpi_iput_var_int', test_nf90mpi_iput_var_int)
            call test('nf90mpi_iput_var_real', test_nf90mpi_iput_var_real)
            call test('nf90mpi_iput_var_double', &
                       test_nf90mpi_iput_var_double)
            call test('nf90mpi_iput_var_int8', test_nf90mpi_iput_var_int8)

            call test('nf90mpi_iput_vara_text', test_nf90mpi_iput_vara_text)
#if defined(NF_INT1_T)
            call test('nf90mpi_iput_vara_int1', test_nf90mpi_iput_vara_int1)
#endif
#if defined(NF_INT2_T)
            call test('nf90mpi_iput_vara_int2', test_nf90mpi_iput_vara_int2)
#endif
            call test('nf90mpi_iput_vara_int', test_nf90mpi_iput_vara_int)
            call test('nf90mpi_iput_vara_real', test_nf90mpi_iput_vara_real)
            call test('nf90mpi_iput_vara_double', &
                       test_nf90mpi_iput_vara_double)
            call test('nf90mpi_iput_vara_int8', test_nf90mpi_iput_vara_int8)

            call test('nf90mpi_iput_vars_text', test_nf90mpi_iput_vars_text)
#if defined(NF_INT1_T)
            call test('nf90mpi_iput_vars_int1', test_nf90mpi_iput_vars_int1)
#endif
#if defined(NF_INT2_T)
            call test('nf90mpi_iput_vars_int2', test_nf90mpi_iput_vars_int2)
#endif
            call test('nf90mpi_iput_vars_int', test_nf90mpi_iput_vars_int)
            call test('nf90mpi_iput_vars_real', test_nf90mpi_iput_vars_real)
            call test('nf90mpi_iput_vars_double', &
                       test_nf90mpi_iput_vars_double)
            call test('nf90mpi_iput_vars_int8', test_nf90mpi_iput_vars_int8)

            call test('nf90mpi_iput_varm_text', test_nf90mpi_iput_varm_text)
#if defined(NF_INT1_T)
            call test('nf90mpi_iput_varm_int1', test_nf90mpi_iput_varm_int1)
#endif
#if defined(NF_INT2_T)
            call test('nf90mpi_iput_varm_int2', test_nf90mpi_iput_varm_int2)
#endif
            call test('nf90mpi_iput_varm_int', test_nf90mpi_iput_varm_int)
            call test('nf90mpi_iput_varm_real', test_nf90mpi_iput_varm_real)
            call test('nf90mpi_iput_varm_double', &
                       test_nf90mpi_iput_varm_double)
            call test('nf90mpi_iput_varm_int8', test_nf90mpi_iput_varm_int8)

            call test('nf90mpi_rename_var', test_nf90mpi_rename_var)
            call test('nf90mpi_put_att_text', test_nf90mpi_put_att_text)
#if defined(NF_INT1_T)
            call test('nf90mpi_put_att_int1', test_nf90mpi_put_att_int1)
#endif
#if defined(NF_INT2_T)
            call test('nf90mpi_put_att_int2', test_nf90mpi_put_att_int2)
#endif
            call test('nf90mpi_put_att_int', test_nf90mpi_put_att_int)
            call test('nf90mpi_put_att_real', test_nf90mpi_put_att_real)
            call test('nf90mpi_put_att_double', &
                       test_nf90mpi_put_att_double)
            call test('nf90mpi_put_att_int8', test_nf90mpi_put_att_int8)
            call test('nf90mpi_copy_att', test_nf90mpi_copy_att)
            call test('nf90mpi_rename_att', test_nf90mpi_rename_att)
            call test('nf90mpi_del_att', test_nf90mpi_del_att)
            call test('nf90mpi_set_fill', test_nf90mpi_set_fill)
#if 0
            call test('nf90mpi_set_default_format', &
                      test_nf90mpi_set_default_format);
#endif
        end if

        call MPI_Info_free(info, err)

        call report_test

        ! if (nfailsTotal .eq. 0) call ud_exit(0)
        call ud_exit(0)
        end

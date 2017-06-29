!
!  Copyright (C) 2015, Northwestern University and Argonne National Laboratory
!  See COPYRIGHT notice in top-level directory.
!
!     This is part of the PnetCDF package.
!
!     $Id: utils.F90 2120 2015-09-22 07:11:58Z wkliao $

    ! This function gets the executable name and output file name from the
    ! command line.
    integer function get_args(max_argc, cmd, filename, verbose, len)
#ifdef NAGf90Fortran
        USE F90_UNIX_ENV, only : iargc, getarg
        implicit none
#else
        implicit none
        integer iargc
#endif
        integer max_argc, len
        character(len=*) cmd, filename
        logical verbose

        ! local variables
        integer argc
        character(len=16) quiet_mode, str


        call getarg(0, cmd)
        argc = IARGC()
        if (argc .GT. max_argc) then
            if (max_argc .EQ. 3) &
                print*,'Usage: ',trim(cmd),' [-q] [filename] [len]'
            if (max_argc .EQ. 2) &
                print*,'Usage: ',trim(cmd),' [-q] [filename]'
            get_args = 0
            return
        endif
        call getarg(1, quiet_mode)
        if (quiet_mode(1:2) .EQ. '-q') then
            verbose = .FALSE.
            if (argc .GE. 2) call getarg(2, filename)
            if (argc .EQ. 3) then
                call getarg(3, str)
                read (str,'(I10)') len
            endif
        else
            if (argc .GE. 1) call getarg(1, filename)
            if (argc .EQ. 2) then
                call getarg(2, str)
                read (str,'(I10)') len
            endif
        endif
    end function get_args

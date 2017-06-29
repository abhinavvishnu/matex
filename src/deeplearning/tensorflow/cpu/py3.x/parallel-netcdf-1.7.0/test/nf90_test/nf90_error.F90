!
!  Copyright (C) 2013, Northwestern University and Argonne National Laboratory
!  See COPYRIGHT notice in top-level directory.
!
! $Id: nf90_error.F90 1468 2013-10-26 16:53:18Z wkliao $
!

!
! Use for logging error messages
!
        subroutine error(msg)
        use pnetcdf
        implicit        none
        character*(*)   msg
#include "tests.inc"

        nfails = nfails + 1
        if (nfails .le. max_nmpt) print *, msg
        end


!
! Use for logging error conditions
!
        subroutine errori(msg, i)
        use pnetcdf
        implicit        none
        character*(*)   msg
        integer         i
#include "tests.inc"

        nfails = nfails + 1
        if (nfails .le. max_nmpt) print *, msg, i
        end


!
! Use for logging error conditions
!
        subroutine errord(msg, d)
        use pnetcdf
        implicit        none
        character*(*)   msg
        doubleprecision d
#include "tests.inc"

        nfails = nfails + 1
        if (nfails .le. max_nmpt) print *, msg, d
        end


!
! Use for logging error conditions
!
        subroutine errorc(msg, string)
        use pnetcdf
        implicit        none
        character*(*)   msg
        character*(*)   string
#include "tests.inc"

        nfails = nfails + 1
        if (nfails .le. max_nmpt) print *, msg, &
            trim(string)
        end


!
! Use for logging error conditions
!
        subroutine errore(msg, err)
        use pnetcdf
        implicit        none
        character*(*)   msg
        integer         err
#include "tests.inc"

        nfails = nfails + 1
        call errorc(msg, nf90mpi_strerror(err))
        end

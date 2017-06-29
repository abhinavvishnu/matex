#if 0
    Copyright 1996, UCAR/Unidata
    See netcdf/COPYRIGHT file for copying and redistribution conditions.
    $Id: nf_error.F 2223 2015-12-15 16:10:30Z wkliao $
#endif


C
C Use for logging error messages
C
        subroutine error(msg)
        implicit        none
        character*(*)   msg
        include "pnetcdf.inc"
#include "tests.inc"

        nfails = nfails + 1
        if (nfails .le. max_nmpt) print *, msg
        end


C
C Use for logging error conditions
C
        subroutine errori(msg, i)
        implicit        none
        character*(*)   msg
        integer         i
        include "pnetcdf.inc"
#include "tests.inc"

        nfails = nfails + 1
        if (nfails .le. max_nmpt) print *, msg, i
        end


C
C Use for logging error conditions
C
        subroutine errord(msg, d)
        implicit        none
        character*(*)   msg
        doubleprecision d
        include "pnetcdf.inc"
#include "tests.inc"

        nfails = nfails + 1
        if (nfails .le. max_nmpt) print *, msg, d
        end


C
C Use for logging error conditions
C
        subroutine errorc(msg, string)
        implicit        none
        character*(*)   msg
        character*(*)   string
        include "pnetcdf.inc"
#include "tests.inc"

        integer MY_LEN_TRIM
        nfails = nfails + 1
        if (nfails .le. max_nmpt) print *, msg, 
     +      TRIM(string) ! string(1:len_trim(string))
        end


C
C Use for logging error conditions
C
        subroutine errore(msg, err)
        implicit        none
        character*(*)   msg
        integer         err
        include "pnetcdf.inc"
#include "tests.inc"

        nfails = nfails + 1
        call errorc(msg, nfmpi_strerror(err))
        end

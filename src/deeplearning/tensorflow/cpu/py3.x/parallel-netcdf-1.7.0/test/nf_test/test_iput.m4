divert(-1)

dnl This is m4 source.
dnl Process using m4 to produce FORTRAN language file.

changequote([,]) dnl

undefine([index])dnl

dnl Macros

dnl Upcase(str)
dnl
define([Upcase],[dnl
translit($1, abcdefghijklmnopqrstuvwxyz, ABCDEFGHIJKLMNOPQRSTUVWXYZ)])

dnl NFT_ITYPE(type)
dnl
define([NFT_ITYPE], [NFT_[]Upcase($1)])

dnl ARITH3(itype, value)
dnl
define([ARITH3], [ifelse($1, text, ichar($2($3:$3)), $2($3))])

dnl VAR_ELEM(itype, value)
dnl
define([VAR_ELEM], [ifelse($1, text, $2($3:$3), $2($3))])

dnl ARITH_VAR1(itype, value)
dnl
define([ARITH_VAR1], [ifelse($1, text, ichar($2), $2)])

dnl  DATATYPE(funf_suffix)
dnl
define([DATATYPE], [dnl
ifelse($1, text, character*MAX_NELS $2,
ifelse($1, int1, NF_INT1_T $2$3,
ifelse($1, int2, NF_INT2_T $2$3,
ifelse($1, int, integer $2$3,
ifelse($1, int8, NF_INT8_T $2$3,
ifelse($1, real, real $2$3,
ifelse($1, double, doubleprecision $2$3)[]dnl
)[]dnl
)[]dnl
)[]dnl
)[]dnl
)[]dnl
)[]dnl
])

dnl  DATATYPE_VAR1(funf_suffix)
dnl
define([DATATYPE_VAR1], [dnl
ifelse($1, text, character $2,
ifelse($1, int1, NF_INT1_T $2,
ifelse($1, int2, NF_INT2_T $2,
ifelse($1, int, integer $2,
ifelse($1, int8, NF_INT8_T $2,
ifelse($1, real, real $2,
ifelse($1, double, doubleprecision $2)[]dnl
)[]dnl
)[]dnl
)[]dnl
)[]dnl
)[]dnl
)[]dnl
])

dnl  MAKE_ARITH_VAR1(funf_suffix, var)
dnl
define([MAKE_ARITH_VAR1], [dnl
ifelse($1, text, ichar($2), $2)[]dnl
])

dnl  MAKE_ARITH3(funf_suffix, var)
dnl
define([MAKE_ARITH3], [dnl
ifelse($1, text, ichar($2($3:$3)), $2($3))[]dnl
])

dnl  MAKE_DOUBLE(funf_suffix, var)
dnl
define([MAKE_DOUBLE], [dnl
ifelse($1, text, dble(ichar($2)), dble($2))[]dnl
])

dnl  MAKE_TYPE(funf_suffix, var)
dnl
define([MAKE_TYPE], [dnl
ifelse($1, text, char(int($2)),
       ifelse($1, int, INT($2),
       ifelse($1, real, REAL($2),
       ifelse($1, double, DBLE($2),
       $2))))[]dnl
])

dnl  MAKE_TYPE2(funf_suffix, var_dest, var_src)
dnl
define([MAKE_TYPE2], [dnl
ifelse($1, text, $2 = char(int($3)),
       ifelse($1, int, $2 = INT($3),
       ifelse($1, int8,
                  if ($3 .EQ. X_INT8_MAX) then
                      $2 = X_INT8_MAX
                  else
                      $2 = $3
                  endif,
       ifelse($1, real, $2 = REAL($3),
       ifelse($1, double, $2 = DBLE($3),
       $2 = $3)))))[]dnl
])

dnl TEST_NFMPI_IPUT_VAR1(TYPE)
dnl
define([TEST_NFMPI_IPUT_VAR1],dnl
[dnl
        subroutine test_nfmpi_iput_var1_$1()
        implicit        none
        include "pnetcdf.inc"
#include "tests.inc"
        integer index2indexes
        double precision hash_$1
        logical inRange3

        integer ncid
        integer i
        integer j
        integer err, flags
        integer*8 index(MAX_RANK)
        logical canConvert      !/* Both text or both numeric */
        DATATYPE_VAR1($1, value)
        doubleprecision val
        integer err_w, reqid(1), st(1)

        value = MAKE_TYPE($1, 5)!/* any value would do - only for error cases */

        flags = IOR(NF_NOCLOBBER, extra_flags)
        err = nfmpi_create(comm, scratch, flags, info, ncid)
        if (err .ne. NF_NOERR) then
            call errore('nfmpi_create: ', err)
            return
        end if
        call def_dims(ncid)
        call def_vars(ncid)
        err = nfmpi_enddef(ncid)
        if (err .ne. NF_NOERR)
     +      call errore('nfmpi_enddef: ', err)
        do 1, i = 1, numVars
            canConvert = (var_type(i) .eq. NF_CHAR) .eqv.
     +                   (NFT_ITYPE($1) .eq. NFT_TEXT)
            do 2, j = 1, var_rank(i)
                index(j) = 1
2           continue
            err = nfmpi_iput_var1_$1(BAD_ID, i, index, value,
     +                               reqid(1))
            if (err .ne. NF_EBADID) 
     +          call errore('bad ncid: ', err)
            err = nfmpi_iput_var1_$1(ncid, BAD_VARID,
     +                           index, value,reqid(1))
            if (err .ne. NF_ENOTVAR) 
     +          call errore('bad var id: ', err)
            do 3, j = 1, var_rank(i)
                if (var_dimid(j,i) .gt. 1) then         !/* skip record dim */
                    index(j) = var_shape(j,i) + 1
                    err = nfmpi_iput_var1_$1(ncid, i,
     +                                   index, value,reqid(1))
                    if (err .eq. NF_NOERR .or. err .eq. NF_ERANGE)
     +                  err_w = nfmpi_wait_all(ncid,1,reqid,st)
                    if (.not. canConvert) then
                        if (err .ne. NF_ECHAR)
     +                      call errore('conversion: ', err)
                    else
                        if (err .ne. NF_EINVALCOORDS)
     +                      call errore('bad index: ', err)
                    endif
                    index(j) = 0
                end if
3           continue
            do 4, j = 1, var_nels(i)
                err = index2indexes(j, var_rank(i), var_shape(1,i), 
     +                              index)
                if (err .ne. NF_NOERR) 
     +              call error('error in index2indexes 1')
                val = hash_$1(var_type(i),var_rank(i),
     +                            index, NFT_ITYPE($1))
                MAKE_TYPE2($1, value, val)
                err = nfmpi_iput_var1_$1(ncid, i, index, value,
     +                                   reqid(1))
                if (err .eq. NF_NOERR .or. err .eq. NF_ERANGE)
     +              err_w = nfmpi_wait_all(ncid,1,reqid,st)
                if (canConvert) then
                    val = ARITH_VAR1($1, value)
                    if (inRange3(val, var_type(i), NFT_ITYPE($1))) then
                        if (st(1) .ne. 0)
     +                      call error(nfmpi_strerror(st(1)))
                    else
                        if (err .ne. NF_ERANGE)
     +                      call errore('Range error: ', err)
                        err = nfmpi_cancel(ncid, 1, reqid,st)
                    end if
                else
                    if (err .ne. NF_ECHAR)
     +                  call errore('wrong type: ', err)
                end if
4           continue
1       continue
        err = nfmpi_close(ncid)
        if (err .ne. NF_NOERR) 
     +      call errore('nfmpi_close: ', err)

        call check_vars_$1(scratch)

        err = nfmpi_delete(scratch, info)
        if (err .ne. NF_NOERR)
     +      call errorc('delete of scratch file failed: ', 
     +                  scratch)
        end
])dnl


dnl TEST_NFMPI_IPUT_VAR(TYPE)
dnl
define([TEST_NFMPI_IPUT_VAR],dnl
[dnl
        subroutine test_nfmpi_iput_var_$1()
        implicit        none
        include "pnetcdf.inc"
#include "tests.inc"
        integer index2indexes
        double precision hash_$1
        logical inRange3

        integer ncid
        integer vid
        integer i
        integer j
        integer err, flags
        integer nels
        integer*8 index(MAX_RANK)
        logical canConvert      !/* Both text or both numeric */
        logical allInExtRange   !/* All values within external range?*/
        DATATYPE($1, value, (MAX_NELS))
        doubleprecision val
        integer err_w, reqid(1), st(1)

        flags = IOR(NF_CLOBBER, extra_flags)
        err = nfmpi_create(comm, scratch, flags, info, ncid)
        if (err .ne. NF_NOERR) then
            call errore('nfmpi_create: ', err)
            return
        end if
        call def_dims(ncid)
        call def_vars(ncid)
        err = nfmpi_enddef(ncid)
        if (err .ne. NF_NOERR)
     +      call errore('nfmpi_enddef: ', err)
        do 1, i = 1, numVars
            canConvert = (var_type(i) .eq. NF_CHAR) .eqv.
     +                   (NFT_ITYPE($1) .eq. NFT_TEXT)
            err = nfmpi_iput_var_$1(BAD_ID, i, value,reqid(1))
            if (err .ne. NF_EBADID) 
     +          call errore('bad ncid: ', err)
            err = nfmpi_iput_var_$1(ncid, BAD_VARID, value,
     +                                   reqid(1))
            if (err .ne. NF_ENOTVAR) 
     +          call errore('bad var id: ', err)
            nels = 1
            do 3, j = 1, var_rank(i)
                nels = nels * int(var_shape(j,i))
3           continue
            allInExtRange = .true.
            do 4, j = 1, var_nels(i)
                err = index2indexes(j, var_rank(i), var_shape(1,i), 
     +                              index)
                if (err .ne. NF_NOERR) 
     +              call error('error in index2indexes 1')
                val = hash_$1(var_type(i), var_rank(i),
     +                        index, NFT_ITYPE($1))
                MAKE_TYPE2($1, VAR_ELEM($1, value, j), val)
                val = ARITH3($1, value, j)
                allInExtRange = allInExtRange .and.
     +              inRange3(val, var_type(i), NFT_ITYPE($1))
4           continue
            err = nfmpi_iput_var_$1(ncid, i, value,reqid(1))
            if (err .eq. NF_NOERR .or. err .eq. NF_ERANGE)
     +          err_w = nfmpi_wait_all(ncid, 1, reqid, st)
                ! NF_ERANGE is not a fatal error

            if (canConvert) then
                if (allInExtRange) then
                    if (err .ne. NF_NOERR)
     +                  call error(nfmpi_strerror(err))
                else
                    if (err .ne. NF_ERANGE .and.
     +                      var_dimid(var_rank(i),i) .ne. RECDIM)
     +                  call errore('Range error: ', err)
                endif
            else
                if (err .ne. NF_ECHAR)
     +              call errore('wrong type: ', err)
            endif
1       continue

C       The preceeding has written nothing for record variables, now try
C       again with more than 0 records.

C       Write record number NRECS to force writing of preceding records.
C       Assumes variable cr is char vector with UNLIMITED dimension.

        err = nfmpi_inq_varid(ncid, "cr", vid)
        if (err .ne. NF_NOERR)
     +      call errore('nfmpi_inq_varid: ', err)
        index(1) = NRECS
        err = nfmpi_iput_var1_text(ncid, vid, index, 'x',reqid(1))
        if (err .ne. NF_NOERR) then
            call errore('nfmpi_iput_var1_text: ', err)
        else
            err_w = nfmpi_wait_all(ncid, 1, reqid, st)
        endif

        do 5 i = 1, numVars
C           Only test record variables here
            if (var_rank(i) .ge. 1 .and.
     +          var_dimid(var_rank(i),i) .eq. RECDIM) then
                canConvert = (var_type(i) .eq. NF_CHAR) .eqv.
     +                   (NFT_ITYPE($1) .eq. NFT_TEXT)
                if (var_rank(i) .gt. MAX_RANK)
     +              stop 'var_rank(i) .gt. MAX_RANK'
                if (var_nels(i) .gt. MAX_NELS)
     +              stop 'var_nels(i) .gt. MAX_NELS'

                nels = 1
                do 6 j = 1, var_rank(i)
                    nels = nels * int(var_shape(j,i))
6               continue
                allInExtRange = .true.
                do 7, j = 1, nels
                    err = index2indexes(j, var_rank(i), var_shape(1,i), 
     +                              index)
                    if (err .ne. NF_NOERR) 
     +                  call error('error in index2indexes()')
                    val = hash_$1(var_type(i), var_rank(i),
     +                            index, NFT_ITYPE($1))
                    MAKE_TYPE2($1, VAR_ELEM($1, value, j), val)
                    val = ARITH3($1, value, j)
                    allInExtRange = allInExtRange .and.
     +                  inRange3(val, var_type(i), NFT_ITYPE($1))
7               continue
                err = nfmpi_iput_var_$1(ncid, i, value,reqid(1))
                if (err .eq. NF_NOERR .or. err .eq. NF_ERANGE)
     +              err_w = nfmpi_wait_all(ncid, 1, reqid, st)
                    ! NF_ERANGE is not a fatal error?
                if (canConvert) then
                    if (allInExtRange) then
                        if (err .ne. NF_NOERR)
     +                      call error(nfmpi_strerror(err))
                    else
                        if (err .ne. NF_ERANGE)
     +                      call errore('range error: ', err)
                    endif
                else
                    if (nels .gt. 0 .and. err .ne. NF_ECHAR)
     +                  call errore('wrong type: ', err)
                endif
            endif
5       continue
        err = nfmpi_close(ncid)
        if (err .ne. NF_NOERR) 
     +      call errore('nfmpi_close: ', err)

        call check_vars_$1(scratch)

        err = nfmpi_delete(scratch, info)
        if (err .ne. NF_NOERR)
     +      call errorc('delete of scratch file failed: ', 
     +                  scratch)
        end
])dnl


dnl TEST_NFMPI_IPUT_VARA(TYPE)
dnl
define([TEST_NFMPI_IPUT_VARA],dnl
[dnl
        subroutine test_nfmpi_iput_vara_$1()
        implicit        none
        include "pnetcdf.inc"
#include "tests.inc"
        integer index2indexes, roll
        double precision hash_$1
        logical inRange3

        integer ncid
        integer i
        integer j
        integer k
        integer d
        integer err, flags
        integer nslabs
        integer nels
        integer*8 start(MAX_RANK)
        integer*8 edge(MAX_RANK)
        integer*8 mid(MAX_RANK)
        integer*8 index(MAX_RANK)
        logical canConvert      !/* Both text or both numeric */
        logical allInExtRange   !/* all values within external range? */
        DATATYPE($1, value, (MAX_NELS))
        doubleprecision val
        integer ud_shift
        integer err_w, reqid(1), st(1)

        flags = IOR(NF_CLOBBER, extra_flags)
        err = nfmpi_create(comm, scratch, flags, info, ncid)
        if (err .ne. NF_NOERR) then
            call errore('nfmpi_create: ', err)
            return
        end if
        call def_dims(ncid)
        call def_vars(ncid)
        err = nfmpi_enddef(ncid)
        if (err .ne. NF_NOERR)
     +      call errore('nfmpi_enddef: ', err)

        do 1, i = 1, numVars
            canConvert = (var_type(i) .eq. NF_CHAR) .eqv.
     +                   (NFT_ITYPE($1) .eq. NFT_TEXT)
            if (.not.(var_rank(i) .le. MAX_RANK))
     +          stop 'assert(var_rank(i) .le. MAX_RANK)'
            if (.not.(var_nels(i) .le. MAX_NELS))
     +          stop 'assert(var_nels(i) .le. MAX_NELS)'
            do 2, j = 1, var_rank(i)
                start(j) = 1
                edge(j) = 1
2           continue
            err = nfmpi_iput_vara_$1(BAD_ID, i, start,
     +                  edge, value,reqid(1))
            if (err .ne. NF_EBADID) 
     +          call errore('bad ncid: ', err)
            err = nfmpi_iput_vara_$1(ncid, BAD_VARID,
     +                  start, edge, value,reqid(1))
            if (err .ne. NF_ENOTVAR) 
     +          call errore('bad var id: ', err)
            do 3, j = 1, var_rank(i)
                if (var_dimid(j,i) .ne. RECDIM) then    !/* skip record dim */
                    start(j) = var_shape(j,i) + 1
                    err = nfmpi_iput_vara_$1(ncid, i, start, 
     +                                   edge, value,reqid(1))
                    if (err .eq. NF_NOERR .or. err .eq. NF_ERANGE)
     +                  err_w = nfmpi_wait_all(ncid, 1, reqid, st)
                        ! NF_ERANGE is not a fatal error
                    if (.not. canConvert) then
                        if (err .ne. NF_ECHAR)
     +                      call errore('conversion: ', err)
                    else
                        if (err .ne. NF_EINVALCOORDS)
     +                      call errore('bad start: ', err)
                    endif
                    start(j) = 1
                    edge(j) = var_shape(j,i) + 1
                    err = nfmpi_iput_vara_$1(ncid, i, start, 
     +                                   edge, value,reqid(1))
                    if (err .eq. NF_NOERR .or. err .eq. NF_ERANGE)
     +                  err_w = nfmpi_wait_all(ncid, 1, reqid, st)
                        ! NF_ERANGE is not a fatal error
                    if (.not. canConvert) then
                        if (err .ne. NF_ECHAR)
     +                      call errore('conversion: ', err)
                    else
                        if (err .ne. NF_EEDGE)
     +                      call errore('bad edge: ', err)
                    endif
                    edge(j) = 1
                end if
3           continue

C           /* Check correct error returned even when nothing to put */
            do 20, j = 1, var_rank(i)
                  edge(j) = 0
20          continue
            err = nfmpi_iput_vara_$1(BAD_ID, i, start,
     +              edge, value,reqid(1))
            if (err .ne. NF_EBADID) 
     +          call errore('bad ncid: ', err)
            err = nfmpi_iput_vara_$1(ncid, BAD_VARID,
     +              start, edge, value,reqid(1))
            if (err .ne. NF_ENOTVAR) 
     +          call errore('bad var id: ', err)
            do 21, j = 1, var_rank(i)
                if (var_dimid(j,i) .gt. 1) then     ! skip record dim
                    start(j) = var_shape(j,i) + 1
                    err = nfmpi_iput_vara_$1(ncid, i, start,
     +                      edge, value,reqid(1))
                    if (err .eq. NF_NOERR .or. err .eq. NF_ERANGE)
     +                  err_w = nfmpi_wait_all(ncid, 1, reqid, st)
                        ! NF_ERANGE is not a fatal error
                    if (.not. canConvert) then
                        if (err .ne. NF_ECHAR)
     +                      call errore('conversion: ', err)
                    else
                        if (err .ne. NF_EINVALCOORDS)
     +                      call errore('bad start: ', err)
                    endif
                    start(j) = 1
                endif
21          continue
            err = nfmpi_iput_vara_$1(ncid, i, start, edge, value,
     +                                   reqid(1))
            if (err .eq. NF_NOERR .or. err .eq. NF_ERANGE)
     +          err_w = nfmpi_wait_all(ncid, 1, reqid, st)
                ! NF_ERANGE is not a fatal error
            if (canConvert) then
                if (st(1) .ne. 0) 
     +              call error(nfmpi_strerror(st(1)))
            else
                if (err .ne. NF_ECHAR)
     +              call errore('wrong type: ', err)
            endif
            do 22, j = 1, var_rank(i)
                  edge(j) = 1
22          continue


            !/* Choose a random point dividing each dim into 2 parts */
            !/* Put 2^rank (nslabs) slabs so defined */
            nslabs = 1
            do 4, j = 1, var_rank(i)
                mid(j) = roll( var_shape(j,i) )
                nslabs = nslabs * 2
4           continue
                !/* bits of k determine whether to put lower or upper part of dim */
            do 5, k = 1, nslabs
                nels = 1
                do 6, j = 1, var_rank(i)
                    if (mod(ud_shift(k-1, -(j-1)), 2) .eq. 1) then
                        start(j) = 1
                        edge(j) = mid(j)
                    else
                        start(j) = 1 + mid(j)
                        edge(j) = var_shape(j,i) - mid(j)
                    end if
                    nels = nels * int(edge(j))
6               continue
                allInExtRange = .true.
                do 7, j = 1, nels
                    err = index2indexes(j, var_rank(i), edge, index)
                    if (err .ne. NF_NOERR) 
     +                  call error('error in index2indexes 1')
                    do 8, d = 1, var_rank(i)
                        index(d) = index(d) + start(d) - 1
8                   continue
                    val = hash_$1(var_type(i), var_rank(i),
     +                            index, NFT_ITYPE($1))
                    MAKE_TYPE2($1, VAR_ELEM($1, value, j), val)
                    val = ARITH3($1, value, j)
                    allInExtRange = allInExtRange .and.
     +                  inRange3(val, var_type(i), NFT_ITYPE($1))
7               continue
                err = nfmpi_iput_vara_$1(ncid, i, start,
     +                  edge, value,reqid(1))
                if (err .eq. NF_NOERR .or. err .eq. NF_ERANGE)
     +              err_w = nfmpi_wait_all(ncid,1,reqid,st)
                    ! NF_ERANGE is not a fatal error?
                if (canConvert) then
                    if (allInExtRange) then
                        if (st(1) .ne. 0) 
     +                      call error(nfmpi_strerror(st(1)))
                    else
                        if (err .ne. NF_ERANGE)
     +                      call errore('range error: ', err)
                    end if
                else
                    if (nels .gt. 0 .and. err .ne. NF_ECHAR)
     +                  call errore('wrong type: ', err)
                end if
5           continue
1       continue

        err = nfmpi_close(ncid)
        if (err .ne. NF_NOERR) 
     +      call errore('nfmpi_close: ', err)

        call check_vars_$1(scratch)

        err = nfmpi_delete(scratch, info)
        if (err .ne. NF_NOERR)
     +      call errorc('delete of scratch file failed: ', 
     +          scratch)
        end
])dnl


dnl TEST_NFMPI_IPUT_VARS(TYPE)
dnl
define([TEST_NFMPI_IPUT_VARS],dnl
[dnl
        subroutine test_nfmpi_iput_vars_$1()
        implicit        none
        include "pnetcdf.inc"
#include "tests.inc"
        integer index2indexes, roll
        double precision hash_$1
        logical inRange3

        integer ncid
        integer d
        integer i
        integer j
        integer k
        integer m
        integer err, flags
        integer nels
        integer nslabs
        integer nstarts        !/* number of different starts */
        integer*8 start(MAX_RANK)
        integer*8 edge(MAX_RANK)
        integer*8 index(MAX_RANK)
        integer*8 index2(MAX_RANK)
        integer*8 mid(MAX_RANK)
        integer*8 count(MAX_RANK)
        integer*8 sstride(MAX_RANK)
        integer*8 stride(MAX_RANK)
        logical canConvert      !/* Both text or both numeric */
        logical allInExtRange   !/* all values within external range? */
        DATATYPE($1, value, (MAX_NELS))
        doubleprecision val
        integer ud_shift
        integer err_w, reqid(1), st(1)

        flags = IOR(NF_CLOBBER, extra_flags)
        err = nfmpi_create(comm, scratch, flags, info, ncid)
        if (err .ne. NF_NOERR) then
            call errore('nfmpi_create: ', err)
            return
        end if
        call def_dims(ncid)
        call def_vars(ncid)
        err = nfmpi_enddef(ncid)
        if (err .ne. NF_NOERR)
     +      call errore('nfmpi_enddef: ', err)

        do 1, i = 1, numVars
            canConvert = (var_type(i) .eq. NF_CHAR) .eqv.
     +                   (NFT_ITYPE($1) .eq. NFT_TEXT)
            if (.not.(var_rank(i) .le. MAX_RANK))
     +          stop 'assert(var_rank(i) .le. MAX_RANK)'
            if (.not.(var_nels(i) .le. MAX_NELS))
     +          stop 'assert(var_nels(i) .le. MAX_NELS)'
            do 2, j = 1, var_rank(i)
                start(j) = 1
                edge(j) = 1
                stride(j) = 1
2           continue
            err = nfmpi_iput_vars_$1(BAD_ID, i, start,
     +                  edge, stride, value,reqid(1))
            if (err .ne. NF_EBADID) 
     +          call errore('bad ncid: ', err)
            err = nfmpi_iput_vars_$1(ncid, BAD_VARID, start,
     +                           edge, stride, 
     +                           value,reqid(1))
            if (err .ne. NF_ENOTVAR) 
     +          call errore('bad var id: ', err)
            do 3, j = 1, var_rank(i)
                if (var_dimid(j,i) .ne. RECDIM) then    ! skip record dim
                    start(j) = var_shape(j,i) + 1
                    err = nfmpi_iput_vars_$1(ncid, i, start,
     +                                   edge, stride, 
     +                                   value,reqid(1))
                    if (err .eq. NF_NOERR .or. err .eq. NF_ERANGE)
     +                  err_w = nfmpi_wait_all(ncid,1,reqid,st)
                        ! NF_ERANGE is not a fatal error?
                    if (.not. canConvert) then
                        if (err .ne. NF_ECHAR)
     +                      call errore('conversion: ', err)
                    else
                        if (err .ne. NF_EINVALCOORDS)
     +                          call errore('bad start: ', err)
                    endif
                    start(j) = 1
                    edge(j) = var_shape(j,i) + 1
                    err = nfmpi_iput_vars_$1(ncid, i, start,
     +                                   edge, stride, 
     +                                   value,reqid(1))
                    if (err .eq. NF_NOERR .or. err .eq. NF_ERANGE)
     +                  err_w = nfmpi_wait_all(ncid,1,reqid,st)
                        ! NF_ERANGE is not a fatal error?
                    if (.not. canConvert) then
                        if (err .ne. NF_ECHAR)
     +                      call errore('conversion: ', err)
                    else
                        if (err .ne. NF_EEDGE)
     +                      call errore('bad edge: ', err)
                    endif
                    edge(j) = 1
                    stride(j) = 0
                    err = nfmpi_iput_vars_$1(ncid, i, start,
     +                                   edge, stride, 
     +                                   value,reqid(1))
                    if (err .eq. NF_NOERR .or. err .eq. NF_ERANGE)
     +                  err_w = nfmpi_wait_all(ncid,1,reqid,st)
                        ! NF_ERANGE is not a fatal error?
                    if (.not. canConvert) then
                        if (err .ne. NF_ECHAR)
     +                      call errore('conversion: ', err)
                    else
                        if (err .ne. NF_ESTRIDE)
     +                      call errore('bad stride: ', err)
                    endif
                    stride(j) = 1
                end if
3           continue
            !/* Choose a random point dividing each dim into 2 parts */
            !/* Put 2^rank (nslabs) slabs so defined */
            nslabs = 1
            do 4, j = 1, var_rank(i)
                mid(j) = roll( var_shape(j,i) )
                nslabs = nslabs * 2
4           continue
            !/* bits of k determine whether to put lower or upper part of dim */
            !/* choose random stride from 1 to edge */
            do 5, k = 1, nslabs
                nstarts = 1
                do 6, j = 1, var_rank(i)
                    if (mod(ud_shift(k-1, -(j-1)), 2) .eq. 1) then
                        start(j) = 1
                        edge(j) = mid(j)
                    else
                        start(j) = 1 + mid(j)
                        edge(j) = var_shape(j,i) - mid(j)
                    end if
                    if (edge(j) .gt. 0) then
                        stride(j) = 1+roll(edge(j))
                    else
                        stride(j) = 1
                    end if
                    sstride(j) = stride(j)
                    nstarts = nstarts * int(stride(j))
6               continue
                do 7, m = 1, nstarts
                    err = index2indexes(m, var_rank(i), sstride, index)
                    if (err .ne. NF_NOERR)
     +                  call error('error in index2indexes')
                    nels = 1
                    do 8, j = 1, var_rank(i)
                        count(j) = 1 + (edge(j) - index(j)) / stride(j)
                        nels = nels * int(count(j))
                        index(j) = index(j) + start(j) - 1
8                   continue
                    !/* Random choice of forward or backward */
C/* TODO
C                   if ( roll(2) ) {
C                       for (j = 1 j .lt. var_rank(i) j++) {
C                           index(j) += (count(j) - 1) * stride(j)
C                           stride(j) = -stride(j)
C                       }
C                   }
C*/
                    allInExtRange = .true.
                    do 9, j = 1, nels
                        err = index2indexes(j, var_rank(i), count, 
     +                                      index2)
                        if (err .ne. NF_NOERR)
     +                      call error('error in index2indexes')
                        do 10, d = 1, var_rank(i)
                            index2(d) = index(d) + 
     +                                  (index2(d)-1) * stride(d)
10                      continue
                        val = hash_$1(var_type(i), var_rank(i),
     +                                index2, NFT_ITYPE($1))
                        MAKE_TYPE2($1, VAR_ELEM($1, value, j), val)
                        val = ARITH3($1, value, j)
                        allInExtRange = allInExtRange .and.
     +                      inRange3(val, var_type(i), 
     +                               NFT_ITYPE($1))
9                   continue
                    err = nfmpi_iput_vars_$1(ncid, i, index,
     +                                   count, stride,
     +                                   value,reqid(1))
                    if (err .eq. NF_NOERR .or. err .eq. NF_ERANGE)
     +                  err_w = nfmpi_wait_all(ncid,1,reqid,st)
                    if (canConvert) then
                        if (allInExtRange) then
                            if (st(1) .ne. 0) 
     +                          call error(nfmpi_strerror(st(1)))
                        else
                            if (err .ne. NF_ERANGE)
     +                          call errore('range error: ', err)
                        end if
                    else
                        if (nels .gt. 0 .and. err .ne. NF_ECHAR)
     +                      call errore('wrong type: ', err)
                    end if
7               continue
5           continue
1       continue

        err = nfmpi_close(ncid)
        if (err .ne. NF_NOERR) 
     +      call errore('nfmpi_close: ', err)

        call check_vars_$1(scratch)

        err = nfmpi_delete(scratch, info)
        if (err .ne. NF_NOERR)
     +      call errorc('delete of scratch file failed:', 
     +          scratch)
        end
])dnl


dnl  since parallel-netcdf doesn't have varm type, we haven't completed the
dnl  parallel-netcdf-ification of these routines
dnl TEST_NFMPI_IPUT_VARM(TYPE)
dnl
define([TEST_NFMPI_IPUT_VARM],dnl
[dnl
        subroutine test_nfmpi_iput_varm_$1()
        implicit        none
        include "pnetcdf.inc"
#include "tests.inc"
        integer index2indexes, roll
        double precision hash_$1
        logical inRange3

        integer ncid
        integer d
        integer i
        integer j
        integer k
        integer m
        integer err, flags
        integer nels
        integer nslabs
        integer nstarts        !/* number of different starts */
        integer*8 start(MAX_RANK)
        integer*8 edge(MAX_RANK)
        integer*8 index(MAX_RANK)
        integer*8 index2(MAX_RANK)
        integer*8 mid(MAX_RANK)
        integer*8 count(MAX_RANK)
        integer*8 sstride(MAX_RANK)
        integer*8 stride(MAX_RANK)
        integer*8 imap(MAX_RANK)
        logical canConvert      !/* Both text or both numeric */
        logical allInExtRange   !/* all values within external range? */
        DATATYPE($1, value, (MAX_NELS))
        doubleprecision val
        integer ud_shift
        integer err_w, reqid(1), st(1)

        flags = IOR(NF_NOCLOBBER, extra_flags)
        err = nfmpi_create(comm, scratch, flags, info, ncid)
        if (err .ne. NF_NOERR) then
            call errore('nfmpi_create: ', err)
            return
        end if
        call def_dims(ncid)
        call def_vars(ncid)
        err = nfmpi_enddef(ncid)
        if (err .ne. NF_NOERR)
     +      call errore('nfmpi_enddef: ', err)

        do 1, i = 1, numVars
            canConvert = (var_type(i) .eq. NF_CHAR) .eqv.
     +                   (NFT_ITYPE($1) .eq. NFT_TEXT)
            if (.not.(var_rank(i) .le. MAX_RANK))
     +          stop 'assert(var_rank(i) .le. MAX_RANK)'
            if (.not.(var_nels(i) .le. MAX_NELS))
     +          stop 'assert(var_nels(i) .le. MAX_NELS)'
            do 2, j = 1, var_rank(i)
                start(j) = 1
                edge(j) = 1
                stride(j) = 1
                imap(j) = 1
2           continue
            err = nfmpi_iput_varm_$1(BAD_ID, i, start,
     +                           edge, stride, imap, 
     +                           value,reqid(1))
            if (err .ne. NF_EBADID) 
     +          call errore('bad ncid: ', err)
            err = nfmpi_iput_varm_$1(ncid, BAD_VARID, start,
     +                           edge, stride, 
     +                           imap, value,reqid(1))
            if (err .ne. NF_ENOTVAR) 
     +          call errore('bad var id: ', err)
            do 3, j = 1, var_rank(i)
                if (var_dimid(j,i) .ne. RECDIM) then    !/* skip record dim */
                    start(j) = var_shape(j,i) + 1
                    err = nfmpi_iput_varm_$1(ncid, i, start,
     +                                   edge, stride, 
     +                                   imap, value,reqid(1))
                    if (err .eq. NF_NOERR .or. err .eq. NF_ERANGE)
     +                  err_w = nfmpi_wait_all(ncid,1,reqid,st)
                        ! NF_ERANGE is not a fatal error?
                    if (.not. canConvert) then
                        if (err .ne. NF_ECHAR)
     +                      call errore('conversion: ', err)
                    else
                        if (err .ne. NF_EINVALCOORDS)
     +                      call errore('bad start: ', err)
                    endif
                    start(j) = 1
                    edge(j) = var_shape(j,i) + 1
                    err = nfmpi_iput_varm_$1(ncid, i, start,
     +                                   edge, stride, 
     +                                   imap, value,reqid(1))
                    if (err .eq. NF_NOERR .or. err .eq. NF_ERANGE)
     +                  err_w = nfmpi_wait_all(ncid,1,reqid,st)
                        ! NF_ERANGE is not a fatal error?
                    if (.not. canConvert) then
                        if (err .ne. NF_ECHAR)
     +                      call errore('conversion: ', err)
                    else
                        if (err .ne. NF_EEDGE)
     +                      call errore('bad edge: ', err)
                    endif
                    edge(j) = 1
                    stride(j) = 0
                    err = nfmpi_iput_varm_$1(ncid, i, start,
     +                                   edge, stride, 
     +                                   imap, value,reqid(1))
                    if (err .eq. NF_NOERR .or. err .eq. NF_ERANGE)
     +                  err_w = nfmpi_wait_all(ncid,1,reqid,st)
                        ! NF_ERANGE is not a fatal error?
                    if (.not. canConvert) then
                        if (err .ne. NF_ECHAR)
     +                      call errore('conversion: ', err)
                    else
                        if (err .ne. NF_ESTRIDE)
     +                      call errore('bad stride: ', err)
                    endif
                    stride(j) = 1
                end if
3           continue
            !/* Choose a random point dividing each dim into 2 parts */
            !/* Put 2^rank (nslabs) slabs so defined */
            nslabs = 1
            do 4, j = 1, var_rank(i)
                mid(j) = roll( var_shape(j,i) )
                nslabs = nslabs * 2
4           continue
            !/* bits of k determine whether to put lower or upper part of dim */
            !/* choose random stride from 1 to edge */
            do 5, k = 1, nslabs
                nstarts = 1
                do 6, j = 1, var_rank(i)
                    if (mod(ud_shift(k-1, -(j-1)), 2) .eq. 1) then
                        start(j) = 1
                        edge(j) = mid(j)
                    else
                        start(j) = 1 + mid(j)
                        edge(j) = var_shape(j,i) - mid(j)
                    end if
                    if (edge(j) .gt. 0) then
                        stride(j) = 1+roll(edge(j))
                    else
                        stride(j) = 1
                    end if
                    sstride(j) = stride(j)
                    nstarts = nstarts * int(stride(j))
6               continue
                do 7, m = 1, nstarts
                    err = index2indexes(m, var_rank(i), sstride, index)
                    if (err .ne. NF_NOERR)
     +                  call error('error in index2indexes')
                    nels = 1
                    do 8, j = 1, var_rank(i)
                        count(j) = 1 + (edge(j) - index(j)) / stride(j)
                        nels = nels * int(count(j))
                        index(j) = index(j) + start(j) - 1
8                   continue
                    !/* Random choice of forward or backward */
C/* TODO
C                   if ( roll(2) ) then
C                       do 9, j = 1, var_rank(i)
C                           index(j) = index(j) + 
C     +                         (count(j) - 1) * stride(j)
C                           stride(j) = -stride(j)
C9                      continue
C                   end if
C*/
                    if (var_rank(i) .gt. 0) then
                        imap(1) = 1
                        do 10, j = 2, var_rank(i)
                            imap(j) = imap(j-1) * count(j-1)
10                      continue
                    end if
                    allInExtRange = .true.
                    do 11 j = 1, nels
                        err = index2indexes(j, var_rank(i), count, 
     +                                      index2)
                        if (err .ne. NF_NOERR)
     +                      call error('error in index2indexes')
                        do 12, d = 1, var_rank(i)
                            index2(d) = index(d) + 
     +                          (index2(d)-1) * stride(d)
12                      continue
                        val = hash_$1(var_type(i),var_rank(i), 
     +                                index2, NFT_ITYPE($1))
                        MAKE_TYPE2($1, VAR_ELEM($1, value, j), val)
                        val = ARITH3($1, value, j)
                        allInExtRange = allInExtRange .and.
     +                      inRange3(val, var_type(i), 
     +                               NFT_ITYPE($1))
11                  continue
                    err = nfmpi_iput_varm_$1(ncid,i,index,count,
     +                                   stride,imap,
     +                                   value,reqid(1))
                    if (err .eq. NF_NOERR .or. err .eq. NF_ERANGE)
     +                  err_w = nfmpi_wait_all(ncid,1,reqid,st)
                    if (canConvert) then
                        if (allInExtRange) then
                            if (st(1) .ne. 0)
     +                          call error(nfmpi_strerror(st(1)))
                        else
                            if (err .ne. NF_ERANGE)
     +                          call errore('range error: ', err)
                        end if
                    else
                        if (nels .gt. 0 .and. err .ne. NF_ECHAR)
     +                      call errore('wrong type: ', err)
                    end if
7               continue
5           continue
1       continue

        err = nfmpi_close(ncid)
        if (err .ne. NF_NOERR) 
     +      call errore('nfmpi_close: ', err)

        call check_vars_$1(scratch)

        err = nfmpi_delete(scratch, info)
        if (err .ne. NF_NOERR)
     +      call errorc('delete of scratch file failed:', 
     +          scratch)
        end
])dnl


divert(0)dnl
dnl If you see this line, you can ignore the next one.
C Do not edit this file. It is produced from the corresponding .m4 source */

C********************************************************************
C   Copyright 1996, UCAR/Unidata
C   See netcdf/COPYRIGHT file for copying and redistribution conditions.
C   $Id: test_iput.m4 2232 2015-12-16 22:16:52Z wkliao $
C********************************************************************

TEST_NFMPI_IPUT_VAR1(text)
#ifdef NF_INT1_T
TEST_NFMPI_IPUT_VAR1(int1)
#endif
#ifdef NF_INT2_T
TEST_NFMPI_IPUT_VAR1(int2)
#endif
TEST_NFMPI_IPUT_VAR1(int)
TEST_NFMPI_IPUT_VAR1(int8)
TEST_NFMPI_IPUT_VAR1(real)
TEST_NFMPI_IPUT_VAR1(double)

TEST_NFMPI_IPUT_VAR(text)
#ifdef NF_INT1_T
TEST_NFMPI_IPUT_VAR(int1)
#endif
#ifdef NF_INT2_T
TEST_NFMPI_IPUT_VAR(int2)
#endif
TEST_NFMPI_IPUT_VAR(int)
TEST_NFMPI_IPUT_VAR(int8)
TEST_NFMPI_IPUT_VAR(real)
TEST_NFMPI_IPUT_VAR(double)

TEST_NFMPI_IPUT_VARA(text)
#ifdef NF_INT1_T
TEST_NFMPI_IPUT_VARA(int1)
#endif
#ifdef NF_INT2_T
TEST_NFMPI_IPUT_VARA(int2)
#endif
TEST_NFMPI_IPUT_VARA(int)
TEST_NFMPI_IPUT_VARA(int8)
TEST_NFMPI_IPUT_VARA(real)
TEST_NFMPI_IPUT_VARA(double)

TEST_NFMPI_IPUT_VARS(text)
#ifdef NF_INT1_T
TEST_NFMPI_IPUT_VARS(int1)
#endif
#ifdef NF_INT2_T
TEST_NFMPI_IPUT_VARS(int2)
#endif
TEST_NFMPI_IPUT_VARS(int)
TEST_NFMPI_IPUT_VARS(int8)
TEST_NFMPI_IPUT_VARS(real)
TEST_NFMPI_IPUT_VARS(double)

TEST_NFMPI_IPUT_VARM(text)
#ifdef NF_INT1_T
TEST_NFMPI_IPUT_VARM(int1)
#endif
#ifdef NF_INT2_T
TEST_NFMPI_IPUT_VARM(int2)
#endif
TEST_NFMPI_IPUT_VARM(int)
TEST_NFMPI_IPUT_VARM(int8)
TEST_NFMPI_IPUT_VARM(real)
TEST_NFMPI_IPUT_VARM(double)

dnl
dnl  Copyright (C) 2013, Northwestern University and Argonne National Laboratory
dnl  See COPYRIGHT notice in top-level directory.
dnl
dnl $Id: test_iput.m4 2156 2015-10-12 01:15:25Z wkliao $
dnl

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

dnl VALUE3(itype, value)
dnl
define([VALUE3], [ifelse($1, text, $2(1:nels), $2(1:nels))])

dnl VAR_ELEM(itype, value)
dnl
define([VAR_ELEM], [ifelse($1, text, $2($3:$3), $2($3))])

dnl ARITH_VAR1(itype, value)
dnl
define([ARITH_VAR1], [ifelse($1, text, ichar($2(1:1)), $2(1))])

dnl  DATATYPE(funf_suffix)
dnl
define([DATATYPE], [dnl
ifelse($1, text, character(len=$3) $2,
ifelse($1, int1, NF_INT1_T $2($3),
ifelse($1, int2, NF_INT2_T $2($3),
ifelse($1, int, integer $2($3),
ifelse($1, int8, NF_INT8_T $2($3),
ifelse($1, real, real $2($3),
ifelse($1, double, doubleprecision $2($3))[]dnl
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
ifelse($1, text, character(len=1) $2,
ifelse($1, int1, NF_INT1_T $2(1),
ifelse($1, int2, NF_INT2_T $2(1),
ifelse($1, int, integer $2(1),
ifelse($1, int8, NF_INT8_T $2(1),
ifelse($1, real, real $2(1),
ifelse($1, double, double precision $2(1))[]dnl
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
       ifelse($1, int1, INT($2,KIND=INT1_KIND),
       ifelse($1, int2, INT($2,KIND=INT2_KIND),
       ifelse($1, int8, INT($2,KIND=INT8_KIND),
       ifelse($1, real, REAL($2),
       $2))))))[]dnl
])

dnl  MAKE_TYPE2(funf_suffix, var_dest, var_src)
dnl
define([MAKE_TYPE2], [dnl
ifelse($1, text, $2 = char(int($3)),
       ifelse($1, int, $2 = INT($3),
       ifelse($1, int1, $2 = INT($3,KIND=INT1_KIND),
       ifelse($1, int2, $2 = INT($3,KIND=INT2_KIND),
       ifelse($1, int8,
                if ($3 .EQ. X_INT8_MAX) then
                    $2 = X_INT8_MAX
                else
                    $2 = INT($3,KIND=INT8_KIND)
                endif,
       ifelse($1, real, $2 = REAL($3),
       $2 = $3))))))[]dnl
])

dnl TEST_NFMPI_IPUT_VAR1(TYPE)
dnl
define([TEST_NFMPI_IPUT_VAR1],dnl
[dnl
        subroutine test_nf90mpi_iput_var1_$1()
        use pnetcdf
        implicit        none
#include "tests.inc"
        integer index2indexes
        double precision hash_$1
        logical inRange3

        integer ncid
        integer i
        integer j
        integer err, flags
        integer(kind=MPI_OFFSET_KIND) index(MAX_RANK)
        logical canConvert      !/* Both text or both numeric */
        DATATYPE_VAR1($1, value)
        double precision val
        integer err_w, reqid(1), st(1)

        value = MAKE_TYPE($1, 5)!/* any value would do - only for error cases */

        flags = IOR(NF90_CLOBBER, extra_flags)
        err = nf90mpi_create(comm, scratch, flags, info, &
                           ncid)
        if (err .ne. NF90_NOERR) then
            call errore('nf90mpi_create: ', err)
            return
        end if
        call def_dims(ncid)
        call def_vars(ncid)
        err = nf90mpi_enddef(ncid)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_enddef: ', err)
        do 1, i = 1, numVars
            canConvert = (var_type(i) .eq. NF90_CHAR) .eqv. &
                         (NFT_ITYPE($1) .eq. NFT_TEXT)
            do 2, j = 1, var_rank(i)
                index(j) = 1
2           continue
            err = nf90mpi_iput_var(BAD_ID, i, value(1:1), reqid(1), index)
            if (err .ne. NF90_EBADID)  &
                call errore('bad ncid: ', err)
            err = nf90mpi_iput_var(ncid, BAD_VARID, value(1:1),reqid(1), index)
            if (err .ne. NF90_ENOTVAR)  &
                call errore('bad var id: ', err)
            do 3, j = 1, var_rank(i)
                if (var_dimid(j,i) .gt. 1) then         !/* skip record dim */
                    index(j) = var_shape(j,i) + 1
                    err = nf90mpi_iput_var(ncid, i, value(1:1),reqid(1), index)
                    if (err .eq. NF90_NOERR .or. err .eq. NF90_ERANGE) &
                        err_w = nf90mpi_wait_all(ncid,1,reqid,st)
                    if (.not. canConvert) then
                        if (err .ne. NF90_ECHAR) &
                            call errore('conversion: ', err)
                    else
                        if (err .ne. NF90_EINVALCOORDS) &
                            call errore('bad index: ', err)
                    endif
                    index(j) = 0
                end if
3           continue
            do 4, j = 1, var_nels(i)
                err = index2indexes(j, var_rank(i), var_shape(1,i),  &
                                    index)
                if (err .ne. NF90_NOERR)  &
                    call error('error in index2indexes 1')
                val = hash_$1(var_type(i),var_rank(i), &
                                  index, NFT_ITYPE($1))
                MAKE_TYPE2($1, value, val)
                err = nf90mpi_iput_var(ncid, i, value(1:1), reqid(1), index)
                if (err .eq. NF90_NOERR .or. err .eq. NF90_ERANGE) &
                    err_w = nf90mpi_wait_all(ncid,1,reqid,st)
                if (canConvert) then
                    val = ARITH_VAR1($1, value)
                    if (inRange3(val, var_type(i), NFT_ITYPE($1))) then
                        if (st(1) .ne. 0) &
                            call error(nf90mpi_strerror(st(1)))
                    else
                        if (err .ne. NF90_ERANGE) &
                            call errore('Range error: ', err)
                    end if
                else
                    if (err .ne. NF90_ECHAR) &
                        call errore('wrong type: ', err)
                end if
4           continue
1       continue
        err = nf90mpi_close(ncid)
        if (err .ne. NF90_NOERR)  &
            call errore('nf90mpi_close: ', err)

        call check_vars_$1(scratch)

        err = nf90mpi_delete(scratch, info)
        if (err .ne. NF90_NOERR) &
            call errorc('delete of scratch file failed: ',  &
                        scratch)
        end
])dnl


dnl TEST_NFMPI_IPUT_VAR(TYPE)
dnl
define([TEST_NFMPI_IPUT_VAR],dnl
[dnl
        subroutine test_nf90mpi_iput_var_$1()
        use pnetcdf
        implicit        none
#include "tests.inc"
        integer index2indexes
        double precision hash_$1
        logical inRange3

        integer ncid
        integer vid
        integer i
        integer j
        integer err, flags
        integer(kind=MPI_OFFSET_KIND) nels
        integer(kind=MPI_OFFSET_KIND) index(MAX_RANK)
        logical canConvert      !/* Both text or both numeric */
        logical allInExtRange   !/* All values within external range?*/
        DATATYPE($1, value, MAX_NELS)
        doubleprecision val
        integer err_w, reqid(1), st(1)

        flags = IOR(NF90_CLOBBER, extra_flags)
        err = nf90mpi_create(comm, scratch, flags, info, &
                           ncid)
        if (err .ne. NF90_NOERR) then
            call errore('nf90mpi_create: ', err)
            return
        end if
        call def_dims(ncid)
        call def_vars(ncid)
        err = nf90mpi_enddef(ncid)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_enddef: ', err)
        do 1, i = 1, numVars
            canConvert = (var_type(i) .eq. NF90_CHAR) .eqv. &
                         (NFT_ITYPE($1) .eq. NFT_TEXT)
            err = nf90mpi_iput_var(BAD_ID, i, value, reqid(1))
            if (err .ne. NF90_EBADID)  &
                call errore('bad ncid: ', err)
            err = nf90mpi_iput_var(ncid, BAD_VARID, value, reqid(1))
            if (err .ne. NF90_ENOTVAR)  &
                call errore('bad var id: ', err)
            nels = 1
            do 3, j = 1, var_rank(i)
                nels = nels * var_shape(j,i)
3           continue
            allInExtRange = .true.
            do 4, j = 1, var_nels(i)
                err = index2indexes(j, var_rank(i), var_shape(1,i),  &
                                    index)
                if (err .ne. NF90_NOERR)  &
                    call error('error in index2indexes 1')
                val = hash_$1(var_type(i), var_rank(i), &
                              index, NFT_ITYPE($1))
                MAKE_TYPE2($1, VAR_ELEM($1, value, j), val)
                val = ARITH3($1, value, j)
                allInExtRange = allInExtRange .and. &
                    inRange3(val, var_type(i), NFT_ITYPE($1))
4           continue
            err = nf90mpi_iput_var(ncid, i, value, reqid(1), count=var_shape(:,i))
            if (err .eq. NF90_NOERR .or. err .eq. NF90_ERANGE) &
                err_w = nf90mpi_wait_all(ncid, 1, reqid, st)
                ! NF90_ERANGE is not a fatal error

            if (canConvert) then
                if (allInExtRange) then
                    if (err .ne. NF90_NOERR) &
                        call error(nf90mpi_strerror(err))
                else
                    if (err .ne. NF90_ERANGE .and. &
                            var_dimid(var_rank(i),i) .ne. RECDIM) &
                        call errore('Range error: ', err)
                endif
            else
                if (err .ne. NF90_ECHAR) &
                    call errore('wrong type: ', err)
            endif
1       continue

!       The preceeding has written nothing for record variables, now try
!       again with more than 0 records.

!       Write record number NRECS to force writing of preceding records.
!       Assumes variable cr is char vector with UNLIMITED dimension.

        err = nf90mpi_inq_varid(ncid, "cr", vid)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_inq_varid: ', err)
        index(1) = NRECS
        err = nf90mpi_iput_var(ncid, vid, 'x',reqid(1), index)
        if (err .ne. NF90_NOERR) then
            call errore('nf90mpi_iput_var: ', err)
        else
            err_w = nf90mpi_wait_all(ncid, 1, reqid, st)
        endif

        do 5 i = 1, numVars
!           Only test record variables here
            if (var_rank(i) .ge. 1 .and. &
                var_dimid(var_rank(i),i) .eq. RECDIM) then
                canConvert = (var_type(i) .eq. NF90_CHAR) .eqv. &
                         (NFT_ITYPE($1) .eq. NFT_TEXT)
                if (var_rank(i) .gt. MAX_RANK) &
                    stop 'var_rank(i) .gt. MAX_RANK'
                if (var_nels(i) .gt. MAX_NELS) &
                    stop 'var_nels(i) .gt. MAX_NELS'

                nels = 1
                do 6 j = 1, var_rank(i)
                    nels = nels * var_shape(j,i)
6               continue
                allInExtRange = .true.
                do 7, j = 1, INT(nels)
                    err = index2indexes(j, var_rank(i), var_shape(1,i),  &
                                    index)
                    if (err .ne. NF90_NOERR)  &
                        call error('error in index2indexes()')
                    val = hash_$1(var_type(i), var_rank(i), &
                                  index, NFT_ITYPE($1))
                    MAKE_TYPE2($1, VAR_ELEM($1, value, j), val)
                    val = ARITH3($1, value, j)
                    allInExtRange = allInExtRange .and. &
                        inRange3(val, var_type(i), NFT_ITYPE($1))
7               continue
                err = nf90mpi_iput_var(ncid, i, value, reqid(1), count=var_shape(:,i))
                if (err .eq. NF90_NOERR .or. err .eq. NF90_ERANGE) &
                    err_w = nf90mpi_wait_all(ncid, 1, reqid, st)
                    ! NF90_ERANGE is not a fatal error
                if (canConvert) then
                    if (allInExtRange) then
                        if (err .ne. NF90_NOERR) &
                            call error(nf90mpi_strerror(err))
                    else
                        if (err .ne. NF90_ERANGE) &
                            call errore('range error: ', err)
                    endif
                else
                    if (nels .gt. 0 .and. err .ne. NF90_ECHAR) &
                        call errore('wrong type: ', err)
                endif
            endif
5       continue
        err = nf90mpi_close(ncid)
        if (err .ne. NF90_NOERR)  &
            call errore('nf90mpi_close: ', err)

        call check_vars_$1(scratch)

        err = nf90mpi_delete(scratch, info)
        if (err .ne. NF90_NOERR) &
            call errorc('delete of scratch file failed: ',  &
                        scratch)
        end
])dnl


dnl TEST_NFMPI_IPUT_VARA(TYPE)
dnl
define([TEST_NFMPI_IPUT_VARA],dnl
[dnl
        subroutine test_nf90mpi_iput_vara_$1()
        use pnetcdf
        implicit        none
#include "tests.inc"
        integer roll, index2indexes
        double precision hash_$1
        logical inRange3

        integer ncid
        integer i
        integer j
        integer k
        integer d
        integer err, flags
        integer nslabs
        integer(kind=MPI_OFFSET_KIND) nels
        integer(kind=MPI_OFFSET_KIND) start(MAX_RANK)
        integer(kind=MPI_OFFSET_KIND) edge(MAX_RANK)
        integer(kind=MPI_OFFSET_KIND) mid(MAX_RANK)
        integer(kind=MPI_OFFSET_KIND) index(MAX_RANK)
        logical canConvert      !/* Both text or both numeric */
        logical allInExtRange   !/* all values within external range? */
        DATATYPE($1, value, (MAX_NELS))
        doubleprecision val
        integer ud_shift
        integer err_w, reqid(1), st(1)

        flags = IOR(NF90_CLOBBER, extra_flags)
        err = nf90mpi_create(comm, scratch, flags, info, &
                           ncid)
        if (err .ne. NF90_NOERR) then
            call errore('nf90mpi_create: ', err)
            return
        end if
        call def_dims(ncid)
        call def_vars(ncid)
        err = nf90mpi_enddef(ncid)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_enddef: ', err)

        do 1, i = 1, numVars
            canConvert = (var_type(i) .eq. NF90_CHAR) .eqv. &
                         (NFT_ITYPE($1) .eq. NFT_TEXT)
            if (.not.(var_rank(i) .le. MAX_RANK)) &
                stop 'assert(var_rank(i) .le. MAX_RANK)'
            if (.not.(var_nels(i) .le. MAX_NELS)) &
                stop 'assert(var_nels(i) .le. MAX_NELS)'
            do 2, j = 1, var_rank(i)
                start(j) = 1
                edge(j) = 1
2           continue
            err = nf90mpi_iput_var(BAD_ID, i, value,reqid(1), start, edge)
            if (err .ne. NF90_EBADID)  &
                call errore('bad ncid: ', err)
            err = nf90mpi_iput_var(ncid, BAD_VARID, &
                        value,reqid(1), start, edge)
            if (err .ne. NF90_ENOTVAR)  &
                call errore('bad var id: ', err)
            do 3, j = 1, var_rank(i)
                if (var_dimid(j,i) .ne. RECDIM) then    !/* skip record dim */
                    start(j) = var_shape(j,i) + 1
                    err = nf90mpi_iput_var(ncid, i, value,reqid(1), start, edge)
                    if (err .eq. NF90_NOERR .or. err .eq. NF90_ERANGE) &
                        err_w = nf90mpi_wait_all(ncid,1,reqid,st)
                        ! NF90_ERANGE is not a fatal error
                    if (.not. canConvert) then
                        if (err .ne. NF90_ECHAR) &
                            call errore('conversion: ', err)
                    else
                        if (err .ne. NF90_EINVALCOORDS) &
                            call errore('bad start: ', err)
                    endif
                    start(j) = 1
                    edge(j) = var_shape(j,i) + 1
                    err = nf90mpi_iput_var(ncid, i, value,reqid(1), start, edge)
                    if (err .eq. NF90_NOERR .or. err .eq. NF90_ERANGE) &
                        err_w = nf90mpi_wait_all(ncid,1,reqid,st)
                        ! NF90_ERANGE is not a fatal error
                    if (.not. canConvert) then
                        if (err .ne. NF90_ECHAR) &
                            call errore('conversion: ', err)
                    else
                        if (err .ne. NF90_EEDGE) &
                            call errore('bad edge: ', err)
                    endif
                    edge(j) = 1
                end if
3           continue

!           /* Check correct error returned even when nothing to put */
            do 20, j = 1, var_rank(i)
                  edge(j) = 0
20          continue
            err = nf90mpi_iput_var(BAD_ID, i, value,reqid(1), start, edge)
            if (err .ne. NF90_EBADID)  &
                call errore('bad ncid: ', err)
            err = nf90mpi_iput_var(ncid, BAD_VARID, value,reqid(1), start, edge)
            if (err .ne. NF90_ENOTVAR)  &
                call errore('bad var id: ', err)
            do 21, j = 1, var_rank(i)
                if (var_dimid(j,i) .gt. 1) then     ! skip record dim
                    start(j) = var_shape(j,i) + 1
                    err = nf90mpi_iput_var(ncid, i, value,reqid(1), start, edge)
                    if (err .eq. NF90_NOERR .or. err .eq. NF90_ERANGE) &
                        err_w = nf90mpi_wait_all(ncid, 1, reqid, st)
                        ! NF90_ERANGE is not a fatal error
                    if (.not. canConvert) then
                        if (err .ne. NF90_ECHAR) &
                            call errore('conversion: ', err)
                    else
                        if (err .ne. NF90_EINVALCOORDS) &
                            call errore('bad start: ', err)
                    endif
                    start(j) = 1
                endif
21          continue
            err = nf90mpi_iput_var(ncid, i, value, reqid(1), start, edge)
            if (err .eq. NF90_NOERR .or. err .eq. NF90_ERANGE) &
                err_w = nf90mpi_wait_all(ncid,1,reqid,st)
            if (canConvert) then
                if (st(1) .ne. 0)  &
                    call error(nf90mpi_strerror(st(1)))
            else
                if (err .ne. NF90_ECHAR) &
                    call errore('wrong type: ', err)
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
                    nels = nels * edge(j)
6               continue
                allInExtRange = .true.
                do 7, j = 1, INT(nels)
                    err = index2indexes(j, var_rank(i), edge, index)
                    if (err .ne. NF90_NOERR)  &
                        call error('error in index2indexes 1')
                    do 8, d = 1, var_rank(i)
                        index(d) = index(d) + start(d) - 1
8                   continue
                    val = hash_$1(var_type(i), var_rank(i), &
                                  index, NFT_ITYPE($1))
                    MAKE_TYPE2($1, VAR_ELEM($1, value, j), val)
                    val = ARITH3($1, value, j)
                    allInExtRange = allInExtRange .and. &
                        inRange3(val, var_type(i), NFT_ITYPE($1))
7               continue
                err = nf90mpi_iput_var(ncid, i, value,reqid(1), start, edge)
                if (err .eq. NF90_NOERR .or. err .eq. NF90_ERANGE) &
                    err_w = nf90mpi_wait_all(ncid,1,reqid,st)
                    ! NF90_ERANGE is not a fatal error
                if (canConvert) then
                    if (allInExtRange) then
                        if (st(1) .ne. 0)  &
                            call error(nf90mpi_strerror(st(1)))
                    else
                        if (err .ne. NF90_ERANGE) &
                            call errore('range error: ', err)
                    end if
                else
                    if (nels .gt. 0 .and. err .ne. NF90_ECHAR) &
                        call errore('wrong type: ', err)
                end if
5           continue
1       continue

        err = nf90mpi_close(ncid)
        if (err .ne. NF90_NOERR)  &
            call errore('nf90mpi_close: ', err)

        call check_vars_$1(scratch)

        err = nf90mpi_delete(scratch, info)
        if (err .ne. NF90_NOERR) &
            call errorc('delete of scratch file failed: ',  &
                scratch)
        end
])dnl


dnl TEST_NFMPI_IPUT_VARS(TYPE)
dnl
define([TEST_NFMPI_IPUT_VARS],dnl
[dnl
        subroutine test_nf90mpi_iput_vars_$1()
        use pnetcdf
        implicit        none
#include "tests.inc"
        integer roll, index2indexes
        double precision hash_$1
        logical inRange3

        integer ncid
        integer d
        integer i
        integer j
        integer k
        integer m
        integer err, flags
        integer(kind=MPI_OFFSET_KIND) nels
        integer nslabs
        integer(kind=MPI_OFFSET_KIND) nstarts        !/* number of different starts */
        integer(kind=MPI_OFFSET_KIND) start(MAX_RANK)
        integer(kind=MPI_OFFSET_KIND) edge(MAX_RANK)
        integer(kind=MPI_OFFSET_KIND) index(MAX_RANK)
        integer(kind=MPI_OFFSET_KIND) index2(MAX_RANK)
        integer(kind=MPI_OFFSET_KIND) mid(MAX_RANK)
        integer(kind=MPI_OFFSET_KIND) count(MAX_RANK)
        integer(kind=MPI_OFFSET_KIND) sstride(MAX_RANK)
        integer(kind=MPI_OFFSET_KIND) stride(MAX_RANK)
        logical canConvert      !/* Both text or both numeric */
        logical allInExtRange   !/* all values within external range? */
        DATATYPE($1, value, (MAX_NELS))
        doubleprecision val
        integer ud_shift
        integer err_w, reqid(1), st(1)

        flags = IOR(NF90_CLOBBER, extra_flags)
        err = nf90mpi_create(comm, scratch, flags, info, &
                           ncid)
        if (err .ne. NF90_NOERR) then
            call errore('nf90mpi_create: ', err)
            return
        end if
        call def_dims(ncid)
        call def_vars(ncid)
        err = nf90mpi_enddef(ncid)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_enddef: ', err)

        do 1, i = 1, numVars
            canConvert = (var_type(i) .eq. NF90_CHAR) .eqv. &
                         (NFT_ITYPE($1) .eq. NFT_TEXT)
            if (.not.(var_rank(i) .le. MAX_RANK)) &
                stop 'assert(var_rank(i) .le. MAX_RANK)'
            if (.not.(var_nels(i) .le. MAX_NELS)) &
                stop 'assert(var_nels(i) .le. MAX_NELS)'
            do 2, j = 1, var_rank(i)
                start(j) = 1
                edge(j) = 1
                stride(j) = 1
2           continue
            err = nf90mpi_iput_var(BAD_ID, i, value,reqid(1), start, edge, stride)
            if (err .ne. NF90_EBADID)  &
                call errore('bad ncid: ', err)
            err = nf90mpi_iput_var(ncid, BAD_VARID, value,reqid(1), start, edge, stride)
            if (err .ne. NF90_ENOTVAR)  &
                call errore('bad var id: ', err)
            do 3, j = 1, var_rank(i)
                if (var_dimid(j,i) .ne. RECDIM) then    ! skip record dim
                    start(j) = var_shape(j,i) + 1
                    err = nf90mpi_iput_var(ncid, i, value,reqid(1), start, edge, stride)
                    if (err .eq. NF90_NOERR .or. err .eq. NF90_ERANGE) &
                        err_w = nf90mpi_wait_all(ncid,1,reqid,st)
                    if (.not. canConvert) then
                        if (err .ne. NF90_ECHAR) &
                            call errore('conversion: ', err)
                    else
                        if (err .ne. NF90_EINVALCOORDS) &
                                call errore('bad start: ', err)
                    endif
                    start(j) = 1
                    edge(j) = var_shape(j,i) + 1
                    err = nf90mpi_iput_var(ncid, i, value,reqid(1), start, edge, stride)
                    if (err .eq. NF90_NOERR .or. err .eq. NF90_ERANGE) &
                        err_w = nf90mpi_wait_all(ncid,1,reqid,st)
                    if (.not. canConvert) then
                        if (err .ne. NF90_ECHAR) &
                            call errore('conversion: ', err)
                    else
                        if (err .ne. NF90_EEDGE) &
                            call errore('bad edge: ', err)
                    endif
                    edge(j) = 1
                    stride(j) = 0
                    err = nf90mpi_iput_var(ncid, i, value,reqid(1), start, edge, stride)
                    if (err .eq. NF90_NOERR .or. err .eq. NF90_ERANGE) &
                        err_w = nf90mpi_wait_all(ncid,1,reqid,st)
                    if (.not. canConvert) then
                        if (err .ne. NF90_ECHAR) &
                            call errore('conversion: ', err)
                    else
                        if (err .ne. NF90_ESTRIDE) &
                            call errore('bad stride: ', err)
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
                    nstarts = nstarts * stride(j)
6               continue
                do 7, m = 1, INT(nstarts)
                    err = index2indexes(m, var_rank(i), sstride, index)
                    if (err .ne. NF90_NOERR) &
                        call error('error in index2indexes')
                    nels = 1
                    do 8, j = 1, var_rank(i)
                        count(j) = 1 + (edge(j) - index(j)) / stride(j)
                        nels = nels * count(j)
                        index(j) = index(j) + start(j) - 1
8                   continue
                    !/* Random choice of forward or backward */
! TODO
!                   if ( roll(2) ) {
!                       for (j = 1 j .lt. var_rank(i) j++) {
!                           index(j) += (count(j) - 1) * stride(j)
!                           stride(j) = -stride(j)
!                       }
!                   }
!
                    allInExtRange = .true.
                    do 9, j = 1, INT(nels)
                        err = index2indexes(j, var_rank(i), count,  &
                                            index2)
                        if (err .ne. NF90_NOERR) &
                            call error('error in index2indexes')
                        do 10, d = 1, var_rank(i)
                            index2(d) = index(d) +  &
                                        (index2(d)-1) * stride(d)
10                      continue
                        val = hash_$1(var_type(i), var_rank(i), &
                                      index2, NFT_ITYPE($1))
                        MAKE_TYPE2($1, VAR_ELEM($1, value, j), val)
                        val = ARITH3($1, value, j)
                        allInExtRange = allInExtRange .and. &
                            inRange3(val, var_type(i),  &
                                     NFT_ITYPE($1))
9                   continue
                    err = nf90mpi_iput_var(ncid, i, value,reqid(1), index, count, stride)
                    if (err .eq. NF90_NOERR .or. err .eq. NF90_ERANGE) &
                        err_w = nf90mpi_wait_all(ncid,1,reqid,st)
                    if (canConvert) then
                        if (allInExtRange) then
                            if (st(1) .ne. 0)  &
                                call error(nf90mpi_strerror(st(1)))
                        else
                            if (err .ne. NF90_ERANGE) &
                                call errore('range error: ', err)
                        end if
                    else
                        if (nels .gt. 0 .and. err .ne. NF90_ECHAR) &
                            call errore('wrong type: ', err)
                    end if
7               continue
5           continue
1       continue

        err = nf90mpi_close(ncid)
        if (err .ne. NF90_NOERR)  &
            call errore('nf90mpi_close: ', err)

        call check_vars_$1(scratch)

        err = nf90mpi_delete(scratch, info)
        if (err .ne. NF90_NOERR) &
            call errorc('delete of scratch file failed:',  &
                scratch)
        end
])dnl


dnl  since parallel-netcdf doesn't have varm type, we haven't completed the
dnl  parallel-netcdf-ification of these routines
dnl TEST_NFMPI_IPUT_VARM(TYPE)
dnl
define([TEST_NFMPI_IPUT_VARM],dnl
[dnl
        subroutine test_nf90mpi_iput_varm_$1()
        use pnetcdf
        implicit        none
#include "tests.inc"
        integer roll, index2indexes
        double precision hash_$1
        logical inRange3

        integer ncid
        integer d
        integer i
        integer j
        integer k
        integer m
        integer err, flags
        integer(kind=MPI_OFFSET_KIND) nels
        integer nslabs
        integer(kind=MPI_OFFSET_KIND) nstarts        !/* number of different starts */
        integer(kind=MPI_OFFSET_KIND) start(MAX_RANK)
        integer(kind=MPI_OFFSET_KIND) edge(MAX_RANK)
        integer(kind=MPI_OFFSET_KIND) index(MAX_RANK)
        integer(kind=MPI_OFFSET_KIND) index2(MAX_RANK)
        integer(kind=MPI_OFFSET_KIND) mid(MAX_RANK)
        integer(kind=MPI_OFFSET_KIND) count(MAX_RANK)
        integer(kind=MPI_OFFSET_KIND) sstride(MAX_RANK)
        integer(kind=MPI_OFFSET_KIND) stride(MAX_RANK)
        integer(kind=MPI_OFFSET_KIND) imap(MAX_RANK)
        logical canConvert      !/* Both text or both numeric */
        logical allInExtRange   !/* all values within external range? */
        DATATYPE($1, value, (MAX_NELS))
        doubleprecision val
        integer ud_shift
        integer err_w, reqid(1), st(1)

        flags = IOR(NF90_CLOBBER, extra_flags)
        err = nf90mpi_create(comm, scratch, flags, info, &
                           ncid)
        if (err .ne. NF90_NOERR) then
            call errore('nf90mpi_create: ', err)
            return
        end if
        call def_dims(ncid)
        call def_vars(ncid)
        err = nf90mpi_enddef(ncid)
        if (err .ne. NF90_NOERR) &
            call errore('nf90mpi_enddef: ', err)

        do 1, i = 1, numVars
            canConvert = (var_type(i) .eq. NF90_CHAR) .eqv. &
                         (NFT_ITYPE($1) .eq. NFT_TEXT)
            if (.not.(var_rank(i) .le. MAX_RANK)) &
                stop 'assert(var_rank(i) .le. MAX_RANK)'
            if (.not.(var_nels(i) .le. MAX_NELS)) &
                stop 'assert(var_nels(i) .le. MAX_NELS)'
            do 2, j = 1, var_rank(i)
                start(j) = 1
                edge(j) = 1
                stride(j) = 1
                imap(j) = 1
2           continue
            err = nf90mpi_iput_var(BAD_ID, i, value,reqid(1), start, edge, stride, imap)
            if (err .ne. NF90_EBADID)  &
                call errore('bad ncid: ', err)
            err = nf90mpi_iput_var(ncid, BAD_VARID, &
                                 value,reqid(1), start, edge, stride, imap)
            if (err .ne. NF90_ENOTVAR)  &
                call errore('bad var id: ', err)
            do 3, j = 1, var_rank(i)
                if (var_dimid(j,i) .ne. RECDIM) then    !/* skip record dim */
                    start(j) = var_shape(j,i) + 1
                    err = nf90mpi_iput_var(ncid, i, &
                                         value,reqid(1), start, edge, stride, imap)
                    if (err .eq. NF90_NOERR .or. err .eq. NF90_ERANGE) &
                        err_w = nf90mpi_wait_all(ncid,1,reqid,st)
                    if (.not. canConvert) then
                        if (err .ne. NF90_ECHAR) &
                            call errore('conversion: ', err)
                    else
                        if (err .ne. NF90_EINVALCOORDS) &
                            call errore('bad start: ', err)
                    endif
                    start(j) = 1
                    edge(j) = var_shape(j,i) + 1
                    err = nf90mpi_iput_var(ncid, i, &
                                         value,reqid(1), start, edge, stride, imap)
                    if (err .eq. NF90_NOERR .or. err .eq. NF90_ERANGE) &
                        err_w = nf90mpi_wait_all(ncid,1,reqid,st)
                    if (.not. canConvert) then
                        if (err .ne. NF90_ECHAR) &
                            call errore('conversion: ', err)
                    else
                        if (err .ne. NF90_EEDGE) &
                            call errore('bad edge: ', err)
                    endif
                    edge(j) = 1
                    stride(j) = 0
                    err = nf90mpi_iput_var(ncid, i, &
                                         value,reqid(1), start, edge, stride, imap)
                    if (err .eq. NF90_NOERR .or. err .eq. NF90_ERANGE) &
                        err_w = nf90mpi_wait_all(ncid,1,reqid,st)
                    if (.not. canConvert) then
                        if (err .ne. NF90_ECHAR) &
                            call errore('conversion: ', err)
                    else
                        if (err .ne. NF90_ESTRIDE) &
                            call errore('bad stride: ', err)
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
                    nstarts = nstarts * stride(j)
6               continue
                do 7, m = 1, INT(nstarts)
                    err = index2indexes(m, var_rank(i), sstride, index)
                    if (err .ne. NF90_NOERR) &
                        call error('error in index2indexes')
                    nels = 1
                    do 8, j = 1, var_rank(i)
                        count(j) = 1 + (edge(j) - index(j)) / stride(j)
                        nels = nels * count(j)
                        index(j) = index(j) + start(j) - 1
8                   continue
                    !/* Random choice of forward or backward */
! TODO
!                   if ( roll(2) ) then
!                       do 9, j = 1, var_rank(i)
!                           index(j) = index(j) +  &
!                               (count(j) - 1) * stride(j)
!                           stride(j) = -stride(j)
!9                      continue
!                   end if
!
                    if (var_rank(i) .gt. 0) then
                        imap(1) = 1
                        do 10, j = 2, var_rank(i)
                            imap(j) = imap(j-1) * count(j-1)
10                      continue
                    end if
                    allInExtRange = .true.
                    do 11 j = 1, INT(nels)
                        err = index2indexes(j, var_rank(i), count,  &
                                            index2)
                        if (err .ne. NF90_NOERR) &
                            call error('error in index2indexes')
                        do 12, d = 1, var_rank(i)
                            index2(d) = index(d) +  &
                                (index2(d)-1) * stride(d)
12                      continue
                        val = hash_$1(var_type(i),var_rank(i), &
                                      index2, NFT_ITYPE($1))
                        MAKE_TYPE2($1, VAR_ELEM($1, value, j), val)
                        val = ARITH3($1, value, j)
                        allInExtRange = allInExtRange .and. &
                            inRange3(val, var_type(i),  &
                                     NFT_ITYPE($1))
11                  continue
                    err = nf90mpi_iput_var(ncid,i,&
                                         value,reqid(1), index, count, stride, imap)
                    if (err .eq. NF90_NOERR .or. err .eq. NF90_ERANGE) &
                        err_w = nf90mpi_wait_all(ncid,1,reqid,st)
                    if (canConvert) then
                        if (allInExtRange) then
                            if (st(1) .ne. 0) &
                                call error(nf90mpi_strerror(st(1)))
                        else
                            if (err .ne. NF90_ERANGE) &
                                call errore('range error: ', err)
                        end if
                    else
                        if (nels .gt. 0 .and. err .ne. NF90_ECHAR) &
                            call errore('wrong type: ', err)
                    end if
7               continue
5           continue
1       continue

        err = nf90mpi_close(ncid)
        if (err .ne. NF90_NOERR)  &
            call errore('nf90mpi_close: ', err)

        call check_vars_$1(scratch)

        err = nf90mpi_delete(scratch, info)
        if (err .ne. NF90_NOERR) &
            call errorc('delete of scratch file failed:',  &
                scratch)
        end
])dnl


divert(0)dnl
dnl If you see this line, you can ignore the next one.
! Do not edit this file. It is produced from the corresponding .m4 source */

!
!  Copyright (C) 2013, Northwestern University and Argonne National Laboratory
!  See COPYRIGHT notice in top-level directory.
!

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

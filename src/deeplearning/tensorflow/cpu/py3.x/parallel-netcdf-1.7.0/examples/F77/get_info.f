!
!  Copyright (C) 2012, Northwestern University and Argonne National
!  Laboratory
!  See COPYRIGHT notice in top-level directory.
!
! $Id: get_info.f 2224 2015-12-16 06:10:36Z wkliao $

!
!    To compile:
!        mpif77 -O2 get_info.f -o get_info -lpnetcdf
!    To run:
!        mpiexec -n 4 ./get_info /pvfs2/wkliao/testfile.nc
!
!    prints all MPI-IO hints used
!
!    Example standard output:
!
!   MPI File Info: nkeys =          18
!   MPI File Info: [ 0] key =            cb_buffer_size, value =16777216
!   MPI File Info: [ 1] key =             romio_cb_read, value =automatic
!   MPI File Info: [ 2] key =            romio_cb_write, value =automatic
!   MPI File Info: [ 3] key =                  cb_nodes, value =1
!   MPI File Info: [ 4] key =         romio_no_indep_rw, value =false
!   MPI File Info: [ 5] key =              romio_cb_pfr, value =disable
!   MPI File Info: [ 6] key =         romio_cb_fr_types, value =aar
!   MPI File Info: [ 7] key =     romio_cb_fr_alignment, value =1
!   MPI File Info: [ 8] key =     romio_cb_ds_threshold, value =0
!   MPI File Info: [ 9] key =         romio_cb_alltoall, value =automatic
!   MPI File Info: [10] key =        ind_rd_buffer_size, value =4194304
!   MPI File Info: [11] key =        ind_wr_buffer_size, value =524288
!   MPI File Info: [12] key =             romio_ds_read, value =automatic
!   MPI File Info: [13] key =            romio_ds_write, value =automatic
!   MPI File Info: [14] key =            cb_config_list, value =*:1
!   MPI File Info: [15] key =      nc_header_align_size, value =0
!   MPI File Info: [16] key =         nc_var_align_size, value =0
!   MPI File Info: [17] key = nc_header_read_chunk_size, value =0


        program main
        implicit none
        include "mpif.h"
        include "pnetcdf.inc"

        character*256 filename, cmd
        integer ncid, rank, info, omode, err, ierr, get_args
        integer*8 malloc_size, sum_size
        logical verbose
        integer dummy

        call MPI_Init(err)
        call MPI_Comm_rank (MPI_COMM_WORLD, rank, err)

        if (rank .EQ. 0) then
            verbose = .TRUE.
            filename = "testfile.nc"
            ierr = get_args(2, cmd, filename, verbose, dummy)
        endif
        call MPI_Bcast(ierr, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, err)
        if (ierr .EQ. 0) goto 999

        call MPI_Bcast(verbose, 1, MPI_LOGICAL, 0, MPI_COMM_WORLD,
     +                 err)
        call MPI_Bcast(filename, 256, MPI_CHARACTER, 0,
     +                 MPI_COMM_WORLD, err)

        omode = NF_NOWRITE + NF_64BIT_OFFSET
        err = nfmpi_open(MPI_COMM_WORLD, filename, omode,
     +                    MPI_INFO_NULL, ncid)
        if (err .ne. NF_NOERR) call handle_err('nfmpi_open',err)


        err = nfmpi_inq_file_info(ncid, info)
        if (err .ne. NF_NOERR) call handle_err('nfmpi_inq_file_info',
     +                                          err)

        err = nfmpi_close(ncid)
        if (err .ne. NF_NOERR) call handle_err('nfmpi_close',err)

        if (rank .EQ. 0 .AND. verbose) call print_info(info)
        call MPI_Info_free(info, err)

        ! check if there is any PnetCDF internal malloc residue
 998    format(A,I13,A)
        err = nfmpi_inq_malloc_size(malloc_size)
        if (err .EQ. NF_NOERR) then
            call MPI_Reduce(malloc_size, sum_size, 1, MPI_OFFSET, 
     +                      MPI_SUM, 0, MPI_COMM_WORLD, err)
            if (rank .EQ. 0 .AND. sum_size .GT. 0)
     +          print 998,
     +          'heap memory allocated by PnetCDF internally has ',
     +          sum_size/1048576, ' MiB yet to be freed'
        endif

 999    call MPI_Finalize(err)
        ! call EXIT(0) ! EXIT() is a GNU extension

        end ! program main

        subroutine print_info(info_used)
            implicit none
            include "mpif.h"
            include "pnetcdf.inc"

            integer info_used

            ! local variables
            character*(MPI_MAX_INFO_VAL) key, value
            integer nkeys, i, err
            logical flag

            call MPI_Info_get_nkeys(info_used, nkeys, err)
            print *, 'MPI File Info: nkeys =', nkeys
            do i=0, nkeys-1
                call MPI_Info_get_nthkey(info_used, i, key, err)
                call MPI_Info_get(info_used, key, MPI_MAX_INFO_VAL,
     +                            value, flag, err)
 123            format('MPI File Info: [',I2,'] key = ',A25,
     +                 ', value =',A)
                print 123, i, key, value
            enddo
            print *

            return
        end ! subroutine print_info

        subroutine handle_err(err_msg, errcode)
            implicit none
            include "mpif.h"
            include "pnetcdf.inc"

            character*(*) err_msg
            integer       errcode

            ! local variables
            integer err 

            print *, 'Error: ',err_msg//' '//nfmpi_strerror(errcode)
            call MPI_Abort(MPI_COMM_WORLD, -1, err)
            return
        end ! subroutine handle_err


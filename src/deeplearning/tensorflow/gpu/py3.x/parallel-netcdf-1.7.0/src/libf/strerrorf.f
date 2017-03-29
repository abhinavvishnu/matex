      character *80 function nfmpi_strerror( err )
      integer err, ierr
      character *(80) tmpstr

      integer nfmpi_xstrerror
      external nfmpi_xstrerror

      ierr = nfmpi_xstrerror( err, tmpstr )
      nfmpi_strerror = tmpstr
      end

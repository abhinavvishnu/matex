      character *80 function nfmpi_inq_libvers()
      integer ierr
      character *(80) tmpstr

      integer nfmpi_xinq_libvers
      external nfmpi_xinq_libvers
      
      ierr = nfmpi_xinq_libvers(tmpstr)
      nfmpi_inq_libvers = tmpstr
      end

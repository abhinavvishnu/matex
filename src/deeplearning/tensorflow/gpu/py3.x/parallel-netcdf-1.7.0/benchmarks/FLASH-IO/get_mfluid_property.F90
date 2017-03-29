subroutine get_mfluid_property(property, value)

! stub function to fill the need for get_mfluid_property from the
! FLASH I/O routines that are used in this benchmark.  All they need
! is a routine to return the names of the isotopes, so do that here

#include "common.fh"

  character (len=*) :: property
  character (len=4) :: value(ionmax)

  if (ionmax == 2) then
     value(1) = "f1"
     value(2) = "f2"
  else if (ionmax == 13) then
     value(1) = "f1"
     value(2) = "f2"
     value(3) = "f3"
     value(4) = "f4"
     value(5) = "f5"
     value(6) = "f6"
     value(7) = "f7"
     value(8) = "f8"
     value(9) = "f9"
     value(10) = "f10"
     value(11) = "f11"
     value(12) = "f12"
     value(13) = "f13"
  endif

  return
end subroutine get_mfluid_property



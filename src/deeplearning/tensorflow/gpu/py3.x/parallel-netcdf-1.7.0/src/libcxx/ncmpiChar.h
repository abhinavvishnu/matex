#include "ncmpiType.h"

#ifndef NcmpiCharClass
#define NcmpiCharClass

namespace PnetCDF
{
  
  /*! Class represents a netCDF atomic Char type. */
  class NcmpiChar : public NcmpiType
  {
  public: 
    
    /*! equivalence operator */
    bool operator==(const NcmpiChar & rhs);
    
    ~NcmpiChar();
    
    /*! Constructor */
    NcmpiChar();
  };

  /*! A global instance  of the NcmpiChar class within the netCDF namespace. */
  extern NcmpiChar ncmpiChar;

}
#endif

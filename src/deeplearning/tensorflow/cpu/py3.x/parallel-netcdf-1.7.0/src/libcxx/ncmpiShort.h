#include "ncmpiType.h"

#ifndef NcmpiShortClass
#define NcmpiShortClass

namespace PnetCDF
{
  
  /*! Class represents a netCDF atomic Short type. */
  class NcmpiShort : public NcmpiType
  {
  public: 
    
    /*! equivalence operator */
    bool operator==(const NcmpiShort & rhs);
    
    /*! destructor */
    ~NcmpiShort();
    
    /*! Constructor */
    NcmpiShort();
  };

  /*! A global instance  of the NcmpiShort class within the netCDF namespace. */
  extern NcmpiShort ncmpiShort;

}
#endif

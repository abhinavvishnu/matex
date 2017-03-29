#include "ncmpiType.h"

#ifndef NcmpiInt64Class
#define NcmpiInt64Class

namespace PnetCDF
{
  
  /*! Class represents a netCDF atomic Int64 type. */
  class NcmpiInt64 : public NcmpiType
  {
  public: 
    
    /*! equivalence operator */
    bool operator==(const NcmpiInt64 & rhs);
    
    /*!  destructor */
    ~NcmpiInt64();
    
    /*! Constructor */
    NcmpiInt64();
  };

  /*! A global instance  of the NcmpiInt64 class within the netCDF namespace. */
  extern NcmpiInt64 ncmpiInt64;

}
#endif

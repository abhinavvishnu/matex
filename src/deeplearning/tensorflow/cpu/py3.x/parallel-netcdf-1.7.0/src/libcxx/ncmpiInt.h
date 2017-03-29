#include "ncmpiType.h"

#ifndef NcmpiIntClass
#define NcmpiIntClass

namespace PnetCDF
{
  
  /*! Class represents a netCDF atomic Int type. */
  class NcmpiInt : public NcmpiType
  {
  public: 
    
    /*! equivalence operator */
    bool operator==(const NcmpiInt & rhs);
    
    /*!  destructor */
    ~NcmpiInt();
    
    /*! Constructor */
    NcmpiInt();
  };

  /*! A global instance  of the NcmpiInt class within the netCDF namespace. */
  extern NcmpiInt ncmpiInt;

}
#endif

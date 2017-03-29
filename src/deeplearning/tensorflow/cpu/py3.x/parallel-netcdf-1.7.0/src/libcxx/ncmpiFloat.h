#include "ncmpiType.h"

#ifndef NcmpiFloatClass
#define NcmpiFloatClass

namespace PnetCDF
{
  
  /*! Class represents a netCDF atomic Float type. */
  class NcmpiFloat : public NcmpiType
  {
  public: 
    
    /*! equivalence operator */
    bool operator==(const NcmpiFloat & rhs);
    
    /*!  destructor */
    ~NcmpiFloat();
    
    /*! Constructor */
    NcmpiFloat();
  };

  /*! A global instance  of the NcmpiFloat class within the netCDF namespace. */
  extern NcmpiFloat ncmpiFloat;

}
#endif

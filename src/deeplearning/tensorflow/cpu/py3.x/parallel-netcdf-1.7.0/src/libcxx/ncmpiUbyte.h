#include "ncmpiType.h"

#ifndef NcmpiUbyteClass
#define NcmpiUbyteClass

namespace PnetCDF
{
  
  /*! Class represents a netCDF atomic Ubyte type. */
  class NcmpiUbyte : public NcmpiType
  {
  public: 
    
    /*! equivalence operator */
    bool operator==(const NcmpiUbyte & rhs);
    
    /*! destructor */
    ~NcmpiUbyte();
    
    /*! Constructor */
    NcmpiUbyte();
  };

  /*! A global instance  of the NcmpiUbyte class within the netCDF namespace. */
  extern NcmpiUbyte ncmpiUbyte;

}
#endif

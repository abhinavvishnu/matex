#include "ncmpiType.h"

#ifndef NcmpiByteClass
#define NcmpiByteClass

namespace PnetCDF
{
  
  /*! Class represents a netCDF atomic Byte type. */
  class NcmpiByte : public NcmpiType
  {
  public: 
    
    /*! equivalence operator */
    bool operator==(const NcmpiByte & rhs);

    /*! storage size */
    int sizeoff();

    ~NcmpiByte();
    
    /*! Constructor */
    NcmpiByte();
  };

  /*! A global instance  of the NcmpiByte class within the netCDF namespace. */
  extern NcmpiByte ncmpiByte;

}
#endif

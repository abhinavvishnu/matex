#include "ncmpiVlenType.h"
#include "ncmpiGroup.h"
#include "ncmpiCheck.h"
#include "ncmpiException.h"
#include "ncmpiByte.h"
#include "ncmpiUbyte.h"
#include "ncmpiChar.h"
#include "ncmpiShort.h"
#include "ncmpiUshort.h"
#include "ncmpiInt.h"
#include "ncmpiUint.h"
#include "ncmpiInt64.h"
#include "ncmpiUint64.h"
#include "ncmpiFloat.h"
#include "ncmpiDouble.h"
#include <pnetcdf.h>
using namespace std;
using namespace PnetCDF;
using namespace PnetCDF::exceptions;

// Class represents a netCDF variable.
using namespace PnetCDF;

// assignment operator
NcmpiVlenType& NcmpiVlenType::operator=(const NcmpiVlenType& rhs)
{
  NcmpiType::operator=(rhs);    // assign base class parts
  return *this;
}

// assignment operator
NcmpiVlenType& NcmpiVlenType::operator=(const NcmpiType& rhs)
{
  if (&rhs != this) {
    // check the rhs is the base of an Opaque type
    if(getTypeClass() != NC_VLEN) 	throw NcmpiException("The NcmpiType object must be the base of an Vlen type.",__FILE__,__LINE__);
    // assign base class parts
    NcmpiType::operator=(rhs);
  }
  return *this;
}

// The copy constructor.
NcmpiVlenType::NcmpiVlenType(const NcmpiVlenType& rhs):   
  NcmpiType(rhs)
{
}


// Constructor generates a null object.
NcmpiVlenType::NcmpiVlenType() :
  NcmpiType()   // invoke base class constructor
{}

// constructor
NcmpiVlenType::NcmpiVlenType(const NcmpiGroup& grp, const string& name) :
  NcmpiType(grp,name)
{}
  
// constructor
NcmpiVlenType::NcmpiVlenType(const NcmpiType& ncmpiType): 
  NcmpiType(ncmpiType)
{
  // check the nctype object is the base of a Vlen type
  if(getTypeClass() != NC_VLEN) throw NcmpiException("The NcmpiType object must be the base of a Vlen type.",__FILE__,__LINE__);
}

// Returns the base type.
NcmpiType NcmpiVlenType::getBaseType() const
{
  char charName[NC_MAX_NAME+1];
  nc_type base_nc_typep;
  MPI_Offset datum_sizep;
  ncmpiCheck(ncmpi_inq_vlen(groupId,myId,charName,&datum_sizep,&base_nc_typep),__FILE__,__LINE__);
  switch (base_nc_typep) {
  case NC_BYTE    : return ncmpiByte;
  case NC_UBYTE   : return ncmpiUbyte;
  case NC_CHAR    : return ncmpiChar;
  case NC_SHORT   : return ncmpiShort;
  case NC_USHORT  : return ncmpiUshort;
  case NC_INT     : return ncmpiInt;
  case NC_UINT    : return ncmpiUint;  
  case NC_INT64   : return ncmpiInt64; 
  case NC_UINT64  : return ncmpiUint64;
  case NC_FLOAT   : return ncmpiFloat;
  case NC_DOUBLE  : return ncmpiDouble;
  default:  
    // this is a user defined type
    return NcmpiType(getParentGroup(),base_nc_typep);
  }
}

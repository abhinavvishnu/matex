#include "ncmpiEnumType.h"
#include "ncmpiGroup.h"
#include "ncmpiCheck.h"
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
#include "ncmpiException.h"

using namespace std;
using namespace PnetCDF;
using namespace PnetCDF::exceptions;

// Class represents a netCDF variable.

// assignment operator
NcmpiEnumType& NcmpiEnumType::operator=(const NcmpiEnumType& rhs)
{
  NcmpiType::operator=(rhs);    // assign base class parts
  return *this;
}

// assignment operator
NcmpiEnumType& NcmpiEnumType::operator=(const NcmpiType& rhs)
{
  if (&rhs != this) {
    // check the rhs is the base of an Enum type
    if(getTypeClass() != NC_ENUM) throw NcmpiException("The NcmpiType object must be the base of an Enum type.",__FILE__,__LINE__);
    // assign base class parts
    NcmpiType::operator=(rhs);
  }
  return *this;
}

// The copy constructor.
NcmpiEnumType::NcmpiEnumType(const NcmpiEnumType& rhs): 
  NcmpiType(rhs)
{
}


// Constructor generates a null object.
NcmpiEnumType::NcmpiEnumType() :
  NcmpiType()   // invoke base class constructor
{}
  
// constructor
NcmpiEnumType::NcmpiEnumType(const NcmpiGroup& grp, const string& name):
  NcmpiType(grp,name)
{}
  

// constructor
NcmpiEnumType::NcmpiEnumType(const NcmpiType& ncmpiType): 
  NcmpiType(ncmpiType)
{
  // check the nctype object is the base of an Enum type
  if(getTypeClass() != NC_ENUM) throw NcmpiException("The NcmpiType object must be the base of an Enum type.",__FILE__,__LINE__);
}

// Returns the base type.
NcmpiType NcmpiEnumType::getBaseType() const
{
  char charName[NC_MAX_NAME+1];
  nc_type base_nc_typep;
  MPI_Offset *base_sizep=NULL;
  MPI_Offset *num_membersp=NULL;
  ncmpiCheck(ncmpi_inq_enum(groupId,myId,charName,&base_nc_typep,base_sizep,num_membersp),__FILE__,__LINE__);
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

  
// Returns number of members in this NcmpiEnumType object.
MPI_Offset   NcmpiEnumType::getMemberCount() const{
  char charName[NC_MAX_NAME+1];
  nc_type* base_nc_typep=NULL;
  MPI_Offset* base_sizep=NULL;
  MPI_Offset num_membersp;
  ncmpiCheck(ncmpi_inq_enum(groupId,myId,charName,base_nc_typep,base_sizep,&num_membersp),__FILE__,__LINE__);
  return num_membersp;
};
  
// Returns the member name for the given zero-based index.
string NcmpiEnumType::getMemberNameFromIndex(int index) const{
  void* value=NULL;
  char charName[NC_MAX_NAME+1];
  ncmpiCheck(ncmpi_inq_enum_member(groupId,myId,index,charName,value),__FILE__,__LINE__);
  return static_cast<string> (charName);
};

#include "ncmpiOpaqueType.h"
#include "ncmpiGroup.h"
#include "ncmpiCheck.h"
#include "ncmpiException.h"
#include <pnetcdf.h>
using namespace std;
using namespace PnetCDF;
using namespace PnetCDF::exceptions;

// Class represents a netCDF variable.
using namespace PnetCDF;
  
// assignment operator
NcmpiOpaqueType& NcmpiOpaqueType::operator=(const NcmpiOpaqueType& rhs)
{
  // assign base class parts
  NcmpiType::operator=(rhs);    
  return *this;
}
  
// assignment operator
NcmpiOpaqueType& NcmpiOpaqueType::operator=(const NcmpiType& rhs)
{
  if (&rhs != this) {
    // check the rhs is the base of an Opaque type
    if(getTypeClass() != NC_OPAQUE) 	throw NcmpiException("The NcmpiType object must be the base of an Opaque type.",__FILE__,__LINE__);
    // assign base class parts
    NcmpiType::operator=(rhs);
  }
  return *this;
}

// The copy constructor.
NcmpiOpaqueType::NcmpiOpaqueType(const NcmpiOpaqueType& rhs): 
  NcmpiType(rhs)
{
}


// Constructor generates a null object.
NcmpiOpaqueType::NcmpiOpaqueType() :
  NcmpiType()   // invoke base class constructor
{}


// constructor
NcmpiOpaqueType::NcmpiOpaqueType(const NcmpiGroup& grp, const string& name) :
  NcmpiType(grp,name)
{}
  
  
// constructor
NcmpiOpaqueType::NcmpiOpaqueType(const NcmpiType& ncmpiType) :
  NcmpiType(ncmpiType)
{
  // check the nctype object is the base of a Opaque type
  if(getTypeClass() != NC_OPAQUE) 	throw NcmpiException("The NcmpiType object must be the base of an Opaque type.",__FILE__,__LINE__);
}
  
// Returns the size of the opaque type in bytes.
MPI_Offset  NcmpiOpaqueType::getTypeSize() const
{
  char* charName;
  charName=NULL;
  MPI_Offset sizep;
  ncmpiCheck(ncmpi_inq_opaque(groupId,myId,charName,&sizep),__FILE__,__LINE__);
  return sizep;
}

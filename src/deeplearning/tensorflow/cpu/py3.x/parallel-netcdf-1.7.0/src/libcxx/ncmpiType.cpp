#include <string>
#include "ncmpiType.h"
#include "ncmpiGroup.h"
#include "ncmpiCheck.h"
using namespace std;


namespace PnetCDF {
  //  Global comparator operator ==============
  // comparator operator 
  bool operator<(const NcmpiType& lhs,const NcmpiType& rhs)
  {
    return false;
  }
  
  // comparator operator 
  bool operator>(const NcmpiType& lhs,const NcmpiType& rhs)
  {
    return true;
  }
}

using namespace PnetCDF;

// assignment operator
NcmpiType& NcmpiType::operator=(const NcmpiType & rhs)
{
  nullObject = rhs.nullObject;
  myId= rhs.myId;
  groupId = rhs.groupId;
  return *this;
}

// The copy constructor.
NcmpiType::NcmpiType(const NcmpiType& rhs): 
  nullObject(rhs.nullObject),
  myId(rhs.myId), 
  groupId(rhs.groupId)
{}


// Constructor generates a null object.
NcmpiType::NcmpiType() : 
  nullObject(true),
  myId(-1),
  groupId(-1)
{}

// constructor
NcmpiType::NcmpiType(const NcmpiGroup& grp, const string& name) :
  nullObject (false)
{
  groupId= grp.getId();
  NcmpiType typTmp(grp.getType(name,NcmpiGroup::ParentsAndCurrent));
  myId = typTmp.getId();
}

// constructor for a global type 
NcmpiType::NcmpiType(nc_type id) :
  nullObject(false),
  myId(id),
  groupId(0)
{
}
  

// Constructor for a non-global type
NcmpiType::NcmpiType(const PnetCDF::NcmpiGroup& grp, nc_type id):
  nullObject(false),
  myId(id),
  groupId(grp.getId())
{}


// equivalence operator
bool NcmpiType::operator==(const NcmpiType & rhs) const
{
  if(nullObject) 
    return nullObject == rhs.nullObject;
  else
    return groupId == rhs.groupId && myId == rhs.myId;
}  
  
//  !=  operator
bool NcmpiType::operator!=(const NcmpiType & rhs) const
{
  return !(*this == rhs);
}  
  
// Gets parent group.
NcmpiGroup  NcmpiType::getParentGroup() const {
  if(groupId == 0) return NcmpiGroup(); else  return NcmpiGroup(groupId);
}
  
static
string inq_type(int myId) {
    switch (myId) {
        case NC_BYTE   : return string("byte");
        case NC_UBYTE  : return string("ubyte");
        case NC_CHAR   : return string("char");
        case NC_SHORT  : return string("short");
        case NC_USHORT : return string("ushort");
        case NC_INT    : return string("int");
        case NC_UINT   : return string("uint");
        case NC_INT64  : return string("int64");
        case NC_UINT64 : return string("uint64");
        case NC_FLOAT  : return string("float");
        case NC_DOUBLE : return string("double");
        default: break;
    }
    return string("");
}

// Returns the type name.
string  NcmpiType::getName() const{
  // char charName[NC_MAX_NAME+1];
  // MPI_Offset *sizep=NULL;
  // ncmpiCheck(ncmpi_inq_type(groupId,myId,charName,sizep),__FILE__,__LINE__);
  // return string(charName);
  return inq_type(myId);
};

// Returns the size in bytes
MPI_Offset NcmpiType::getSize() const{
  char* charName=NULL;
  MPI_Offset sizep;
  ncmpiCheck(ncmpi_inq_type(groupId,myId,charName,&sizep),__FILE__,__LINE__);
  return sizep;
};
  
// The type class returned as an enumeration type.
NcmpiType::ncmpiType NcmpiType::getTypeClass() const{
  switch (myId) {
  case NC_BYTE    : return ncmpi_BYTE;
  case NC_UBYTE   : return ncmpi_UBYTE;
  case NC_CHAR    : return ncmpi_CHAR;
  case NC_SHORT   : return ncmpi_SHORT;
  case NC_USHORT  : return ncmpi_USHORT;
  case NC_INT     : return ncmpi_INT;
  case NC_UINT    : return ncmpi_UINT;  
  case NC_INT64   : return ncmpi_INT64; 
  case NC_UINT64  : return ncmpi_UINT64;
  case NC_FLOAT   : return ncmpi_FLOAT;
  case NC_DOUBLE  : return ncmpi_DOUBLE;
  default:  
    // this is a user defined type
    // establish its type class, ie whether it is: NC_VLEN, NC_OPAQUE, NC_ENUM, or NC_COMPOUND. 
    char* name=NULL;
    MPI_Offset* sizep=NULL;
    nc_type* base_nc_typep=NULL;
    MPI_Offset* nfieldsp=NULL;
    int classp;
    ncmpiCheck(ncmpi_inq_user_type(groupId,myId,name,sizep,base_nc_typep,nfieldsp,&classp),__FILE__,__LINE__);
    return static_cast<ncmpiType>(classp);
  }
}
  
// The type class returned as a string.
string NcmpiType::getTypeClassName() const{
  ncmpiType typeClass=getTypeClass();
  switch (typeClass) {
  case ncmpi_BYTE    : return string("ncmpi_BYTE");
  case ncmpi_UBYTE   : return string("ncmpi_UBYTE");
  case ncmpi_CHAR    : return string("ncmpi_CHAR");
  case ncmpi_SHORT   : return string("ncmpi_SHORT");
  case ncmpi_USHORT  : return string("ncmpi_USHORT");
  case ncmpi_INT     : return string("ncmpi_INT");
  case ncmpi_UINT    : return string("ncmpi_UINT");  
  case ncmpi_INT64   : return string("ncmpi_INT64"); 
  case ncmpi_UINT64  : return string("ncmpi_UINT64");
  case ncmpi_FLOAT   : return string("ncmpi_FLOAT");
  case ncmpi_DOUBLE  : return string("ncmpi_DOUBLE");
  case ncmpi_STRING  : return string("ncmpi_STRING");
  case ncmpi_VLEN    : return string("ncmpi_VLEN");
  case ncmpi_OPAQUE  : return string("ncmpi_OPAQUE");
  case ncmpi_ENUM    : return string("ncmpi_ENUM");
  case ncmpi_COMPOUND: return string("ncmpi_COMPOUND");
  }
  // we never get here!
  return "Dummy";
}

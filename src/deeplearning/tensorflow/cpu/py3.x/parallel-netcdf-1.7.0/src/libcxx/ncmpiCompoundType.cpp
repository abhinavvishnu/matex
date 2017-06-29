#include "ncmpiGroup.h"
#include "ncmpiCheck.h"
#include "ncmpiCompoundType.h"
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
NcmpiCompoundType& NcmpiCompoundType::operator=(const NcmpiCompoundType& rhs)
{
  NcmpiType::operator=(rhs);    // assign base class parts
  return *this;
}

// assignment operator
NcmpiCompoundType& NcmpiCompoundType::operator=(const NcmpiType& rhs)
{
  if (&rhs != this) {
    // check the rhs is the base of a Compound type
    if(getTypeClass() != ncmpi_COMPOUND) 	throw NcmpiException("The NcmpiType object must be the base of a Compound type.",__FILE__,__LINE__);
    // assign base class parts
    NcmpiType::operator=(rhs);
  }
  return *this;
}

// The copy constructor.
NcmpiCompoundType::NcmpiCompoundType(const NcmpiCompoundType& rhs): 
  NcmpiType(rhs)
{
}


// equivalence operator
bool NcmpiCompoundType::operator==(const NcmpiCompoundType& rhs)
{
  if(nullObject) 
    return nullObject == rhs.nullObject;
  else
    return myId ==rhs.myId && groupId == rhs.groupId;
}  
  
// Constructor generates a null object.
NcmpiCompoundType::NcmpiCompoundType() : 
  NcmpiType()   // invoke base class constructor
{}
  
// constructor
NcmpiCompoundType::NcmpiCompoundType(const NcmpiGroup& grp, const string& name): 
  NcmpiType(grp,name)
{
}

// constructor
// The copy constructor.
NcmpiCompoundType::NcmpiCompoundType(const NcmpiType& rhs): 
  NcmpiType()
{
  // assign base class parts
  NcmpiType::operator=(rhs);
}
  
//  Inserts a named field.
void NcmpiCompoundType::addMember(const string& memberName, const NcmpiType& newMemberType,MPI_Offset offset)
{
  ncmpiCheck(ncmpi_insert_compound(groupId,myId,const_cast<char*>(memberName.c_str()),offset,newMemberType.getId()),__FILE__,__LINE__);
}



//  Inserts a named array field.
void NcmpiCompoundType::addMember(const string& memberName, const NcmpiType& newMemberType, MPI_Offset offset, const vector<int>& shape)
{
  if (!shape.empty())
    ncmpiCheck(ncmpi_insert_array_compound(groupId, myId,const_cast<char*>(memberName.c_str()), offset, newMemberType.getId(), shape.size(), const_cast<int*>(&shape[0])),__FILE__,__LINE__);
  else
    addMember(memberName, newMemberType, offset);
}



// Returns number of members in this NcmpiCompoundType object.
MPI_Offset  NcmpiCompoundType::getMemberCount() const
{
  MPI_Offset nfieldsp;
  ncmpiCheck(ncmpi_inq_compound_nfields(groupId,myId,&nfieldsp),__FILE__,__LINE__);
  return nfieldsp;
}
  

// Returns a NcmpiType object for a single member. */
NcmpiType NcmpiCompoundType::getMember(int memberIndex) const 
{
  nc_type fieldtypeidp;
  ncmpiCheck(ncmpi_inq_compound_fieldtype(groupId,myId,memberIndex,&fieldtypeidp),__FILE__,__LINE__);
  switch (fieldtypeidp) {
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
    return NcmpiType(getParentGroup(),fieldtypeidp);
  }
}

  
// Returns the number of dimensions of a member with the given index.
int NcmpiCompoundType::getMemberDimCount(int memberIndex) const 
{
  int ndimsp;
  ncmpiCheck(ncmpi_inq_compound_fieldndims(groupId,myId,memberIndex, &ndimsp),__FILE__,__LINE__);
  return ndimsp;
}
  
  
// Returns the shape of the given member.
vector<int> NcmpiCompoundType::getMemberShape(int memberIndex) const 
{
  vector<int> dim_size;
  dim_size.resize(getMemberDimCount(memberIndex));
  if(!dim_size.empty())
    ncmpiCheck(ncmpi_inq_compound_fielddim_sizes(groupId,myId,memberIndex,&dim_size[0]),__FILE__,__LINE__);
  return dim_size;
}
 

// Returns the offset of the member with given index.
MPI_Offset NcmpiCompoundType::getMemberOffset(const int index) const
{
  MPI_Offset offsetp;
  ncmpiCheck(ncmpi_inq_compound_fieldoffset(groupId,myId, index,&offsetp),__FILE__,__LINE__);
  return offsetp;
}

#include "ncmpiAtt.h"
#include "ncmpiGroup.h"
#include "ncmpiCheck.h"
#include <vector>
#include <assert.h>

using namespace std;
using namespace PnetCDF;
  

// destructor  (defined even though it is virtual)
NcmpiAtt::~NcmpiAtt() {}

// assignment operator
NcmpiAtt& NcmpiAtt::operator=(const NcmpiAtt& rhs)
{
  nullObject = rhs.nullObject;
  myName = rhs.myName;
  groupId = rhs.groupId;
  varId =rhs.varId;
  return *this;
}

// Constructor generates a null object.
NcmpiAtt::NcmpiAtt() : 
  nullObject(true),
  groupId(-1),
  varId(-1)
{}

// Constructor for non-null instances.
NcmpiAtt::NcmpiAtt(bool nullObject): 
  nullObject(nullObject),
  groupId(-1),
  varId(-1)
{}

// The copy constructor.
NcmpiAtt::NcmpiAtt(const NcmpiAtt& rhs) :
  nullObject(rhs.nullObject),
  myName(rhs.myName),
  groupId(rhs.groupId),
   varId(rhs.varId)
{}


// equivalence operator
bool NcmpiAtt::operator==(const NcmpiAtt & rhs) const
{
  if(nullObject) 
    return nullObject == rhs.nullObject;
  else
    return myName == rhs.myName && groupId == rhs.groupId && varId == rhs.varId;
}  

//  !=  operator
bool NcmpiAtt::operator!=(const NcmpiAtt & rhs) const
{
  return !(*this == rhs);
}  

// Gets parent group.
PnetCDF::NcmpiGroup  NcmpiAtt::getParentGroup() const {
  return PnetCDF::NcmpiGroup(groupId);
}
      

// Returns the attribute type.
NcmpiType  NcmpiAtt::getType() const{
  // get the identifier for the netCDF type of this attribute.
  nc_type xtypep;
  ncmpiCheck(ncmpi_inq_atttype(groupId,varId,myName.c_str(),&xtypep),__FILE__,__LINE__);
  if(xtypep <= 12)
    // This is an atomic type
    return NcmpiType(xtypep);
  else
    // this is a user-defined type
    {
      // now get the set of NcmpiType objects in this file.
      multimap<string,NcmpiType> typeMap(getParentGroup().getTypes(NcmpiGroup::ParentsAndCurrent));
      multimap<string,NcmpiType>::iterator iter;
      // identify the Nctype object with the same id as this attribute.
      for (iter=typeMap.begin(); iter!= typeMap.end();iter++) {
	if(iter->second.getId() == xtypep) return iter->second;
      }
      // return a null object, as no type was identified.
      return NcmpiType();
    }
}

// Gets attribute length.
MPI_Offset  NcmpiAtt::getAttLength() const{
  MPI_Offset lenp;
  ncmpiCheck(ncmpi_inq_attlen(groupId, varId, myName.c_str(), &lenp),__FILE__,__LINE__);
  return lenp;
}

// Gets a netCDF variable attribute.
void NcmpiAtt::getValues(string& dataValues) const {
  NcmpiType::ncmpiType typeClass(getType().getTypeClass());

  MPI_Offset att_len=getAttLength();
  char* tmpValues;
  assert(att_len == (MPI_Offset)(size_t)att_len);
  tmpValues = (char *) malloc((size_t)att_len + 1);  /* + 1 for trailing null */

  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_get_att(groupId,varId,myName.c_str(),tmpValues),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_get_att_text(groupId,varId,myName.c_str(),tmpValues),__FILE__,__LINE__);
  dataValues=string(tmpValues, (size_t)att_len);
  free(tmpValues);
}

// Gets a netCDF variable attribute.
void NcmpiAtt::getValues(char* dataValues) const {
  NcmpiType::ncmpiType typeClass(getType().getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_get_att(groupId,varId,myName.c_str(),dataValues),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_get_att_text(groupId,varId,myName.c_str(),dataValues),__FILE__,__LINE__);
}


// Gets a netCDF variable attribute.
void NcmpiAtt::getValues(unsigned char* dataValues) const {
  NcmpiType::ncmpiType typeClass(getType().getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_get_att(groupId,varId,myName.c_str(),dataValues),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_get_att_uchar(groupId,varId,myName.c_str(),dataValues),__FILE__,__LINE__);
}

// Gets a netCDF variable attribute.
void NcmpiAtt::getValues(signed char* dataValues) const {
  NcmpiType::ncmpiType typeClass(getType().getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_get_att(groupId,varId,myName.c_str(),dataValues),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_get_att_schar(groupId,varId,myName.c_str(),dataValues),__FILE__,__LINE__);
}

// Gets a netCDF variable attribute.
void NcmpiAtt::getValues(short* dataValues) const {
  NcmpiType::ncmpiType typeClass(getType().getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_get_att(groupId,varId,myName.c_str(),dataValues),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_get_att_short(groupId,varId,myName.c_str(),dataValues),__FILE__,__LINE__);
}

// Gets a netCDF variable attribute.
void NcmpiAtt::getValues(int* dataValues) const {
  NcmpiType::ncmpiType typeClass(getType().getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_get_att(groupId,varId,myName.c_str(),dataValues),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_get_att_int(groupId,varId,myName.c_str(),dataValues),__FILE__,__LINE__);
}

// Gets a netCDF variable attribute.
void NcmpiAtt::getValues(long* dataValues) const {
  NcmpiType::ncmpiType typeClass(getType().getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_get_att(groupId,varId,myName.c_str(),dataValues),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_get_att_long(groupId,varId,myName.c_str(),dataValues),__FILE__,__LINE__);
}

// Gets a netCDF variable attribute.
void NcmpiAtt::getValues(float* dataValues) const {
  NcmpiType::ncmpiType typeClass(getType().getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_get_att(groupId,varId,myName.c_str(),dataValues),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_get_att_float(groupId,varId,myName.c_str(),dataValues),__FILE__,__LINE__);
}

// Gets a netCDF variable attribute.
void NcmpiAtt::getValues(double* dataValues) const {
  NcmpiType::ncmpiType typeClass(getType().getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_get_att(groupId,varId,myName.c_str(),dataValues),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_get_att_double(groupId,varId,myName.c_str(),dataValues),__FILE__,__LINE__);
}

// Gets a netCDF variable attribute.
void NcmpiAtt::getValues(unsigned short* dataValues) const {
  NcmpiType::ncmpiType typeClass(getType().getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_get_att(groupId,varId,myName.c_str(),dataValues),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_get_att_ushort(groupId,varId,myName.c_str(),dataValues),__FILE__,__LINE__);
}

// Gets a netCDF variable attribute.
void NcmpiAtt::getValues(unsigned int* dataValues) const {
  NcmpiType::ncmpiType typeClass(getType().getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_get_att(groupId,varId,myName.c_str(),dataValues),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_get_att_uint(groupId,varId,myName.c_str(),dataValues),__FILE__,__LINE__);
}

// Gets a netCDF variable attribute.
void NcmpiAtt::getValues(long long* dataValues) const {
  NcmpiType::ncmpiType typeClass(getType().getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_get_att(groupId,varId,myName.c_str(),dataValues),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_get_att_longlong(groupId,varId,myName.c_str(),dataValues),__FILE__,__LINE__);
}

// Gets a netCDF variable attribute.
void NcmpiAtt::getValues(unsigned long long* dataValues) const {
  NcmpiType::ncmpiType typeClass(getType().getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_get_att(groupId,varId,myName.c_str(),dataValues),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_get_att_ulonglong(groupId,varId,myName.c_str(),dataValues),__FILE__,__LINE__);
}

// Gets a netCDF variable attribute.
void NcmpiAtt::getValues(void* dataValues) const {
  ncmpiCheck(ncmpi_get_att(groupId,varId,myName.c_str(),dataValues),__FILE__,__LINE__);
}


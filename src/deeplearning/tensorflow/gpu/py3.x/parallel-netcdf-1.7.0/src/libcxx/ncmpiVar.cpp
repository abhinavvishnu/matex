#include "ncmpiVarAtt.h"
#include "ncmpiDim.h"
#include "ncmpiVar.h"
#include "ncmpiGroup.h"
#include "ncmpiCheck.h"
#include "ncmpiException.h"
#include<pnetcdf.h>
using namespace std;
using namespace PnetCDF::exceptions;
  
namespace PnetCDF {
  //  Global comparator operator ==============
  // comparator operator 
  bool operator<(const NcmpiVar& lhs,const NcmpiVar& rhs)
  {
    return false;
  }
  
  // comparator operator 
  bool operator>(const NcmpiVar& lhs,const NcmpiVar& rhs)
  {
    return true;
  }
}

using namespace PnetCDF;

// assignment operator
NcmpiVar& NcmpiVar::operator=(const NcmpiVar & rhs)
{
  nullObject = rhs.nullObject;
  myId = rhs.myId;
  groupId = rhs.groupId;
  return *this;
}

// The copy constructor.
NcmpiVar::NcmpiVar(const NcmpiVar& rhs) : 
  nullObject(rhs.nullObject),
  myId(rhs.myId),
  groupId(rhs.groupId)
{}


// equivalence operator
bool NcmpiVar::operator==(const NcmpiVar & rhs) const
{
  // simply check the netCDF id.
  return (myId == rhs.myId);
}  

//  !=  operator
bool NcmpiVar::operator!=(const NcmpiVar & rhs) const
{
  return !(*this == rhs);
}  

/////////////////
  
// Constructors and intialization
  
/////////////////
  
// Constructor generates a null object.
NcmpiVar::NcmpiVar() :
  nullObject(true),
  myId(-1),
  groupId(-1)
{}

// Constructor for a variable (must already exist in the netCDF file.)
NcmpiVar::NcmpiVar (const NcmpiGroup& grp, const int& varId) :
  nullObject (false),
  myId (varId),
  groupId(grp.getId())
{}

  
  
// Gets parent group.
NcmpiGroup  NcmpiVar::getParentGroup() const {
  return NcmpiGroup(groupId);
}
  

// Get the variable id.
int  NcmpiVar::getId() const {return myId;}
    
//////////////////////

//  Information about the variable type

/////////////////////


// Gets the NcmpiType object with a given name.
NcmpiType NcmpiVar::getType() const {

  // if this variable has not been defined, return a NULL type
  if(isNull()) return NcmpiType();

  // first get the typeid
  nc_type xtypep;
  ncmpiCheck(ncmpi_inq_vartype(groupId,myId,&xtypep),__FILE__,__LINE__);

  if(xtypep ==  ncmpiByte.getId()    ) return ncmpiByte;
  if(xtypep ==  ncmpiUbyte.getId()   ) return ncmpiUbyte;
  if(xtypep ==  ncmpiChar.getId()    ) return ncmpiChar;
  if(xtypep ==  ncmpiShort.getId()   ) return ncmpiShort;
  if(xtypep ==  ncmpiUshort.getId()  ) return ncmpiUshort;
  if(xtypep ==  ncmpiInt.getId()     ) return ncmpiInt;
  if(xtypep ==  ncmpiUint.getId()    ) return ncmpiUint;  
  if(xtypep ==  ncmpiInt64.getId()   ) return ncmpiInt64; 
  if(xtypep ==  ncmpiUint64.getId()  ) return ncmpiUint64;
  if(xtypep ==  ncmpiFloat.getId()   ) return ncmpiFloat;
  if(xtypep ==  ncmpiDouble.getId()  ) return ncmpiDouble;

  multimap<string,NcmpiType>::const_iterator it;
  multimap<string,NcmpiType> types(NcmpiGroup(groupId).getTypes(NcmpiGroup::ParentsAndCurrent));
  for(it=types.begin(); it!=types.end(); it++) {
    if(it->second.getId() == xtypep) return it->second;
  }
  // we will never reach here
  return true;
}







  
/////////////////
  
// Information about Dimensions
  
/////////////////

  
// Gets the number of dimensions.
int NcmpiVar::getDimCount() const
{
  // get the number of dimensions
  int dimCount;
  ncmpiCheck(ncmpi_inq_varndims(groupId,myId, &dimCount),__FILE__,__LINE__);
  return dimCount;
}  

// Gets the set of Ncdim objects.
vector<NcmpiDim> NcmpiVar::getDims() const
{
  // get the number of dimensions
  int dimCount = getDimCount();  
  // create a vector of dimensions.
  vector<NcmpiDim> ncmpiDims;
  if (dimCount){
    vector<int> dimids(dimCount);
    ncmpiCheck(ncmpi_inq_vardimid(groupId,myId, &dimids[0]),__FILE__,__LINE__);
    ncmpiDims.reserve(dimCount);
    for (int i=0; i<dimCount; i++){
      NcmpiDim tmpDim(getParentGroup(),dimids[i]);
      ncmpiDims.push_back(tmpDim);
    }
  }
  return ncmpiDims;
}  

  
// Gets the i'th NcmpiDim object.
NcmpiDim NcmpiVar::getDim(int i) const
{
  vector<NcmpiDim> ncmpiDims = getDims();
  if((size_t)i >= ncmpiDims.size() || i < 0) throw NcmpiException("Index out of range",__FILE__,__LINE__);
  return ncmpiDims[i];
}


/////////////////
  
// Information about Attributes
  
/////////////////

  
// Gets the number of attributes.
int NcmpiVar::getAttCount() const
{
  // get the number of attributes
  int attCount;
  ncmpiCheck(ncmpi_inq_varnatts(groupId,myId, &attCount),__FILE__,__LINE__);
  return attCount;
}  

// Gets the set of attributes.
map<string,NcmpiVarAtt> NcmpiVar::getAtts() const
{
  // get the number of attributes
  int attCount = getAttCount();  
  // create a container of attributes.
  map<string,NcmpiVarAtt> ncmpiAtts;
  for (int i=0; i<attCount; i++){
    NcmpiVarAtt tmpAtt(getParentGroup(),*this,i);
    ncmpiAtts.insert(pair<const string,NcmpiVarAtt>(tmpAtt.getName(),tmpAtt));
  }
  return ncmpiAtts;
}  

  
// Gets attribute by name.
NcmpiVarAtt NcmpiVar::getAtt(const string& name) const
{
  map<string,NcmpiVarAtt> attributeList = getAtts();
  map<string,NcmpiVarAtt>::iterator myIter;
  myIter = attributeList.find(name);
  if(myIter == attributeList.end()){
    string msg("Attribute '"+name+"' not found");
    throw NcmpiException(msg.c_str(),__FILE__,__LINE__);
  }
  return NcmpiVarAtt(myIter->second);
}



/////////////////////////


// Creates a new NetCDF variable attribute or if already exisiting replaces it.
NcmpiVarAtt NcmpiVar::putAtt(const string& name, const string& dataValues) const {
  ncmpiCheckDefineMode(groupId);
  ncmpiCheck(ncmpi_put_att_text(groupId,myId,name.c_str(),dataValues.size(),dataValues.c_str()),__FILE__,__LINE__);
  // finally instantiate this attribute and return
  return getAtt(name);
}

// Creates a new NetCDF variable attribute or if already exisiting replaces it.
NcmpiVarAtt NcmpiVar::putAtt(const string& name, const NcmpiType& type, MPI_Offset len, const unsigned char* dataValues) const {
  ncmpiCheckDefineMode(groupId);
  NcmpiType::ncmpiType typeClass(type.getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_put_att(groupId,myId,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_put_att_uchar(groupId,myId,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  // finally instantiate this attribute and return
  return getAtt(name);
}


// Creates a new NetCDF variable attribute or if already exisiting replaces it.
NcmpiVarAtt NcmpiVar::putAtt(const string& name, const NcmpiType& type, MPI_Offset len, const signed char* dataValues) const {
  ncmpiCheckDefineMode(groupId);
  NcmpiType::ncmpiType typeClass(type.getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_put_att(groupId,myId,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_put_att_schar(groupId,myId,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  // finally instantiate this attribute and return
  return getAtt(name);
}



/////////////////////////////////
// Creates a new NetCDF variable attribute or if already exisiting replaces it.
NcmpiVarAtt NcmpiVar::putAtt(const string& name, const NcmpiType& type, short datumValue) const {
  ncmpiCheckDefineMode(groupId);
  NcmpiType::ncmpiType typeClass(type.getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_put_att(groupId,myId,name.c_str(),type.getId(),1,&datumValue),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_put_att_short(groupId,myId,name.c_str(),type.getId(),1,&datumValue),__FILE__,__LINE__);
  // finally instantiate this attribute and return
  return getAtt(name);
}


// Creates a new NetCDF variable attribute or if already exisiting replaces it.
NcmpiVarAtt NcmpiVar::putAtt(const string& name, const NcmpiType& type, int datumValue) const {
  ncmpiCheckDefineMode(groupId);
  NcmpiType::ncmpiType typeClass(type.getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_put_att(groupId,myId,name.c_str(),type.getId(),1,&datumValue),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_put_att_int(groupId,myId,name.c_str(),type.getId(),1,&datumValue),__FILE__,__LINE__);
  // finally instantiate this attribute and return
  return getAtt(name);
}

// Creates a new NetCDF variable attribute or if already exisiting replaces it.
NcmpiVarAtt NcmpiVar::putAtt(const string& name, const NcmpiType& type, long datumValue) const {
  ncmpiCheckDefineMode(groupId);
  NcmpiType::ncmpiType typeClass(type.getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_put_att(groupId,myId,name.c_str(),type.getId(),1,&datumValue),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_put_att_long(groupId,myId,name.c_str(),type.getId(),1,&datumValue),__FILE__,__LINE__);
  // finally instantiate this attribute and return
  return getAtt(name);
}

// Creates a new NetCDF variable attribute or if already exisiting replaces it.
NcmpiVarAtt NcmpiVar::putAtt(const string& name, const NcmpiType& type, float datumValue) const {
  ncmpiCheckDefineMode(groupId);
  NcmpiType::ncmpiType typeClass(type.getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_put_att(groupId,myId,name.c_str(),type.getId(),1,&datumValue),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_put_att_float(groupId,myId,name.c_str(),type.getId(),1,&datumValue),__FILE__,__LINE__);
  // finally instantiate this attribute and return
  return getAtt(name);
}


// Creates a new NetCDF variable attribute or if already exisiting replaces it.
NcmpiVarAtt NcmpiVar::putAtt(const string& name, const NcmpiType& type, double datumValue) const {
  ncmpiCheckDefineMode(groupId);
  NcmpiType::ncmpiType typeClass(type.getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_put_att(groupId,myId,name.c_str(),type.getId(),1,&datumValue),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_put_att_double(groupId,myId,name.c_str(),type.getId(),1,&datumValue),__FILE__,__LINE__);
  // finally instantiate this attribute and return
  return getAtt(name);
}


// Creates a new NetCDF variable attribute or if already exisiting replaces it.
NcmpiVarAtt NcmpiVar::putAtt(const string& name, const NcmpiType& type, unsigned short datumValue) const {
  ncmpiCheckDefineMode(groupId);
  NcmpiType::ncmpiType typeClass(type.getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_put_att(groupId,myId,name.c_str(),type.getId(),1,&datumValue),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_put_att_ushort(groupId,myId,name.c_str(),type.getId(),1,&datumValue),__FILE__,__LINE__);
  // finally instantiate this attribute and return
  return getAtt(name);
}

// Creates a new NetCDF variable attribute or if already exisiting replaces it.
NcmpiVarAtt NcmpiVar::putAtt(const string& name, const NcmpiType& type, unsigned int datumValue) const {
  ncmpiCheckDefineMode(groupId);
  NcmpiType::ncmpiType typeClass(type.getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_put_att(groupId,myId,name.c_str(),type.getId(),1,&datumValue),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_put_att_uint(groupId,myId,name.c_str(),type.getId(),1,&datumValue),__FILE__,__LINE__);
  // finally instantiate this attribute and return
  return getAtt(name);
}

// Creates a new NetCDF variable attribute or if already exisiting replaces it.
NcmpiVarAtt NcmpiVar::putAtt(const string& name, const NcmpiType& type, long long datumValue) const {
  ncmpiCheckDefineMode(groupId);
  NcmpiType::ncmpiType typeClass(type.getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_put_att(groupId,myId,name.c_str(),type.getId(),1,&datumValue),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_put_att_longlong(groupId,myId,name.c_str(),type.getId(),1,&datumValue),__FILE__,__LINE__);
  // finally instantiate this attribute and return
  return getAtt(name);
}


// Creates a new NetCDF variable attribute or if already exisiting replaces it.
NcmpiVarAtt NcmpiVar::putAtt(const string& name, const NcmpiType& type, unsigned long long datumValue) const {
  ncmpiCheckDefineMode(groupId);
  NcmpiType::ncmpiType typeClass(type.getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_put_att(groupId,myId,name.c_str(),type.getId(),1,&datumValue),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_put_att_ulonglong(groupId,myId,name.c_str(),type.getId(),1,&datumValue),__FILE__,__LINE__);
  // finally instantiate this attribute and return
  return getAtt(name);
}


/////////////////////////////////











// Creates a new NetCDF variable attribute or if already exisiting replaces it.
NcmpiVarAtt NcmpiVar::putAtt(const string& name, const NcmpiType& type, MPI_Offset len, const short* dataValues) const {
  ncmpiCheckDefineMode(groupId);
  NcmpiType::ncmpiType typeClass(type.getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_put_att(groupId,myId,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_put_att_short(groupId,myId,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  // finally instantiate this attribute and return
  return getAtt(name);
}


// Creates a new NetCDF variable attribute or if already exisiting replaces it.
NcmpiVarAtt NcmpiVar::putAtt(const string& name, const NcmpiType& type, MPI_Offset len, const int* dataValues) const {
  ncmpiCheckDefineMode(groupId);
  NcmpiType::ncmpiType typeClass(type.getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_put_att(groupId,myId,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_put_att_int(groupId,myId,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  // finally instantiate this attribute and return
  return getAtt(name);
}

// Creates a new NetCDF variable attribute or if already exisiting replaces it.
NcmpiVarAtt NcmpiVar::putAtt(const string& name, const NcmpiType& type, MPI_Offset len, const long* dataValues) const {
  ncmpiCheckDefineMode(groupId);
  NcmpiType::ncmpiType typeClass(type.getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_put_att(groupId,myId,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_put_att_long(groupId,myId,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  // finally instantiate this attribute and return
  return getAtt(name);
}

// Creates a new NetCDF variable attribute or if already exisiting replaces it.
NcmpiVarAtt NcmpiVar::putAtt(const string& name, const NcmpiType& type, MPI_Offset len, const float* dataValues) const {
  ncmpiCheckDefineMode(groupId);
  NcmpiType::ncmpiType typeClass(type.getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_put_att(groupId,myId,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_put_att_float(groupId,myId,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  // finally instantiate this attribute and return
  return getAtt(name);
}


// Creates a new NetCDF variable attribute or if already exisiting replaces it.
NcmpiVarAtt NcmpiVar::putAtt(const string& name, const NcmpiType& type, MPI_Offset len, const double* dataValues) const {
  ncmpiCheckDefineMode(groupId);
  NcmpiType::ncmpiType typeClass(type.getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_put_att(groupId,myId,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_put_att_double(groupId,myId,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  // finally instantiate this attribute and return
  return getAtt(name);
}


// Creates a new NetCDF variable attribute or if already exisiting replaces it.
NcmpiVarAtt NcmpiVar::putAtt(const string& name, const NcmpiType& type, MPI_Offset len, const unsigned short* dataValues) const {
  ncmpiCheckDefineMode(groupId);
  NcmpiType::ncmpiType typeClass(type.getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_put_att(groupId,myId,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_put_att_ushort(groupId,myId,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  // finally instantiate this attribute and return
  return getAtt(name);
}

// Creates a new NetCDF variable attribute or if already exisiting replaces it.
NcmpiVarAtt NcmpiVar::putAtt(const string& name, const NcmpiType& type, MPI_Offset len, const unsigned int* dataValues) const {
  ncmpiCheckDefineMode(groupId);
  NcmpiType::ncmpiType typeClass(type.getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_put_att(groupId,myId,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_put_att_uint(groupId,myId,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  // finally instantiate this attribute and return
  return getAtt(name);
}

// Creates a new NetCDF variable attribute or if already exisiting replaces it.
NcmpiVarAtt NcmpiVar::putAtt(const string& name, const NcmpiType& type, MPI_Offset len, const long long* dataValues) const {
  ncmpiCheckDefineMode(groupId);
  NcmpiType::ncmpiType typeClass(type.getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_put_att(groupId,myId,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_put_att_longlong(groupId,myId,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  // finally instantiate this attribute and return
  return getAtt(name);
}


// Creates a new NetCDF variable attribute or if already exisiting replaces it.
NcmpiVarAtt NcmpiVar::putAtt(const string& name, const NcmpiType& type, MPI_Offset len, const unsigned long long* dataValues) const {
  ncmpiCheckDefineMode(groupId);
  NcmpiType::ncmpiType typeClass(type.getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_put_att(groupId,myId,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_put_att_ulonglong(groupId,myId,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  // finally instantiate this attribute and return
  return getAtt(name);
}


// Creates a new NetCDF variable attribute or if already exisiting replaces it.
NcmpiVarAtt NcmpiVar::putAtt(const string& name, const NcmpiType& type, MPI_Offset len, const void* dataValues) const {
  ncmpiCheckDefineMode(groupId);
  ncmpiCheck(ncmpi_put_att(groupId,myId ,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  // finally instantiate this attribute and return
  return getAtt(name);
}




////////////////////
  
// Other Basic variable info
  
////////////////////
  
// The name of this variable.
string NcmpiVar::getName() const{
  char charName[NC_MAX_NAME+1];
  ncmpiCheck(ncmpi_inq_varname(groupId, myId, charName),__FILE__,__LINE__);
  return string(charName);
}


////////////////////
  
// Chunking details
  
////////////////////


// Sets chunking parameters.
void NcmpiVar::setChunking(ChunkMode chunkMode, vector<MPI_Offset>& chunkSizes) const {
  MPI_Offset *chunkSizesPtr = chunkSizes.empty() ? 0 : &chunkSizes[0];
  ncmpiCheck(ncmpi_def_var_chunking(groupId,myId,static_cast<int> (chunkMode), chunkSizesPtr),__FILE__,__LINE__);
}
  
  
// Gets the chunking parameters
void NcmpiVar::getChunkingParameters(ChunkMode& chunkMode, vector<MPI_Offset>& chunkSizes) const {
  int chunkModeInt;
  chunkSizes.resize(getDimCount());
  MPI_Offset *chunkSizesPtr = chunkSizes.empty() ? 0 : &chunkSizes[0];
  ncmpiCheck(ncmpi_inq_var_chunking(groupId,myId, &chunkModeInt, chunkSizesPtr),__FILE__,__LINE__);
  chunkMode = static_cast<ChunkMode> (chunkModeInt);
}
  
  
  
  
////////////////////
  
// Fill details
  
////////////////////


// Sets the fill parameters
void NcmpiVar::setFill(bool fillMode, void* fillValue) const {
  // If fillMode is enabled, check that fillValue has a legal pointer.
  // if(fillMode && fillValue == NULL)
    // throw NcmpiException("FillMode was set to zero but fillValue has invalid pointer",__FILE__,__LINE__);
  
  ncmpiCheck(ncmpi_def_var_fill(groupId,myId,static_cast<int> (!fillMode),fillValue),__FILE__,__LINE__);
}

// Sets the fill parameters
/*
void NcmpiVar::setFill(bool fillMode, const void* fillValue) const {
  setFill(fillMode,const_cast<void*>(fillValue));
}
*/


// Gets the fill parameters
void NcmpiVar::getFillModeParameters(bool& fillMode, void* fillValue)const{
    
  int fillModeInt;
  ncmpiCheck(ncmpi_inq_var_fill(groupId,myId,&fillModeInt,fillValue),__FILE__,__LINE__);
  fillMode= static_cast<bool> (fillModeInt == 0);
}
  
/* fill variable */
void NcmpiVar::fillRec(MPI_Offset recno) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_fill_var_rec(groupId, myId, recno),__FILE__,__LINE__);
}


////////////////////
  
// Compression details
  
////////////////////


// Sets the compression parameters
void NcmpiVar::setCompression(bool enableShuffleFilter, bool enableDeflateFilter, int deflateLevel) const {
    
  // Check that the deflate level is legal
  if(enableDeflateFilter & (deflateLevel < 0 || deflateLevel >9))
    throw NcmpiException("The deflateLevel must be set between 0 and 9.",__FILE__,__LINE__);
    
  ncmpiCheck(ncmpi_def_var_deflate(groupId,myId,
			     static_cast<int> (enableShuffleFilter),
			     static_cast<int> (enableDeflateFilter),
			     deflateLevel),__FILE__,__LINE__);
}
  

// Gets the compression parameters
void NcmpiVar::getCompressionParameters(bool& shuffleFilterEnabled, bool& deflateFilterEnabled, int& deflateLevel) const {
    
  int enableShuffleFilterInt;
  int enableDeflateFilterInt;
  ncmpiCheck(ncmpi_inq_var_deflate(groupId,myId,
			     &enableShuffleFilterInt,
			     &enableDeflateFilterInt,
			     &deflateLevel),__FILE__,__LINE__);
  shuffleFilterEnabled =  static_cast<bool> (enableShuffleFilterInt);
  deflateFilterEnabled =  static_cast<bool> (enableDeflateFilterInt);
}
  


////////////////////
  
// Endianness details
  
////////////////////
      

// Sets the endianness of the variable.
void NcmpiVar::setEndianness(EndianMode endianMode) const {
    
  ncmpiCheck(ncmpi_def_var_endian(groupId,myId,static_cast<int> (endianMode)),__FILE__,__LINE__);
}
  
  
// Gets the endianness of the variable.
NcmpiVar::EndianMode NcmpiVar::getEndianness() const {
    
  int endianInt;
  ncmpiCheck(ncmpi_inq_var_endian(groupId,myId,&endianInt),__FILE__,__LINE__);
  return static_cast<EndianMode> (endianInt);
}
  
      
  
////////////////////
  
// Checksum details
  
////////////////////
  
  
// Sets the checksum parameters of a variable.
void NcmpiVar::setChecksum(ChecksumMode checksumMode) const {
  ncmpiCheck(ncmpi_def_var_fletcher32(groupId,myId,static_cast<int> (checksumMode)),__FILE__,__LINE__);
}
  
  
// Gets the checksum parameters of the variable.
NcmpiVar::ChecksumMode NcmpiVar::getChecksum() const {
  int checksumInt;
  ncmpiCheck(ncmpi_inq_var_fletcher32(groupId,myId,&checksumInt),__FILE__,__LINE__);
  return static_cast<ChecksumMode> (checksumInt);
}
  
  

  
////////////////////
  
//  renaming the variable
  
////////////////////
  
void NcmpiVar::rename( const string& newname ) const 
{
  ncmpiCheck(ncmpi_rename_var(groupId,myId,newname.c_str()),__FILE__,__LINE__);
}





////////////////////

//  data writing (independent I/O APIs)

////////////////////

// Writes the entire data into the netCDF variable.
void NcmpiVar::putVar(const char* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var_text(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable.
void NcmpiVar::putVar(const unsigned char* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var_uchar(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable.
void NcmpiVar::putVar(const signed char* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var_schar(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable.
void NcmpiVar::putVar(const short* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var_short(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable.
void NcmpiVar::putVar(const int* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var_int(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable.
void NcmpiVar::putVar(const long* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var_long(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable.
void NcmpiVar::putVar(const float* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var_float(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable.
void NcmpiVar::putVar(const double* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var_double(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable.
void NcmpiVar::putVar(const unsigned short* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var_ushort(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable.
void NcmpiVar::putVar(const unsigned int* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var_uint(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable.
void NcmpiVar::putVar(const long long* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var_longlong(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable.
void NcmpiVar::putVar(const unsigned long long* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var_ulonglong(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable with no data conversion.
void NcmpiVar::putVar(const void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const {
    ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var(groupId, myId,dataValues, bufcount, buftype),__FILE__,__LINE__);
}


///////////////////

// Writes a single datum value into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& index, const unsigned char* datumValue) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var1_uchar(groupId, myId,&index[0],datumValue),__FILE__,__LINE__);
}
// Writes a single datum value into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& index, const signed char* datumValue) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var1_schar(groupId, myId,&index[0],datumValue),__FILE__,__LINE__);
}
// Writes a single datum value into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& index, const short datumValue) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var1_short(groupId, myId,&index[0],&datumValue),__FILE__,__LINE__);
}
// Writes a single datum value into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& index, const int datumValue) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var1_int(groupId, myId,&index[0],&datumValue),__FILE__,__LINE__);
}
// Writes a single datum value into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& index, const long datumValue) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var1_long(groupId, myId,&index[0],&datumValue),__FILE__,__LINE__);
}
// Writes a single datum value into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& index, const float datumValue) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var1_float(groupId, myId,&index[0],&datumValue),__FILE__,__LINE__);
}
// Writes a single datum value into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& index, const double datumValue) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var1_double(groupId, myId,&index[0],&datumValue),__FILE__,__LINE__);
}
// Writes a single datum value into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& index, const unsigned short datumValue) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var1_ushort(groupId, myId,&index[0],&datumValue),__FILE__,__LINE__);
}
// Writes a single datum value into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& index, const unsigned int datumValue) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var1_uint(groupId, myId,&index[0],&datumValue),__FILE__,__LINE__);
}
// Writes a single datum value into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& index, const long long datumValue) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var1_longlong(groupId, myId,&index[0],&datumValue),__FILE__,__LINE__);
}
// Writes a single datum value into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& index, const unsigned long long datumValue) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var1_ulonglong(groupId, myId,&index[0],&datumValue),__FILE__,__LINE__);
}
// Writes a single datum value into the netCDF variable with no data conversion.
void NcmpiVar::putVar(const vector<MPI_Offset>& index, const void* datumValue, MPI_Offset bufcount, MPI_Datatype buftype) const {
    ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var1(groupId, myId,&index[0],datumValue, bufcount, buftype),__FILE__,__LINE__);
}


////////////////////

// Writes an array of values into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const char* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vara_text(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const unsigned char* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vara_uchar(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const signed char* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vara_schar(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const short* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vara_short(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const int* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vara_int(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const long* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vara_long(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const float* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vara_float(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const double* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vara_double(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const unsigned short* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vara_ushort(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const unsigned int* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vara_uint(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const long long* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vara_longlong(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const unsigned long long* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vara_ulonglong(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable with no data conversion.
void NcmpiVar::putVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const {
    ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vara(groupId, myId,&startp[0],&countp[0],dataValues, bufcount, buftype),__FILE__,__LINE__);
}

////////////////////

// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const char* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vars_text(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const unsigned char* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vars_uchar(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const signed char* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vars_schar(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const short* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vars_short(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const int* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vars_int(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const long* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vars_long(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const float* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vars_float(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const double* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vars_double(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const unsigned short* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vars_ushort(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const unsigned int* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vars_uint(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const long long* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vars_longlong(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const unsigned long long* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vars_ulonglong(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable with no data conversion.
void NcmpiVar::putVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const {
    ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vars(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues, bufcount, buftype),__FILE__,__LINE__);
}

////////////////////

// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const char* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varm_text(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const unsigned char* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varm_uchar(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const signed char* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varm_schar(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const short* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varm_short(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const int* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varm_int(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const long* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varm_long(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const float* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varm_float(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const double* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varm_double(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const unsigned short* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varm_ushort(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const unsigned int* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varm_uint(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const long long* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varm_longlong(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::putVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const unsigned long long* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varm_ulonglong(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable with no data conversion.
void NcmpiVar::putVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const {
    ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varm(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues, bufcount, buftype),__FILE__,__LINE__);
}

////////////////////

//  data writing (collective data mode)

////////////////////

// Writes the entire data into the netCDF variable.
void NcmpiVar::putVar_all(const char* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var_text_all(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable.
void NcmpiVar::putVar_all(const unsigned char* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var_uchar_all(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable.
void NcmpiVar::putVar_all(const signed char* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var_schar_all(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable.
void NcmpiVar::putVar_all(const short* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var_short_all(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable.
void NcmpiVar::putVar_all(const int* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var_int_all(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable.
void NcmpiVar::putVar_all(const long* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var_long_all(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable.
void NcmpiVar::putVar_all(const float* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var_float_all(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable.
void NcmpiVar::putVar_all(const double* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var_double_all(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable.
void NcmpiVar::putVar_all(const unsigned short* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var_ushort_all(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable.
void NcmpiVar::putVar_all(const unsigned int* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var_uint_all(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable.
void NcmpiVar::putVar_all(const long long* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var_longlong_all(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable.
void NcmpiVar::putVar_all(const unsigned long long* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var_ulonglong_all(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable with no data conversion.
void NcmpiVar::putVar_all(const void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const {
    ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var_all(groupId, myId,dataValues, bufcount, buftype),__FILE__,__LINE__);
}

///////////////////

// Writes a single datum value into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& index, const unsigned char* datumValue) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var1_uchar_all(groupId, myId,&index[0],datumValue),__FILE__,__LINE__);
}
// Writes a single datum value into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& index, const signed char* datumValue) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var1_schar_all(groupId, myId,&index[0],datumValue),__FILE__,__LINE__);
}
// Writes a single datum value into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& index, const short datumValue) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var1_short_all(groupId, myId,&index[0],&datumValue),__FILE__,__LINE__);
}
// Writes a single datum value into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& index, const int datumValue) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var1_int_all(groupId, myId,&index[0],&datumValue),__FILE__,__LINE__);
}
// Writes a single datum value into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& index, const long datumValue) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var1_long_all(groupId, myId,&index[0],&datumValue),__FILE__,__LINE__);
}
// Writes a single datum value into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& index, const float datumValue) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var1_float_all(groupId, myId,&index[0],&datumValue),__FILE__,__LINE__);
}
// Writes a single datum value into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& index, const double datumValue) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var1_double_all(groupId, myId,&index[0],&datumValue),__FILE__,__LINE__);
}
// Writes a single datum value into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& index, const unsigned short datumValue) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var1_ushort_all(groupId, myId,&index[0],&datumValue),__FILE__,__LINE__);
}
// Writes a single datum value into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& index, const unsigned int datumValue) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var1_uint_all(groupId, myId,&index[0],&datumValue),__FILE__,__LINE__);
}
// Writes a single datum value into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& index, const long long datumValue) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var1_longlong_all(groupId, myId,&index[0],&datumValue),__FILE__,__LINE__);
}
// Writes a single datum value into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& index, const unsigned long long datumValue) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var1_ulonglong_all(groupId, myId,&index[0],&datumValue),__FILE__,__LINE__);
}
// Writes a single datum value into the netCDF variable with no data conversion.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& index, const void* datumValue, MPI_Offset bufcount, MPI_Datatype buftype) const {
    ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_var1_all(groupId, myId,&index[0],datumValue, bufcount, buftype),__FILE__,__LINE__);
}


////////////////////

// Writes an array of values into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const char* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vara_text_all(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const unsigned char* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vara_uchar_all(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const signed char* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vara_schar_all(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const short* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vara_short_all(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const int* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vara_int_all(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const long* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vara_long_all(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const float* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vara_float_all(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const double* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vara_double_all(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const unsigned short* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vara_ushort_all(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const unsigned int* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vara_uint_all(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const long long* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vara_longlong_all(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const unsigned long long* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vara_ulonglong_all(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable with no data conversion.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const {
    ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vara_all(groupId, myId,&startp[0],&countp[0],dataValues, bufcount, buftype),__FILE__,__LINE__);
}

////////////////////

// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const char* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vars_text_all(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const unsigned char* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vars_uchar_all(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const signed char* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vars_schar_all(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const short* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vars_short_all(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const int* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vars_int_all(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const long* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vars_long_all(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const float* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vars_float_all(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const double* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vars_double_all(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const unsigned short* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vars_ushort_all(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const unsigned int* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vars_uint_all(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const long long* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vars_longlong_all(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const unsigned long long* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vars_ulonglong_all(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable with no data conversion.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const {
    ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vars_all(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues, bufcount, buftype),__FILE__,__LINE__);
}

////////////////////

// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const char* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varm_text_all(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const unsigned char* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varm_uchar_all(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const signed char* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varm_schar_all(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const short* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varm_short_all(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const int* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varm_int_all(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const long* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varm_long_all(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const float* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varm_float_all(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const double* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varm_double_all(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const unsigned short* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varm_ushort_all(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const unsigned int* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varm_uint_all(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const long long* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varm_longlong_all(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const unsigned long long* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varm_ulonglong_all(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable with no data conversion.
void NcmpiVar::putVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const {
    ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varm_all(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues, bufcount, buftype),__FILE__,__LINE__);
}

////////////////////

// Writes a list of subarrays into the netCDF variable. (independent I/O APIs)
void NcmpiVar::putVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const char* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varn_text(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::putVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const unsigned char* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varn_uchar(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::putVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const signed char* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varn_schar(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::putVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const short* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varn_short(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::putVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const int* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varn_int(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::putVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const long* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varn_long(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::putVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const float* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varn_float(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::putVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const double* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varn_double(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::putVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const unsigned short* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varn_ushort(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::putVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const unsigned int* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varn_uint(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::putVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const long long* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varn_longlong(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::putVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const unsigned long long* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varn_ulonglong(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable with no data conversion.
void NcmpiVar::putVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const {
    ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varn(groupId, myId, num, starts, counts, dataValues, bufcount, buftype),__FILE__,__LINE__);
}

////////////////////

// Writes a list of subarrays into the netCDF variable. (collective I/O APIs)
void NcmpiVar::putVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const char* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varn_text_all(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::putVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const unsigned char* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varn_uchar_all(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::putVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const signed char* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varn_schar_all(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::putVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const short* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varn_short_all(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::putVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const int* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varn_int_all(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::putVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const long* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varn_long_all(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::putVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const float* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varn_float_all(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::putVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const double* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varn_double_all(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::putVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const unsigned short* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varn_ushort_all(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::putVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const unsigned int* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varn_uint_all(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::putVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const long long* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varn_longlong_all(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::putVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const unsigned long long* dataValues) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varn_ulonglong_all(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable with no data conversion.
void NcmpiVar::putVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const {
    ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_varn_all(groupId, myId, num, starts, counts, dataValues, bufcount, buftype),__FILE__,__LINE__);
}

////////////////////

// Writes an array of values into the netCDF variable with filetype and buftype.
void NcmpiVar::putVard(MPI_Datatype filetype, const void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const {
    ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vard(groupId, myId, filetype, dataValues, bufcount, buftype),__FILE__,__LINE__);
}

void NcmpiVar::putVard_all(MPI_Datatype filetype, const void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const {
    ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_put_vard_all(groupId, myId, filetype, dataValues, bufcount, buftype),__FILE__,__LINE__);
}

////////////////////

//  Nonblocking data writing

////////////////////

// Writes the entire data into the netCDF variable.
void NcmpiVar::iputVar(const char* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_var_text(groupId, myId,dataValues, req),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable.
void NcmpiVar::iputVar(const unsigned char* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_var_uchar(groupId, myId,dataValues, req),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable.
void NcmpiVar::iputVar(const signed char* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_var_schar(groupId, myId,dataValues, req),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable.
void NcmpiVar::iputVar(const short* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_var_short(groupId, myId,dataValues, req),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable.
void NcmpiVar::iputVar(const int* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_var_int(groupId, myId,dataValues, req),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable.
void NcmpiVar::iputVar(const long* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_var_long(groupId, myId,dataValues, req),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable.
void NcmpiVar::iputVar(const float* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_var_float(groupId, myId,dataValues, req),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable.
void NcmpiVar::iputVar(const double* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_var_double(groupId, myId,dataValues, req),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable.
void NcmpiVar::iputVar(const unsigned short* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_var_ushort(groupId, myId,dataValues, req),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable.
void NcmpiVar::iputVar(const unsigned int* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_var_uint(groupId, myId,dataValues, req),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable.
void NcmpiVar::iputVar(const long long* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_var_longlong(groupId, myId,dataValues, req),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable.
void NcmpiVar::iputVar(const unsigned long long* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_var_ulonglong(groupId, myId,dataValues, req),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable with no data conversion.
void NcmpiVar::iputVar(const void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype, int *req) const {
    ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_var(groupId, myId,dataValues, bufcount, buftype, req),__FILE__,__LINE__);
}


///////////////////
// Writes a single datum value into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& index, const unsigned char* datumValue, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_var1_uchar(groupId, myId,&index[0],datumValue, req),__FILE__,__LINE__);
}
// Writes a single datum value into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& index, const signed char* datumValue, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_var1_schar(groupId, myId,&index[0],datumValue, req),__FILE__,__LINE__);
}
// Writes a single datum value into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& index, const short datumValue, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_var1_short(groupId, myId,&index[0],&datumValue, req),__FILE__,__LINE__);
}
// Writes a single datum value into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& index, const int datumValue, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_var1_int(groupId, myId,&index[0],&datumValue, req),__FILE__,__LINE__);
}
// Writes a single datum value into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& index, const long datumValue, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_var1_long(groupId, myId,&index[0],&datumValue, req),__FILE__,__LINE__);
}
// Writes a single datum value into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& index, const float datumValue, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_var1_float(groupId, myId,&index[0],&datumValue, req),__FILE__,__LINE__);
}
// Writes a single datum value into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& index, const double datumValue, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_var1_double(groupId, myId,&index[0],&datumValue, req),__FILE__,__LINE__);
}
// Writes a single datum value into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& index, const unsigned short datumValue, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_var1_ushort(groupId, myId,&index[0],&datumValue, req),__FILE__,__LINE__);
}
// Writes a single datum value into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& index, const unsigned int datumValue, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_var1_uint(groupId, myId,&index[0],&datumValue, req),__FILE__,__LINE__);
}
// Writes a single datum value into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& index, const long long datumValue, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_var1_longlong(groupId, myId,&index[0],&datumValue, req),__FILE__,__LINE__);
}
// Writes a single datum value into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& index, const unsigned long long datumValue, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_var1_ulonglong(groupId, myId,&index[0],&datumValue, req),__FILE__,__LINE__);
}
// Writes a single datum value into the netCDF variable with no data conversion.
void NcmpiVar::iputVar(const vector<MPI_Offset>& index, const void* datumValue, MPI_Offset bufcount, MPI_Datatype buftype, int *req) const {
    ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_var1(groupId, myId,&index[0],datumValue, bufcount, buftype, req),__FILE__,__LINE__);
}


////////////////////

// Writes an array of values into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const char* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_vara_text(groupId, myId,&startp[0],&countp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const unsigned char* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_vara_uchar(groupId, myId,&startp[0],&countp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const signed char* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_vara_schar(groupId, myId,&startp[0],&countp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const short* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_vara_short(groupId, myId,&startp[0],&countp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const int* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_vara_int(groupId, myId,&startp[0],&countp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const long* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_vara_long(groupId, myId,&startp[0],&countp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const float* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_vara_float(groupId, myId,&startp[0],&countp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const double* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_vara_double(groupId, myId,&startp[0],&countp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const unsigned short* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_vara_ushort(groupId, myId,&startp[0],&countp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const unsigned int* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_vara_uint(groupId, myId,&startp[0],&countp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const long long* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_vara_longlong(groupId, myId,&startp[0],&countp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const unsigned long long* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_vara_ulonglong(groupId, myId,&startp[0],&countp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable with no data conversion.
void NcmpiVar::iputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype, int *req) const {
    ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_vara(groupId, myId,&startp[0],&countp[0],dataValues, bufcount, buftype, req),__FILE__,__LINE__);
}



////////////////////

// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const char* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_vars_text(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const unsigned char* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_vars_uchar(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const signed char* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_vars_schar(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const short* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_vars_short(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const int* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_vars_int(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const long* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_vars_long(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const float* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_vars_float(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const double* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_vars_double(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const unsigned short* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_vars_ushort(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const unsigned int* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_vars_uint(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const long long* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_vars_longlong(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const unsigned long long* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_vars_ulonglong(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable with no data conversion.
void NcmpiVar::iputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype, int *req) const {
    ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_vars(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues, bufcount, buftype, req),__FILE__,__LINE__);
}


////////////////////
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const char* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_varm_text(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const unsigned char* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_varm_uchar(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const signed char* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_varm_schar(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const short* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_varm_short(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const int* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_varm_int(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const long* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_varm_long(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const float* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_varm_float(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const double* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_varm_double(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const unsigned short* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_varm_ushort(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const unsigned int* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_varm_uint(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const long long* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_varm_longlong(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::iputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const unsigned long long* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_varm_ulonglong(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable with no data conversion.
void NcmpiVar::iputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype, int *req) const {
    ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_varm(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues, bufcount, buftype, req),__FILE__,__LINE__);
}

////////////////////

// Nonblocking writes a list of subarrays into the netCDF variable.
void NcmpiVar::iputVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const char* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_varn_text(groupId, myId, num, starts, counts, dataValues, req),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::iputVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const unsigned char* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_varn_uchar(groupId, myId, num, starts, counts, dataValues, req),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::iputVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const signed char* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_varn_schar(groupId, myId, num, starts, counts, dataValues, req),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::iputVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const short* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_varn_short(groupId, myId, num, starts, counts, dataValues, req),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::iputVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const int* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_varn_int(groupId, myId, num, starts, counts, dataValues, req),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::iputVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const long* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_varn_long(groupId, myId, num, starts, counts, dataValues, req),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::iputVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const float* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_varn_float(groupId, myId, num, starts, counts, dataValues, req),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::iputVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const double* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_varn_double(groupId, myId, num, starts, counts, dataValues, req),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::iputVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const unsigned short* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_varn_ushort(groupId, myId, num, starts, counts, dataValues, req),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::iputVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const unsigned int* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_varn_uint(groupId, myId, num, starts, counts, dataValues, req),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::iputVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const long long* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_varn_longlong(groupId, myId, num, starts, counts, dataValues, req),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::iputVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const unsigned long long* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_varn_ulonglong(groupId, myId, num, starts, counts, dataValues, req),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable with no data conversion.
void NcmpiVar::iputVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype, int *req) const {
    ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_iput_varn(groupId, myId, num, starts, counts, dataValues, bufcount, buftype, req),__FILE__,__LINE__);
}

////////////////////

//  Buffered nonblocking data writing

////////////////////

// Writes the entire data into the netCDF variable.
void NcmpiVar::bputVar(const char* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_var_text(groupId, myId,dataValues, req),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable.
void NcmpiVar::bputVar(const unsigned char* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_var_uchar(groupId, myId,dataValues, req),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable.
void NcmpiVar::bputVar(const signed char* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_var_schar(groupId, myId,dataValues, req),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable.
void NcmpiVar::bputVar(const short* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_var_short(groupId, myId,dataValues, req),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable.
void NcmpiVar::bputVar(const int* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_var_int(groupId, myId,dataValues, req),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable.
void NcmpiVar::bputVar(const long* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_var_long(groupId, myId,dataValues, req),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable.
void NcmpiVar::bputVar(const float* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_var_float(groupId, myId,dataValues, req),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable.
void NcmpiVar::bputVar(const double* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_var_double(groupId, myId,dataValues, req),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable.
void NcmpiVar::bputVar(const unsigned short* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_var_ushort(groupId, myId,dataValues, req),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable.
void NcmpiVar::bputVar(const unsigned int* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_var_uint(groupId, myId,dataValues, req),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable.
void NcmpiVar::bputVar(const long long* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_var_longlong(groupId, myId,dataValues, req),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable.
void NcmpiVar::bputVar(const unsigned long long* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_var_ulonglong(groupId, myId,dataValues, req),__FILE__,__LINE__);
}
// Writes the entire data into the netCDF variable with no data conversion.
void NcmpiVar::bputVar(const void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype, int *req) const {
    ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_var(groupId, myId,dataValues, bufcount, buftype, req),__FILE__,__LINE__);
}


///////////////////
// Writes a single datum value into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& index, const unsigned char* datumValue, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_var1_uchar(groupId, myId,&index[0],datumValue, req),__FILE__,__LINE__);
}
// Writes a single datum value into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& index, const signed char* datumValue, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_var1_schar(groupId, myId,&index[0],datumValue, req),__FILE__,__LINE__);
}
// Writes a single datum value into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& index, const short datumValue, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_var1_short(groupId, myId,&index[0],&datumValue, req),__FILE__,__LINE__);
}
// Writes a single datum value into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& index, const int datumValue, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_var1_int(groupId, myId,&index[0],&datumValue, req),__FILE__,__LINE__);
}
// Writes a single datum value into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& index, const long datumValue, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_var1_long(groupId, myId,&index[0],&datumValue, req),__FILE__,__LINE__);
}
// Writes a single datum value into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& index, const float datumValue, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_var1_float(groupId, myId,&index[0],&datumValue, req),__FILE__,__LINE__);
}
// Writes a single datum value into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& index, const double datumValue, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_var1_double(groupId, myId,&index[0],&datumValue, req),__FILE__,__LINE__);
}
// Writes a single datum value into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& index, const unsigned short datumValue, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_var1_ushort(groupId, myId,&index[0],&datumValue, req),__FILE__,__LINE__);
}
// Writes a single datum value into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& index, const unsigned int datumValue, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_var1_uint(groupId, myId,&index[0],&datumValue, req),__FILE__,__LINE__);
}
// Writes a single datum value into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& index, const long long datumValue, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_var1_longlong(groupId, myId,&index[0],&datumValue, req),__FILE__,__LINE__);
}
// Writes a single datum value into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& index, const unsigned long long datumValue, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_var1_ulonglong(groupId, myId,&index[0],&datumValue, req),__FILE__,__LINE__);
}
// Writes a single datum value into the netCDF variable with no data conversion.
void NcmpiVar::bputVar(const vector<MPI_Offset>& index, const void* datumValue, MPI_Offset bufcount, MPI_Datatype buftype, int *req) const {
    ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_var1(groupId, myId,&index[0],datumValue, bufcount, buftype, req),__FILE__,__LINE__);
}


////////////////////

// Writes an array of values into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const char* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_vara_text(groupId, myId,&startp[0],&countp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const unsigned char* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_vara_uchar(groupId, myId,&startp[0],&countp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const signed char* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_vara_schar(groupId, myId,&startp[0],&countp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const short* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_vara_short(groupId, myId,&startp[0],&countp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const int* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_vara_int(groupId, myId,&startp[0],&countp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const long* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_vara_long(groupId, myId,&startp[0],&countp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const float* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_vara_float(groupId, myId,&startp[0],&countp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const double* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_vara_double(groupId, myId,&startp[0],&countp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const unsigned short* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_vara_ushort(groupId, myId,&startp[0],&countp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const unsigned int* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_vara_uint(groupId, myId,&startp[0],&countp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const long long* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_vara_longlong(groupId, myId,&startp[0],&countp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const unsigned long long* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_vara_ulonglong(groupId, myId,&startp[0],&countp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes an array of values into the netCDF variable with no data conversion.
void NcmpiVar::bputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype, int *req) const {
    ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_vara(groupId, myId,&startp[0],&countp[0],dataValues, bufcount, buftype, req),__FILE__,__LINE__);
}



////////////////////

// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const char* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_vars_text(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const unsigned char* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_vars_uchar(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const signed char* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_vars_schar(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const short* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_vars_short(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const int* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_vars_int(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const long* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_vars_long(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const float* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_vars_float(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const double* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_vars_double(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const unsigned short* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_vars_ushort(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const unsigned int* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_vars_uint(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const long long* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_vars_longlong(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const unsigned long long* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_vars_ulonglong(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a set of subsampled array values into the netCDF variable with no data conversion.
void NcmpiVar::bputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep,  const void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype, int *req) const {
    ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_vars(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues, bufcount, buftype, req),__FILE__,__LINE__);
}


////////////////////
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const char* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_varm_text(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const unsigned char* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_varm_uchar(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const signed char* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_varm_schar(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const short* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_varm_short(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const int* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_varm_int(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const long* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_varm_long(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const float* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_varm_float(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const double* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_varm_double(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const unsigned short* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_varm_ushort(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const unsigned int* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_varm_uint(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const long long* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_varm_longlong(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable.
void NcmpiVar::bputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const unsigned long long* dataValues, int *req) const {
  ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_varm_ulonglong(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues, req),__FILE__,__LINE__);
}
// Writes a mapped array section of values into the netCDF variable with no data conversion.
void NcmpiVar::bputVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>&countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, const void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype, int *req) const {
    ncmpiCheckDataMode(groupId);
    ncmpiCheck(ncmpi_bput_varm(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues, bufcount, buftype, req),__FILE__,__LINE__);
}


// Data reading (independent I/O APIs)

// Reads the entire data of the netCDF variable.
void NcmpiVar::getVar(char* dataValues) const {
    ncmpiCheck(ncmpi_get_var_text(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Reads the entire data of the netCDF variable.
void NcmpiVar::getVar(unsigned char* dataValues) const {
    ncmpiCheck(ncmpi_get_var_uchar(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Reads the entire data of the netCDF variable.
void NcmpiVar::getVar(signed char* dataValues) const {
    ncmpiCheck(ncmpi_get_var_schar(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Reads the entire data of the netCDF variable.
void NcmpiVar::getVar(short* dataValues) const {
    ncmpiCheck(ncmpi_get_var_short(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Reads the entire data of the netCDF variable.
void NcmpiVar::getVar(int* dataValues) const {
    ncmpiCheck(ncmpi_get_var_int(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Reads the entire data of the netCDF variable.
void NcmpiVar::getVar(long* dataValues) const {
    ncmpiCheck(ncmpi_get_var_long(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Reads the entire data of the netCDF variable.
void NcmpiVar::getVar(float* dataValues) const {
    ncmpiCheck(ncmpi_get_var_float(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Reads the entire data of the netCDF variable.
void NcmpiVar::getVar(double* dataValues) const {
    ncmpiCheck(ncmpi_get_var_double(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Reads the entire data of the netCDF variable.
void NcmpiVar::getVar(unsigned short* dataValues) const {
    ncmpiCheck(ncmpi_get_var_ushort(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Reads the entire data of the netCDF variable.
void NcmpiVar::getVar(unsigned int* dataValues) const {
    ncmpiCheck(ncmpi_get_var_uint(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Reads the entire data of the netCDF variable.
void NcmpiVar::getVar(long long* dataValues) const {
    ncmpiCheck(ncmpi_get_var_longlong(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Reads the entire data of the netCDF variable.
void NcmpiVar::getVar(unsigned long long* dataValues) const {
    ncmpiCheck(ncmpi_get_var_ulonglong(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Reads the entire data of the netCDF variable with no data conversion.
void NcmpiVar::getVar(void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const {
    ncmpiCheck(ncmpi_get_var(groupId, myId,dataValues, bufcount, buftype),__FILE__,__LINE__);
}

///////////

// Reads a single datum value of a netCDF variable.
void NcmpiVar::getVar(const vector<MPI_Offset>& index, char* datumValue) const {
    ncmpiCheck(ncmpi_get_var1_text(groupId, myId,&index[0],datumValue),__FILE__,__LINE__);
}
// Reads a single datum value of a netCDF variable.
void NcmpiVar::getVar(const vector<MPI_Offset>& index, unsigned char* datumValue) const {
    ncmpiCheck(ncmpi_get_var1_uchar(groupId, myId,&index[0],datumValue),__FILE__,__LINE__);
}
// Reads a single datum value of a netCDF variable.
void NcmpiVar::getVar(const vector<MPI_Offset>& index, signed char* datumValue) const {
    ncmpiCheck(ncmpi_get_var1_schar(groupId, myId,&index[0],datumValue),__FILE__,__LINE__);
}
// Reads a single datum value of a netCDF variable.
void NcmpiVar::getVar(const vector<MPI_Offset>& index, short* datumValue) const {
    ncmpiCheck(ncmpi_get_var1_short(groupId, myId,&index[0],datumValue),__FILE__,__LINE__);
}
// Reads a single datum value of a netCDF variable.
void NcmpiVar::getVar(const vector<MPI_Offset>& index, int* datumValue) const {
    ncmpiCheck(ncmpi_get_var1_int(groupId, myId,&index[0],datumValue),__FILE__,__LINE__);
}
// Reads a single datum value of a netCDF variable.
void NcmpiVar::getVar(const vector<MPI_Offset>& index, long* datumValue) const {
    ncmpiCheck(ncmpi_get_var1_long(groupId, myId,&index[0],datumValue),__FILE__,__LINE__);
}
// Reads a single datum value of a netCDF variable.
void NcmpiVar::getVar(const vector<MPI_Offset>& index, float* datumValue) const {
    ncmpiCheck(ncmpi_get_var1_float(groupId, myId,&index[0],datumValue),__FILE__,__LINE__);
}
// Reads a single datum value of a netCDF variable.
void NcmpiVar::getVar(const vector<MPI_Offset>& index, double* datumValue) const {
    ncmpiCheck(ncmpi_get_var1_double(groupId, myId,&index[0],datumValue),__FILE__,__LINE__);
}
// Reads a single datum value of a netCDF variable.
void NcmpiVar::getVar(const vector<MPI_Offset>& index, unsigned short* datumValue) const {
    ncmpiCheck(ncmpi_get_var1_ushort(groupId, myId,&index[0],datumValue),__FILE__,__LINE__);
}
// Reads a single datum value of a netCDF variable.
void NcmpiVar::getVar(const vector<MPI_Offset>& index, unsigned int* datumValue) const {
    ncmpiCheck(ncmpi_get_var1_uint(groupId, myId,&index[0],datumValue),__FILE__,__LINE__);
}
// Reads a single datum value of a netCDF variable.
void NcmpiVar::getVar(const vector<MPI_Offset>& index, long long* datumValue) const {
    ncmpiCheck(ncmpi_get_var1_longlong(groupId, myId,&index[0],datumValue),__FILE__,__LINE__);
}
// Reads a single datum value of a netCDF variable
void NcmpiVar::getVar(const vector<MPI_Offset>& index, unsigned long long* datumValue) const {
    ncmpiCheck(ncmpi_get_var1_ulonglong(groupId, myId,&index[0],datumValue),__FILE__,__LINE__);
}
// Reads a single datum value of a netCDF variable with no data conversion.
void NcmpiVar::getVar(const vector<MPI_Offset>& index, void* datumValue, MPI_Offset bufcount, MPI_Datatype buftype) const {
    ncmpiCheck(ncmpi_get_var1(groupId, myId,&index[0],datumValue, bufcount, buftype),__FILE__,__LINE__);
}

///////////

// Reads an array of values from  a netCDF variable.
void NcmpiVar::getVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, char* dataValues) const {
    ncmpiCheck(ncmpi_get_vara_text(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::getVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, unsigned char* dataValues) const {
    ncmpiCheck(ncmpi_get_vara_uchar(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::getVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, signed char* dataValues) const {
    ncmpiCheck(ncmpi_get_vara_schar(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::getVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, short* dataValues) const {
    ncmpiCheck(ncmpi_get_vara_short(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::getVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, int* dataValues) const {
    ncmpiCheck(ncmpi_get_vara_int(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::getVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, long* dataValues) const {
    ncmpiCheck(ncmpi_get_vara_long(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::getVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, float* dataValues) const {
    ncmpiCheck(ncmpi_get_vara_float(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::getVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, double* dataValues) const {
    ncmpiCheck(ncmpi_get_vara_double(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::getVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, unsigned short* dataValues) const {
    ncmpiCheck(ncmpi_get_vara_ushort(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::getVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, unsigned int* dataValues) const {
    ncmpiCheck(ncmpi_get_vara_uint(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::getVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, long long* dataValues) const {
    ncmpiCheck(ncmpi_get_vara_longlong(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable
void NcmpiVar::getVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, unsigned long long* dataValues) const {
    ncmpiCheck(ncmpi_get_vara_ulonglong(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable with no data conversion.
void NcmpiVar::getVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const {
    ncmpiCheck(ncmpi_get_vara(groupId, myId,&startp[0],&countp[0],dataValues, bufcount, buftype),__FILE__,__LINE__);
}


///////////

// Reads a subsampled (strided) array section of values from a netCDF variable.
void NcmpiVar::getVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, char* dataValues) const {
    ncmpiCheck(ncmpi_get_vars_text(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Reads a subsampled (strided) array section of values from a netCDF variable.
void NcmpiVar::getVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, unsigned char* dataValues) const {
    ncmpiCheck(ncmpi_get_vars_uchar(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Reads a subsampled (strided) array section of values from a netCDF variable.
void NcmpiVar::getVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, signed char* dataValues) const {
    ncmpiCheck(ncmpi_get_vars_schar(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Reads a subsampled (strided) array section of values from a netCDF variable.
void NcmpiVar::getVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, short* dataValues) const {
    ncmpiCheck(ncmpi_get_vars_short(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Reads a subsampled (strided) array section of values from a netCDF variable.
void NcmpiVar::getVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, int* dataValues) const {
    ncmpiCheck(ncmpi_get_vars_int(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Reads a subsampled (strided) array section of values from a netCDF variable.
void NcmpiVar::getVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, long* dataValues) const {
    ncmpiCheck(ncmpi_get_vars_long(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Reads a subsampled (strided) array section of values from a netCDF variable.
void NcmpiVar::getVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, float* dataValues) const {
    ncmpiCheck(ncmpi_get_vars_float(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Reads a subsampled (strided) array section of values from a netCDF variable.
void NcmpiVar::getVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, double* dataValues) const {
    ncmpiCheck(ncmpi_get_vars_double(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Reads a subsampled (strided) array section of values from a netCDF variable.
void NcmpiVar::getVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, unsigned short* dataValues) const {
    ncmpiCheck(ncmpi_get_vars_ushort(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Reads a subsampled (strided) array section of values from a netCDF variable.
void NcmpiVar::getVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, unsigned int* dataValues) const {
    ncmpiCheck(ncmpi_get_vars_uint(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Reads a subsampled (strided) array section of values from a netCDF variable.
void NcmpiVar::getVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, long long* dataValues) const {
    ncmpiCheck(ncmpi_get_vars_longlong(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Reads a subsampled (strided) array section of values from a netCDF variable
void NcmpiVar::getVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, unsigned long long* dataValues) const {
    ncmpiCheck(ncmpi_get_vars_ulonglong(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Reads a subsampled (strided) array section of values from a netCDF variable with no data conversion.
void NcmpiVar::getVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const {
    ncmpiCheck(ncmpi_get_vars(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues, bufcount, buftype),__FILE__,__LINE__);
}


///////////

// Reads a mapped array section of values from a netCDF variable.
void NcmpiVar::getVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, char* dataValues) const {
    ncmpiCheck(ncmpi_get_varm_text(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Reads a mapped array section of values from a netCDF variable.
void NcmpiVar::getVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, unsigned char* dataValues) const {
    ncmpiCheck(ncmpi_get_varm_uchar(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Reads a mapped array section of values from a netCDF variable.
void NcmpiVar::getVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, signed char* dataValues) const {
    ncmpiCheck(ncmpi_get_varm_schar(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Reads a mapped array section of values from a netCDF variable.
void NcmpiVar::getVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, short* dataValues) const {
    ncmpiCheck(ncmpi_get_varm_short(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Reads a mapped array section of values from a netCDF variable.
void NcmpiVar::getVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, int* dataValues) const {
    ncmpiCheck(ncmpi_get_varm_int(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Reads a mapped array section of values from a netCDF variable.
void NcmpiVar::getVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, long* dataValues) const {
    ncmpiCheck(ncmpi_get_varm_long(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Reads a mapped array section of values from a netCDF variable.
void NcmpiVar::getVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, float* dataValues) const {
    ncmpiCheck(ncmpi_get_varm_float(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Reads a mapped array section of values from a netCDF variable.
void NcmpiVar::getVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, double* dataValues) const {
    ncmpiCheck(ncmpi_get_varm_double(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Reads a mapped array section of values from a netCDF variable.
void NcmpiVar::getVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, unsigned short* dataValues) const {
    ncmpiCheck(ncmpi_get_varm_ushort(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Reads a mapped array section of values from a netCDF variable.
void NcmpiVar::getVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, unsigned int* dataValues) const {
    ncmpiCheck(ncmpi_get_varm_uint(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Reads a mapped array section of values from a netCDF variable.
void NcmpiVar::getVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, long long* dataValues) const {
    ncmpiCheck(ncmpi_get_varm_longlong(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Reads a mapped array section of values from a netCDF variable
void NcmpiVar::getVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, unsigned long long* dataValues) const {
    ncmpiCheck(ncmpi_get_varm_ulonglong(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Reads a mapped array section of values from a netCDF variable with no data conversion.
void NcmpiVar::getVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const {
    ncmpiCheck(ncmpi_get_varm(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues, bufcount, buftype),__FILE__,__LINE__);
}

//
// Data reading (collective data mode)
//

// Reads the entire data of the netCDF variable.
void NcmpiVar::getVar_all(char* dataValues) const {
    ncmpiCheck(ncmpi_get_var_text_all(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Reads the entire data of the netCDF variable.
void NcmpiVar::getVar_all(unsigned char* dataValues) const {
    ncmpiCheck(ncmpi_get_var_uchar_all(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Reads the entire data of the netCDF variable.
void NcmpiVar::getVar_all(signed char* dataValues) const {
    ncmpiCheck(ncmpi_get_var_schar_all(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Reads the entire data of the netCDF variable.
void NcmpiVar::getVar_all(short* dataValues) const {
    ncmpiCheck(ncmpi_get_var_short_all(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Reads the entire data of the netCDF variable.
void NcmpiVar::getVar_all(int* dataValues) const {
    ncmpiCheck(ncmpi_get_var_int_all(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Reads the entire data of the netCDF variable.
void NcmpiVar::getVar_all(long* dataValues) const {
    ncmpiCheck(ncmpi_get_var_long_all(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Reads the entire data of the netCDF variable.
void NcmpiVar::getVar_all(float* dataValues) const {
    ncmpiCheck(ncmpi_get_var_float_all(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Reads the entire data of the netCDF variable.
void NcmpiVar::getVar_all(double* dataValues) const {
    ncmpiCheck(ncmpi_get_var_double_all(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Reads the entire data of the netCDF variable.
void NcmpiVar::getVar_all(unsigned short* dataValues) const {
    ncmpiCheck(ncmpi_get_var_ushort_all(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Reads the entire data of the netCDF variable.
void NcmpiVar::getVar_all(unsigned int* dataValues) const {
    ncmpiCheck(ncmpi_get_var_uint_all(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Reads the entire data of the netCDF variable.
void NcmpiVar::getVar_all(long long* dataValues) const {
    ncmpiCheck(ncmpi_get_var_longlong_all(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Reads the entire data of the netCDF variable.
void NcmpiVar::getVar_all(unsigned long long* dataValues) const {
    ncmpiCheck(ncmpi_get_var_ulonglong_all(groupId, myId,dataValues),__FILE__,__LINE__);
}
// Reads the entire data of the netCDF variable with no data conversion.
void NcmpiVar::getVar_all(void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const {
    ncmpiCheck(ncmpi_get_var_all(groupId, myId,dataValues, bufcount, buftype),__FILE__,__LINE__);
}


///////////

// Reads a single datum value of a netCDF variable.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& index, char* datumValue) const {
    ncmpiCheck(ncmpi_get_var1_text_all(groupId, myId,&index[0],datumValue),__FILE__,__LINE__);
}
// Reads a single datum value of a netCDF variable.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& index, unsigned char* datumValue) const {
    ncmpiCheck(ncmpi_get_var1_uchar_all(groupId, myId,&index[0],datumValue),__FILE__,__LINE__);
}
// Reads a single datum value of a netCDF variable.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& index, signed char* datumValue) const {
    ncmpiCheck(ncmpi_get_var1_schar_all(groupId, myId,&index[0],datumValue),__FILE__,__LINE__);
}
// Reads a single datum value of a netCDF variable.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& index, short* datumValue) const {
    ncmpiCheck(ncmpi_get_var1_short_all(groupId, myId,&index[0],datumValue),__FILE__,__LINE__);
}
// Reads a single datum value of a netCDF variable.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& index, int* datumValue) const {
    ncmpiCheck(ncmpi_get_var1_int_all(groupId, myId,&index[0],datumValue),__FILE__,__LINE__);
}
// Reads a single datum value of a netCDF variable.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& index, long* datumValue) const {
    ncmpiCheck(ncmpi_get_var1_long_all(groupId, myId,&index[0],datumValue),__FILE__,__LINE__);
}
// Reads a single datum value of a netCDF variable.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& index, float* datumValue) const {
    ncmpiCheck(ncmpi_get_var1_float_all(groupId, myId,&index[0],datumValue),__FILE__,__LINE__);
}
// Reads a single datum value of a netCDF variable.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& index, double* datumValue) const {
    ncmpiCheck(ncmpi_get_var1_double_all(groupId, myId,&index[0],datumValue),__FILE__,__LINE__);
}
// Reads a single datum value of a netCDF variable.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& index, unsigned short* datumValue) const {
    ncmpiCheck(ncmpi_get_var1_ushort_all(groupId, myId,&index[0],datumValue),__FILE__,__LINE__);
}
// Reads a single datum value of a netCDF variable.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& index, unsigned int* datumValue) const {
    ncmpiCheck(ncmpi_get_var1_uint_all(groupId, myId,&index[0],datumValue),__FILE__,__LINE__);
}
// Reads a single datum value of a netCDF variable.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& index, long long* datumValue) const {
    ncmpiCheck(ncmpi_get_var1_longlong_all(groupId, myId,&index[0],datumValue),__FILE__,__LINE__);
}
// Reads a single datum value of a netCDF variable
void NcmpiVar::getVar_all(const vector<MPI_Offset>& index, unsigned long long* datumValue) const {
    ncmpiCheck(ncmpi_get_var1_ulonglong_all(groupId, myId,&index[0],datumValue),__FILE__,__LINE__);
}
// Reads a single datum value of a netCDF variable with no data conversion.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& index, void* datumValue, MPI_Offset bufcount, MPI_Datatype buftype) const {
    ncmpiCheck(ncmpi_get_var1_all(groupId, myId,&index[0],datumValue, bufcount, buftype),__FILE__,__LINE__);
}

///////////

// Reads an array of values from  a netCDF variable.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, char* dataValues) const {
    ncmpiCheck(ncmpi_get_vara_text_all(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, unsigned char* dataValues) const {
    ncmpiCheck(ncmpi_get_vara_uchar_all(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, signed char* dataValues) const {
    ncmpiCheck(ncmpi_get_vara_schar_all(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, short* dataValues) const {
    ncmpiCheck(ncmpi_get_vara_short_all(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, int* dataValues) const {
    ncmpiCheck(ncmpi_get_vara_int_all(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, long* dataValues) const {
    ncmpiCheck(ncmpi_get_vara_long_all(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, float* dataValues) const {
    ncmpiCheck(ncmpi_get_vara_float_all(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, double* dataValues) const {
    ncmpiCheck(ncmpi_get_vara_double_all(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, unsigned short* dataValues) const {
    ncmpiCheck(ncmpi_get_vara_ushort_all(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, unsigned int* dataValues) const {
    ncmpiCheck(ncmpi_get_vara_uint_all(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, long long* dataValues) const {
    ncmpiCheck(ncmpi_get_vara_longlong_all(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable
void NcmpiVar::getVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, unsigned long long* dataValues) const {
    ncmpiCheck(ncmpi_get_vara_ulonglong_all(groupId, myId,&startp[0],&countp[0],dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable with no data conversion.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const {
    ncmpiCheck(ncmpi_get_vara_all(groupId, myId,&startp[0],&countp[0],dataValues, bufcount, buftype),__FILE__,__LINE__);
}


///////////

// Reads a subsampled (strided) array section of values from a netCDF variable.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, char* dataValues) const {
    ncmpiCheck(ncmpi_get_vars_text_all(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Reads a subsampled (strided) array section of values from a netCDF variable.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, unsigned char* dataValues) const {
    ncmpiCheck(ncmpi_get_vars_uchar_all(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Reads a subsampled (strided) array section of values from a netCDF variable.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, signed char* dataValues) const {
    ncmpiCheck(ncmpi_get_vars_schar_all(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Reads a subsampled (strided) array section of values from a netCDF variable.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, short* dataValues) const {
    ncmpiCheck(ncmpi_get_vars_short_all(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Reads a subsampled (strided) array section of values from a netCDF variable.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, int* dataValues) const {
    ncmpiCheck(ncmpi_get_vars_int_all(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Reads a subsampled (strided) array section of values from a netCDF variable.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, long* dataValues) const {
    ncmpiCheck(ncmpi_get_vars_long_all(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Reads a subsampled (strided) array section of values from a netCDF variable.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, float* dataValues) const {
    ncmpiCheck(ncmpi_get_vars_float_all(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Reads a subsampled (strided) array section of values from a netCDF variable.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, double* dataValues) const {
    ncmpiCheck(ncmpi_get_vars_double_all(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Reads a subsampled (strided) array section of values from a netCDF variable.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, unsigned short* dataValues) const {
    ncmpiCheck(ncmpi_get_vars_ushort_all(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Reads a subsampled (strided) array section of values from a netCDF variable.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, unsigned int* dataValues) const {
    ncmpiCheck(ncmpi_get_vars_uint_all(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Reads a subsampled (strided) array section of values from a netCDF variable.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, long long* dataValues) const {
    ncmpiCheck(ncmpi_get_vars_longlong_all(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Reads a subsampled (strided) array section of values from a netCDF variable
void NcmpiVar::getVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, unsigned long long* dataValues) const {
    ncmpiCheck(ncmpi_get_vars_ulonglong_all(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues),__FILE__,__LINE__);
}
// Reads a subsampled (strided) array section of values from a netCDF variable with no data conversion.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const {
    ncmpiCheck(ncmpi_get_vars_all(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues, bufcount, buftype),__FILE__,__LINE__);
}


///////////

// Reads a mapped array section of values from a netCDF variable.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, char* dataValues) const {
    ncmpiCheck(ncmpi_get_varm_text_all(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Reads a mapped array section of values from a netCDF variable.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, unsigned char* dataValues) const {
    ncmpiCheck(ncmpi_get_varm_uchar_all(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Reads a mapped array section of values from a netCDF variable.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, signed char* dataValues) const {
    ncmpiCheck(ncmpi_get_varm_schar_all(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Reads a mapped array section of values from a netCDF variable.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, short* dataValues) const {
    ncmpiCheck(ncmpi_get_varm_short_all(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Reads a mapped array section of values from a netCDF variable.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, int* dataValues) const {
    ncmpiCheck(ncmpi_get_varm_int_all(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Reads a mapped array section of values from a netCDF variable.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, long* dataValues) const {
    ncmpiCheck(ncmpi_get_varm_long_all(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Reads a mapped array section of values from a netCDF variable.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, float* dataValues) const {
    ncmpiCheck(ncmpi_get_varm_float_all(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Reads a mapped array section of values from a netCDF variable.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, double* dataValues) const {
    ncmpiCheck(ncmpi_get_varm_double_all(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Reads a mapped array section of values from a netCDF variable.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, unsigned short* dataValues) const {
    ncmpiCheck(ncmpi_get_varm_ushort_all(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Reads a mapped array section of values from a netCDF variable.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, unsigned int* dataValues) const {
    ncmpiCheck(ncmpi_get_varm_uint_all(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Reads a mapped array section of values from a netCDF variable.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, long long* dataValues) const {
    ncmpiCheck(ncmpi_get_varm_longlong_all(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Reads a mapped array section of values from a netCDF variable
void NcmpiVar::getVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, unsigned long long* dataValues) const {
    ncmpiCheck(ncmpi_get_varm_ulonglong_all(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues),__FILE__,__LINE__);
}
// Reads a mapped array section of values from a netCDF variable with no data conversion.
void NcmpiVar::getVar_all(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const {
    ncmpiCheck(ncmpi_get_varm_all(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues, bufcount, buftype),__FILE__,__LINE__);
}

//////////////////////

// Reads a list of subarrays from  a netCDF variable. (independent I/O APIs)
void NcmpiVar::getVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], char* dataValues) const {
    ncmpiCheck(ncmpi_get_varn_text(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::getVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], unsigned char* dataValues) const {
    ncmpiCheck(ncmpi_get_varn_uchar(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::getVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], signed char* dataValues) const {
    ncmpiCheck(ncmpi_get_varn_schar(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::getVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], short* dataValues) const {
    ncmpiCheck(ncmpi_get_varn_short(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::getVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], int* dataValues) const {
    ncmpiCheck(ncmpi_get_varn_int(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::getVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], long* dataValues) const {
    ncmpiCheck(ncmpi_get_varn_long(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::getVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], float* dataValues) const {
    ncmpiCheck(ncmpi_get_varn_float(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::getVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], double* dataValues) const {
    ncmpiCheck(ncmpi_get_varn_double(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::getVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], unsigned short* dataValues) const {
    ncmpiCheck(ncmpi_get_varn_ushort(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::getVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], unsigned int* dataValues) const {
    ncmpiCheck(ncmpi_get_varn_uint(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::getVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], long long* dataValues) const {
    ncmpiCheck(ncmpi_get_varn_longlong(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable
void NcmpiVar::getVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], unsigned long long* dataValues) const {
    ncmpiCheck(ncmpi_get_varn_ulonglong(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable with no data conversion.
void NcmpiVar::getVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const {
    ncmpiCheck(ncmpi_get_varn(groupId, myId, num, starts, counts, dataValues, bufcount, buftype),__FILE__,__LINE__);
}

//////////////////////

// Reads a list of subarrays from  a netCDF variable. (collective I/O APIs)
void NcmpiVar::getVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], char* dataValues) const {
    ncmpiCheck(ncmpi_get_varn_text_all(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::getVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], unsigned char* dataValues) const {
    ncmpiCheck(ncmpi_get_varn_uchar_all(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::getVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], signed char* dataValues) const {
    ncmpiCheck(ncmpi_get_varn_schar_all(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::getVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], short* dataValues) const {
    ncmpiCheck(ncmpi_get_varn_short_all(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::getVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], int* dataValues) const {
    ncmpiCheck(ncmpi_get_varn_int_all(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::getVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], long* dataValues) const {
    ncmpiCheck(ncmpi_get_varn_long_all(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::getVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], float* dataValues) const {
    ncmpiCheck(ncmpi_get_varn_float_all(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::getVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], double* dataValues) const {
    ncmpiCheck(ncmpi_get_varn_double_all(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::getVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], unsigned short* dataValues) const {
    ncmpiCheck(ncmpi_get_varn_ushort_all(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::getVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], unsigned int* dataValues) const {
    ncmpiCheck(ncmpi_get_varn_uint_all(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::getVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], long long* dataValues) const {
    ncmpiCheck(ncmpi_get_varn_longlong_all(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable
void NcmpiVar::getVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], unsigned long long* dataValues) const {
    ncmpiCheck(ncmpi_get_varn_ulonglong_all(groupId, myId, num, starts, counts, dataValues),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable with no data conversion.
void NcmpiVar::getVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const {
    ncmpiCheck(ncmpi_get_varn_all(groupId, myId, num, starts, counts, dataValues, bufcount, buftype),__FILE__,__LINE__);
}


// Reads an array of values from a netCDF variable with filetype and buftype.
void NcmpiVar::getVard(MPI_Datatype filetype, void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const {
    ncmpiCheck(ncmpi_get_vard(groupId, myId, filetype, dataValues, bufcount, buftype),__FILE__,__LINE__);
}

// Reads an array of values from a netCDF variable with filetype and buftype.
void NcmpiVar::getVard_all(MPI_Datatype filetype, void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const {
    ncmpiCheck(ncmpi_get_vard_all(groupId, myId, filetype, dataValues, bufcount, buftype),__FILE__,__LINE__);
}

// Nonblocking data reading

// Reads the entire data of the netCDF variable.
void NcmpiVar::igetVar(char* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_var_text(groupId, myId,dataValues, req),__FILE__,__LINE__);
}
// Reads the entire data of the netCDF variable.
void NcmpiVar::igetVar(unsigned char* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_var_uchar(groupId, myId,dataValues, req),__FILE__,__LINE__);
}
// Reads the entire data of the netCDF variable.
void NcmpiVar::igetVar(signed char* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_var_schar(groupId, myId,dataValues, req),__FILE__,__LINE__);
}
// Reads the entire data of the netCDF variable.
void NcmpiVar::igetVar(short* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_var_short(groupId, myId,dataValues, req),__FILE__,__LINE__);
}
// Reads the entire data of the netCDF variable.
void NcmpiVar::igetVar(int* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_var_int(groupId, myId,dataValues, req),__FILE__,__LINE__);
}
// Reads the entire data of the netCDF variable.
void NcmpiVar::igetVar(long* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_var_long(groupId, myId,dataValues, req),__FILE__,__LINE__);
}
// Reads the entire data of the netCDF variable.
void NcmpiVar::igetVar(float* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_var_float(groupId, myId,dataValues, req),__FILE__,__LINE__);
}
// Reads the entire data of the netCDF variable.
void NcmpiVar::igetVar(double* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_var_double(groupId, myId,dataValues, req),__FILE__,__LINE__);
}
// Reads the entire data of the netCDF variable.
void NcmpiVar::igetVar(unsigned short* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_var_ushort(groupId, myId,dataValues, req),__FILE__,__LINE__);
}
// Reads the entire data of the netCDF variable.
void NcmpiVar::igetVar(unsigned int* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_var_uint(groupId, myId,dataValues, req),__FILE__,__LINE__);
}
// Reads the entire data of the netCDF variable.
void NcmpiVar::igetVar(long long* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_var_longlong(groupId, myId,dataValues, req),__FILE__,__LINE__);
}
// Reads the entire data of the netCDF variable.
void NcmpiVar::igetVar(unsigned long long* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_var_ulonglong(groupId, myId,dataValues, req),__FILE__,__LINE__);
}
// Reads the entire data of the netCDF variable with no data conversion.
void NcmpiVar::igetVar(void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype, int *req) const {
    ncmpiCheck(ncmpi_iget_var(groupId, myId,dataValues, bufcount, buftype, req),__FILE__,__LINE__);
}



///////////

// Reads a single datum value of a netCDF variable.
void NcmpiVar::igetVar(const vector<MPI_Offset>& index, char* datumValue, int *req) const {
    ncmpiCheck(ncmpi_iget_var1_text(groupId, myId,&index[0],datumValue, req),__FILE__,__LINE__);
}
// Reads a single datum value of a netCDF variable.
void NcmpiVar::igetVar(const vector<MPI_Offset>& index, unsigned char* datumValue, int *req) const {
    ncmpiCheck(ncmpi_iget_var1_uchar(groupId, myId,&index[0],datumValue, req),__FILE__,__LINE__);
}
// Reads a single datum value of a netCDF variable.
void NcmpiVar::igetVar(const vector<MPI_Offset>& index, signed char* datumValue, int *req) const {
    ncmpiCheck(ncmpi_iget_var1_schar(groupId, myId,&index[0],datumValue, req),__FILE__,__LINE__);
}
// Reads a single datum value of a netCDF variable.
void NcmpiVar::igetVar(const vector<MPI_Offset>& index, short* datumValue, int *req) const {
    ncmpiCheck(ncmpi_iget_var1_short(groupId, myId,&index[0],datumValue, req),__FILE__,__LINE__);
}
// Reads a single datum value of a netCDF variable.
void NcmpiVar::igetVar(const vector<MPI_Offset>& index, int* datumValue, int *req) const {
    ncmpiCheck(ncmpi_iget_var1_int(groupId, myId,&index[0],datumValue, req),__FILE__,__LINE__);
}
// Reads a single datum value of a netCDF variable.
void NcmpiVar::igetVar(const vector<MPI_Offset>& index, long* datumValue, int *req) const {
    ncmpiCheck(ncmpi_iget_var1_long(groupId, myId,&index[0],datumValue, req),__FILE__,__LINE__);
}
// Reads a single datum value of a netCDF variable.
void NcmpiVar::igetVar(const vector<MPI_Offset>& index, float* datumValue, int *req) const {
    ncmpiCheck(ncmpi_iget_var1_float(groupId, myId,&index[0],datumValue, req),__FILE__,__LINE__);
}
// Reads a single datum value of a netCDF variable.
void NcmpiVar::igetVar(const vector<MPI_Offset>& index, double* datumValue, int *req) const {
    ncmpiCheck(ncmpi_iget_var1_double(groupId, myId,&index[0],datumValue, req),__FILE__,__LINE__);
}
// Reads a single datum value of a netCDF variable.
void NcmpiVar::igetVar(const vector<MPI_Offset>& index, unsigned short* datumValue, int *req) const {
    ncmpiCheck(ncmpi_iget_var1_ushort(groupId, myId,&index[0],datumValue, req),__FILE__,__LINE__);
}
// Reads a single datum value of a netCDF variable.
void NcmpiVar::igetVar(const vector<MPI_Offset>& index, unsigned int* datumValue, int *req) const {
    ncmpiCheck(ncmpi_iget_var1_uint(groupId, myId,&index[0],datumValue, req),__FILE__,__LINE__);
}
// Reads a single datum value of a netCDF variable.
void NcmpiVar::igetVar(const vector<MPI_Offset>& index, long long* datumValue, int *req) const {
    ncmpiCheck(ncmpi_iget_var1_longlong(groupId, myId,&index[0],datumValue, req),__FILE__,__LINE__);
}
// Reads a single datum value of a netCDF variable
void NcmpiVar::igetVar(const vector<MPI_Offset>& index, unsigned long long* datumValue, int *req) const {
    ncmpiCheck(ncmpi_iget_var1_ulonglong(groupId, myId,&index[0],datumValue, req),__FILE__,__LINE__);
}
// Reads a single datum value of a netCDF variable with no data conversion.
void NcmpiVar::igetVar(const vector<MPI_Offset>& index, void* datumValue, MPI_Offset bufcount, MPI_Datatype buftype, int *req) const {
    ncmpiCheck(ncmpi_iget_var1(groupId, myId,&index[0],datumValue, bufcount, buftype, req),__FILE__,__LINE__);
}



///////////

// Reads an array of values from  a netCDF variable.
void NcmpiVar::igetVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, char* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_vara_text(groupId, myId,&startp[0],&countp[0],dataValues, req),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::igetVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, unsigned char* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_vara_uchar(groupId, myId,&startp[0],&countp[0],dataValues, req),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::igetVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, signed char* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_vara_schar(groupId, myId,&startp[0],&countp[0],dataValues, req),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::igetVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, short* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_vara_short(groupId, myId,&startp[0],&countp[0],dataValues, req),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::igetVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, int* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_vara_int(groupId, myId,&startp[0],&countp[0],dataValues, req),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::igetVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, long* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_vara_long(groupId, myId,&startp[0],&countp[0],dataValues, req),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::igetVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, float* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_vara_float(groupId, myId,&startp[0],&countp[0],dataValues, req),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::igetVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, double* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_vara_double(groupId, myId,&startp[0],&countp[0],dataValues, req),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::igetVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, unsigned short* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_vara_ushort(groupId, myId,&startp[0],&countp[0],dataValues, req),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::igetVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, unsigned int* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_vara_uint(groupId, myId,&startp[0],&countp[0],dataValues, req),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::igetVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, long long* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_vara_longlong(groupId, myId,&startp[0],&countp[0],dataValues, req),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable
void NcmpiVar::igetVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, unsigned long long* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_vara_ulonglong(groupId, myId,&startp[0],&countp[0],dataValues, req),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable with no data conversion.
void NcmpiVar::igetVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype, int *req) const {
    ncmpiCheck(ncmpi_iget_vara(groupId, myId,&startp[0],&countp[0],dataValues, bufcount, buftype, req),__FILE__,__LINE__);
}


///////////

// Reads a subsampled (strided) array section of values from a netCDF variable.
void NcmpiVar::igetVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, char* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_vars_text(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues, req),__FILE__,__LINE__);
}
// Reads a subsampled (strided) array section of values from a netCDF variable.
void NcmpiVar::igetVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, unsigned char* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_vars_uchar(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues, req),__FILE__,__LINE__);
}
// Reads a subsampled (strided) array section of values from a netCDF variable.
void NcmpiVar::igetVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, signed char* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_vars_schar(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues, req),__FILE__,__LINE__);
}
// Reads a subsampled (strided) array section of values from a netCDF variable.
void NcmpiVar::igetVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, short* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_vars_short(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues, req),__FILE__,__LINE__);
}
// Reads a subsampled (strided) array section of values from a netCDF variable.
void NcmpiVar::igetVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, int* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_vars_int(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues, req),__FILE__,__LINE__);
}
// Reads a subsampled (strided) array section of values from a netCDF variable.
void NcmpiVar::igetVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, long* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_vars_long(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues, req),__FILE__,__LINE__);
}
// Reads a subsampled (strided) array section of values from a netCDF variable.
void NcmpiVar::igetVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, float* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_vars_float(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues, req),__FILE__,__LINE__);
}
// Reads a subsampled (strided) array section of values from a netCDF variable.
void NcmpiVar::igetVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, double* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_vars_double(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues, req),__FILE__,__LINE__);
}
// Reads a subsampled (strided) array section of values from a netCDF variable.
void NcmpiVar::igetVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, unsigned short* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_vars_ushort(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues, req),__FILE__,__LINE__);
}
// Reads a subsampled (strided) array section of values from a netCDF variable.
void NcmpiVar::igetVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, unsigned int* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_vars_uint(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues, req),__FILE__,__LINE__);
}
// Reads a subsampled (strided) array section of values from a netCDF variable.
void NcmpiVar::igetVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, long long* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_vars_longlong(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues, req),__FILE__,__LINE__);
}
// Reads a subsampled (strided) array section of values from a netCDF variable
void NcmpiVar::igetVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, unsigned long long* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_vars_ulonglong(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues, req),__FILE__,__LINE__);
}
// Reads a subsampled (strided) array section of values from a netCDF variable with no data conversion.
void NcmpiVar::igetVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype, int *req) const {
    ncmpiCheck(ncmpi_iget_vars(groupId, myId,&startp[0],&countp[0],&stridep[0],dataValues, bufcount, buftype, req),__FILE__,__LINE__);
}


///////////

// Reads a mapped array section of values from a netCDF variable.
void NcmpiVar::igetVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, char* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_varm_text(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues, req),__FILE__,__LINE__);
}
// Reads a mapped array section of values from a netCDF variable.
void NcmpiVar::igetVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, unsigned char* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_varm_uchar(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues, req),__FILE__,__LINE__);
}
// Reads a mapped array section of values from a netCDF variable.
void NcmpiVar::igetVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, signed char* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_varm_schar(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues, req),__FILE__,__LINE__);
}
// Reads a mapped array section of values from a netCDF variable.
void NcmpiVar::igetVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, short* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_varm_short(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues, req),__FILE__,__LINE__);
}
// Reads a mapped array section of values from a netCDF variable.
void NcmpiVar::igetVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, int* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_varm_int(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues, req),__FILE__,__LINE__);
}
// Reads a mapped array section of values from a netCDF variable.
void NcmpiVar::igetVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, long* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_varm_long(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues, req),__FILE__,__LINE__);
}
// Reads a mapped array section of values from a netCDF variable.
void NcmpiVar::igetVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, float* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_varm_float(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues, req),__FILE__,__LINE__);
}
// Reads a mapped array section of values from a netCDF variable.
void NcmpiVar::igetVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, double* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_varm_double(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues, req),__FILE__,__LINE__);
}
// Reads a mapped array section of values from a netCDF variable.
void NcmpiVar::igetVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, unsigned short* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_varm_ushort(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues, req),__FILE__,__LINE__);
}
// Reads a mapped array section of values from a netCDF variable.
void NcmpiVar::igetVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, unsigned int* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_varm_uint(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues, req),__FILE__,__LINE__);
}
// Reads a mapped array section of values from a netCDF variable.
void NcmpiVar::igetVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, long long* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_varm_longlong(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues, req),__FILE__,__LINE__);
}
// Reads a mapped array section of values from a netCDF variable
void NcmpiVar::igetVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, unsigned long long* dataValues, int *req) const {
    ncmpiCheck(ncmpi_iget_varm_ulonglong(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues, req),__FILE__,__LINE__);
}
// Reads a mapped array section of values from a netCDF variable with no data conversion.
void NcmpiVar::igetVar(const vector<MPI_Offset>& startp, const vector<MPI_Offset>& countp, const vector<MPI_Offset>& stridep, const vector<MPI_Offset>& imapp, void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype, int *req) const {
    ncmpiCheck(ncmpi_iget_varm(groupId, myId,&startp[0],&countp[0],&stridep[0],&imapp[0],dataValues, bufcount, buftype, req),__FILE__,__LINE__);
}

//////////////////////

// Reads a list of subarrays from  a netCDF variable. (independent I/O APIs)
void NcmpiVar::igetVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], char* dataValues, int*req) const {
    ncmpiCheck(ncmpi_iget_varn_text(groupId, myId, num, starts, counts, dataValues, req),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::igetVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], unsigned char* dataValues, int*req) const {
    ncmpiCheck(ncmpi_iget_varn_uchar(groupId, myId, num, starts, counts, dataValues, req),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::igetVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], signed char* dataValues, int*req) const {
    ncmpiCheck(ncmpi_iget_varn_schar(groupId, myId, num, starts, counts, dataValues, req),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::igetVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], short* dataValues, int*req) const {
    ncmpiCheck(ncmpi_iget_varn_short(groupId, myId, num, starts, counts, dataValues, req),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::igetVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], int* dataValues, int*req) const {
    ncmpiCheck(ncmpi_iget_varn_int(groupId, myId, num, starts, counts, dataValues, req),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::igetVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], long* dataValues, int*req) const {
    ncmpiCheck(ncmpi_iget_varn_long(groupId, myId, num, starts, counts, dataValues, req),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::igetVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], float* dataValues, int*req) const {
    ncmpiCheck(ncmpi_iget_varn_float(groupId, myId, num, starts, counts, dataValues, req),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::igetVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], double* dataValues, int*req) const {
    ncmpiCheck(ncmpi_iget_varn_double(groupId, myId, num, starts, counts, dataValues, req),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::igetVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], unsigned short* dataValues, int*req) const {
    ncmpiCheck(ncmpi_iget_varn_ushort(groupId, myId, num, starts, counts, dataValues, req),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::igetVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], unsigned int* dataValues, int*req) const {
    ncmpiCheck(ncmpi_iget_varn_uint(groupId, myId, num, starts, counts, dataValues, req),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable.
void NcmpiVar::igetVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], long long* dataValues, int*req) const {
    ncmpiCheck(ncmpi_iget_varn_longlong(groupId, myId, num, starts, counts, dataValues, req),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable
void NcmpiVar::igetVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], unsigned long long* dataValues, int*req) const {
    ncmpiCheck(ncmpi_iget_varn_ulonglong(groupId, myId, num, starts, counts, dataValues, req),__FILE__,__LINE__);
}
// Reads an array of values from  a netCDF variable with no data conversion.
void NcmpiVar::igetVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype, int*req) const {
    ncmpiCheck(ncmpi_iget_varn(groupId, myId, num, starts, counts, dataValues, bufcount, buftype, req),__FILE__,__LINE__);
}

//////////////////////

void NcmpiVar::Inq_file_offset(MPI_Offset *offset)
{
    ncmpiCheck(ncmpi_inq_varoffset(groupId, myId, offset),__FILE__,__LINE__);
}


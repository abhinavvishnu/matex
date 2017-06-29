#include "ncmpiGroupAtt.h"
#include "ncmpiGroup.h"
#include "ncmpiCheck.h"
#include <pnetcdf.h>
using namespace std;


namespace PnetCDF {
  //  Global comparator operator ==============
  // comparator operator 
  bool operator<(const NcmpiGroupAtt& lhs,const NcmpiGroupAtt& rhs)
  {
    return false;
  }
  
  // comparator operator 
  bool operator>(const NcmpiGroupAtt& lhs,const NcmpiGroupAtt& rhs)
  {
    return true;
  }
}


using namespace PnetCDF;

// assignment operator
NcmpiGroupAtt& NcmpiGroupAtt::operator=(const NcmpiGroupAtt & rhs)
{
  NcmpiAtt::operator=(rhs);    // assign base class parts
  return *this;
}

//! The copy constructor.
NcmpiGroupAtt::NcmpiGroupAtt(const NcmpiGroupAtt& rhs): 
  NcmpiAtt(rhs)   // invoke base class copy constructor
{}


// Constructor generates a null object.
NcmpiGroupAtt::NcmpiGroupAtt() : 
  NcmpiAtt()  // invoke base class constructor
{}

// equivalence operator (doesn't bother compaing varid's of each object).
bool NcmpiGroupAtt::operator==(const NcmpiGroupAtt & rhs)
{
  if(nullObject) 
    return nullObject == rhs.isNull();
  else
    return myName == rhs.myName && groupId == rhs.groupId;
}  

// Constructor for an existing global attribute.
NcmpiGroupAtt::NcmpiGroupAtt(const NcmpiGroup& grp, const int index):
  NcmpiAtt(false)
{
  groupId =  grp.getId();
  varId = NC_GLOBAL;
  // get the name of this attribute
  char attName[NC_MAX_NAME+1];
  ncmpiCheck(ncmpi_inq_attname(groupId,varId, index, attName),__FILE__,__LINE__);
  ncmpiCheck(ncmpi_inq_attname(groupId,varId,index,attName),__FILE__,__LINE__);
  myName = attName;
}


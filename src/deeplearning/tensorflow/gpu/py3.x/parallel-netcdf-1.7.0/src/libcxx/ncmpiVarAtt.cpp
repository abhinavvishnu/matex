#include "ncmpiVar.h"
#include "ncmpiVarAtt.h"
#include "ncmpiGroup.h"
#include "ncmpiCheck.h"
#include <pnetcdf.h>
using namespace std;


namespace PnetCDF {
  //  Global comparator operator ==============
  // comparator operator 
  bool operator<(const NcmpiVarAtt& lhs,const NcmpiVarAtt& rhs)
  {
    return false;
  }
  
  // comparator operator 
  bool operator>(const NcmpiVarAtt& lhs,const NcmpiVarAtt& rhs)
  {
    return true;
  }
}


using namespace PnetCDF;


// assignment operator
NcmpiVarAtt& NcmpiVarAtt::operator=(const NcmpiVarAtt & rhs)
{
  NcmpiAtt::operator=(rhs);    // assign base class parts
  return *this;
}

//! The copy constructor.
NcmpiVarAtt::NcmpiVarAtt(const NcmpiVarAtt& rhs): 
  NcmpiAtt(rhs) // invoke base class copy constructor
{}


// Constructor generates a null object.
NcmpiVarAtt::NcmpiVarAtt() :
  NcmpiAtt()  // invoke base class constructor
{}


// Constructor for an existing local attribute.
NcmpiVarAtt::NcmpiVarAtt(const NcmpiGroup& grp, const NcmpiVar& ncmpiVar, const int index):
  NcmpiAtt(false)
{
  groupId =  grp.getId();
  varId = ncmpiVar.getId();
  // get the name of this attribute
  char attName[NC_MAX_NAME+1];
  ncmpiCheck(ncmpi_inq_attname(groupId,varId, index, attName),__FILE__,__LINE__);
  ncmpiCheck(ncmpi_inq_attname(groupId,varId,index,attName),__FILE__,__LINE__);
  myName = attName;
}

// Returns the NcmpiVar parent object.
NcmpiVar NcmpiVarAtt::getParentVar() const {
  return NcmpiVar(groupId,varId);
}

#include "ncmpiGroup.h"
#include "ncmpiVar.h"
#include "ncmpiDim.h"
#include "ncmpiVlenType.h"
#include "ncmpiCompoundType.h"
#include "ncmpiOpaqueType.h"
#include "ncmpiGroupAtt.h"
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
#include "ncmpiCheck.h"
using namespace std;
using namespace PnetCDF::exceptions;

namespace PnetCDF {
  //  Global comparator operator ==============
  // comparator operator 
  bool operator<(const NcmpiGroup& lhs,const NcmpiGroup& rhs)
  {
    return false;
  }
  
  // comparator operator 
  bool operator>(const NcmpiGroup& lhs,const NcmpiGroup& rhs)
  {
    return true;
  }
}

using namespace PnetCDF;
    
/////////////////////////////////////////////

NcmpiGroup::~NcmpiGroup()
{
}

// Constructor generates a null object.
NcmpiGroup::NcmpiGroup() :
  nullObject(true)
{}

   
// constructor
NcmpiGroup::NcmpiGroup(const int groupId) :
  nullObject(false),
  myId(groupId)
{ }

// assignment operator
NcmpiGroup& NcmpiGroup::operator=(const NcmpiGroup & rhs)
{
  nullObject = rhs.nullObject;
  myId = rhs.myId;
  return *this;
}

// The copy constructor.
NcmpiGroup::NcmpiGroup(const NcmpiGroup& rhs): 
  nullObject(rhs.nullObject),
  myId(rhs.myId)
{}


// equivalence operator
bool NcmpiGroup::operator==(const NcmpiGroup & rhs) const
{
  if(nullObject) 
    return nullObject == rhs.nullObject;
  else
    return myId == rhs.myId;
}  
  
//  !=  operator
bool NcmpiGroup::operator!=(const NcmpiGroup & rhs) const
{
  return !(*this == rhs);
}  
  
  
// /////////////
// NcmpiGroup-related methods
// /////////////
  
// Get the group name.
string NcmpiGroup::getName(bool fullName) const {
  if(isNull()) throw NcNullGrp("Attempt to invoke NcmpiGroup::getName on a Null group",__FILE__,__LINE__);
  string groupName;
  if(fullName){
    // return full name of group with foward "/" separarating sub-groups.
    MPI_Offset lenp;
    ncmpiCheck(ncmpi_inq_grpname_len(myId,&lenp),__FILE__,__LINE__);
    char* charName= new char[lenp+1];
    ncmpiCheck(ncmpi_inq_grpname_full(myId,&lenp,charName),__FILE__,__LINE__);
    groupName = charName;
    delete charName;
  }
  else {
    // return the (local) name of this group.
    char charName[NC_MAX_NAME+1];
    // ncmpiCheck(ncmpi_inq_grpname(myId,charName),__FILE__,__LINE__);
    charName[0] = '/'; charName[1] = '\0';
    groupName = charName;
  }
  return groupName;
}

// returns true if this is the root group.
bool NcmpiGroup::isRootGroup()  const{
  bool result = getName() == "/";
  return result;
}
  
// Get the parent group.
NcmpiGroup NcmpiGroup::getParentGroup() const {
  if(isNull()) throw NcNullGrp("Attempt to invoke NcmpiGroup::getParentGroup on a Null group",__FILE__,__LINE__);
  try {
    int parentId;
    ncmpiCheck(ncmpi_inq_grp_parent(myId,&parentId),__FILE__,__LINE__);
    NcmpiGroup ncmpiGroupParent(parentId);
    return ncmpiGroupParent;
  }
  catch (NcEnoGrp& e) {
    // no group found, so return null group
    return NcmpiGroup();
  }
}
  
  
// Get the group id.
int  NcmpiGroup::getId() const {
  if(isNull()) throw NcNullGrp("Attempt to invoke NcmpiGroup::getId on a Null group",__FILE__,__LINE__);
  return myId;
}
  
// Get the number of NcmpiGroup objects.
int NcmpiGroup::getGroupCount(NcmpiGroup::GroupLocation location) const {
  if(isNull()) throw NcNullGrp("Attempt to invoke NcmpiGroup::getGroupCount on a Null group",__FILE__,__LINE__);
  // initialize group counter
  int ngroups=0;

  // record this group
  if(location == ParentsAndCurrentGrps || location == AllGrps) {
    ngroups ++;
  }

  // number of children in current group
  if(location == ChildrenGrps || location == AllChildrenGrps || location == AllGrps ) {
    int numgrps;
    int* ncids=NULL;
    ncmpiCheck(ncmpi_inq_grps(getId(), &numgrps,ncids),__FILE__,__LINE__);
    ngroups += numgrps;
  }

  // search in parent groups
  if(location == ParentsGrps || location == ParentsAndCurrentGrps || location == AllGrps ) {
    multimap<string,NcmpiGroup> groups(getGroups(ParentsGrps));
    ngroups += groups.size();
  }


  // get the number of all children that are childreof children
  if(location == ChildrenOfChildrenGrps || location == AllChildrenGrps || location == AllGrps ) {
    multimap<string,NcmpiGroup> groups(getGroups(ChildrenOfChildrenGrps));
    ngroups += groups.size();
  }

  return ngroups;
}

  
// Get the set of child NcmpiGroup objects.
multimap<std::string,NcmpiGroup> NcmpiGroup::getGroups(NcmpiGroup::GroupLocation location) const {
  if(isNull()) throw NcNullGrp("Attempt to invoke NcmpiGroup::getGroups on a Null group",__FILE__,__LINE__);
  // create a container to hold the NcmpiGroup's.
  multimap<string,NcmpiGroup> ncmpiGroups;

  // record this group
  if(location == ParentsAndCurrentGrps || location == AllGrps) {
    ncmpiGroups.insert(pair<const string,NcmpiGroup>(getName(),*this));
  }

  // the child groups of the current group
  if(location == ChildrenGrps || location == AllChildrenGrps || location == AllGrps ) {
    // get the number of groups
    int groupCount = getGroupCount();
    if (groupCount){
      vector<int> ncids(groupCount);
      int* numgrps=NULL;
      // now get the id of each NcmpiGroup and populate the ncmpiGroups container.
      ncmpiCheck(ncmpi_inq_grps(myId, numgrps,&ncids[0]),__FILE__,__LINE__);
      for(int i=0; i<groupCount;i++){
        NcmpiGroup tmpGroup(ncids[i]);
        ncmpiGroups.insert(pair<const string,NcmpiGroup>(tmpGroup.getName(),tmpGroup));
      }
    }
  }

  // search in parent groups.
  if(location == ParentsGrps || location == ParentsAndCurrentGrps || location == AllGrps ) {
    NcmpiGroup tmpGroup(*this); 
    if(!tmpGroup.isRootGroup()) {
      while(1) {
	const NcmpiGroup parentGroup(tmpGroup.getParentGroup());
	if(parentGroup.isNull()) break;
	ncmpiGroups.insert(pair<const string,NcmpiGroup>(parentGroup.getName(),parentGroup));
	tmpGroup=parentGroup;
      }
    }
  }
 
  // search in child groups of the children
  if(location == ChildrenOfChildrenGrps || location == AllChildrenGrps || location == AllGrps ) {
    multimap<string,NcmpiGroup>::iterator it;
    multimap<string,NcmpiGroup> groups(getGroups(ChildrenGrps));
    for (it=groups.begin();it!=groups.end();it++) {
      multimap<string,NcmpiGroup> childGroups(it->second.getGroups(AllChildrenGrps));
      ncmpiGroups.insert(childGroups.begin(),childGroups.end());
    }
  }
  
  return ncmpiGroups;
}
  
// Get the named child NcmpiGroup object.
NcmpiGroup NcmpiGroup::getGroup(const string& name,NcmpiGroup::GroupLocation location) const{
  if(isNull()) throw NcNullGrp("Attempt to invoke NcmpiGroup::getGroup on a Null group",__FILE__,__LINE__);
  multimap<string,NcmpiGroup> ncmpiGroups(getGroups(location));
  pair<multimap<string,NcmpiGroup>::iterator,multimap<string,NcmpiGroup>::iterator> ret;
  ret = ncmpiGroups.equal_range(name);
  if(ret.first == ret.second) 
    return NcmpiGroup();  // null group is returned
  else 
    return ret.first->second;
}

  

// Get all NcmpiGroup objects with a given name.
set<NcmpiGroup> NcmpiGroup::getGroups(const std::string& name,NcmpiGroup::GroupLocation location) const {
  if(isNull()) throw NcNullGrp("Attempt to invoke NcmpiGroup::getGroups on a Null group",__FILE__,__LINE__);
  // get the set of ncmpiGroups in this group and above.
  multimap<std::string,NcmpiGroup> ncmpiGroups(getGroups(location));
  pair<multimap<string,NcmpiGroup>::iterator,multimap<string,NcmpiGroup>::iterator> ret;
  multimap<string,NcmpiGroup>::iterator it;
  ret = ncmpiGroups.equal_range(name);
  set<NcmpiGroup> tmpGroup;
  for (it=ret.first; it!=ret.second; ++it) {
    tmpGroup.insert(it->second);
  }
  return tmpGroup;
}

// Add a new child NcmpiGroup object.
NcmpiGroup NcmpiGroup::addGroup(const string& name) const {
  if(isNull()) throw NcNullGrp("Attempt to invoke NcmpiGroup::addGroup on a Null group",__FILE__,__LINE__);
  int new_ncid;
  ncmpiCheck(ncmpi_def_grp(myId,const_cast<char*> (name.c_str()),&new_ncid),__FILE__,__LINE__);
  return NcmpiGroup(new_ncid);
}


  
// /////////////
// NcmpiVar-related accessors
// /////////////
  
// Get the number of NcmpiVar objects in this group.
int NcmpiGroup::getVarCount(NcmpiGroup::Location location) const {

  // search in current group.
  NcmpiGroup tmpGroup(*this); 
  int nvars=0;
  // search in current group
  if((location == ParentsAndCurrent || location == ChildrenAndCurrent || location == Current || location ==All) && !tmpGroup.isNull()) {
    ncmpiCheck(ncmpi_inq_nvars(tmpGroup.getId(), &nvars),__FILE__,__LINE__);
  }

  // search recursively in all parent groups.
  if(location == Parents || location == ParentsAndCurrent || location ==All) {
    tmpGroup=getParentGroup();
    while(!tmpGroup.isNull()) {
      int nvarsp;
      ncmpiCheck(ncmpi_inq_nvars(tmpGroup.getId(), &nvarsp),__FILE__,__LINE__);
      nvars += nvarsp;
      // continue loop with the parent.
      tmpGroup=tmpGroup.getParentGroup();
    }
  }

  // search recursively in all child groups
  if(location == ChildrenAndCurrent || location == Children || location == All) {
    multimap<string,NcmpiGroup>::iterator it;
    multimap<string,NcmpiGroup> groups(getGroups());
    for (it=groups.begin();it!=groups.end();it++) {
      nvars += it->second.getVarCount(ChildrenAndCurrent);
    }
  }
  return nvars;
}

// Get the number of record variable NcmpiVar objects in this group.
int NcmpiGroup::getRecVarCount(NcmpiGroup::Location location) const {

  // search in current group.
  NcmpiGroup tmpGroup(*this); 
  int nvars=0;
  // search in current group
  if((location == ParentsAndCurrent || location == ChildrenAndCurrent || location == Current || location ==All) && !tmpGroup.isNull()) {
    ncmpiCheck(ncmpi_inq_num_rec_vars(tmpGroup.getId(), &nvars),__FILE__,__LINE__);
  }

  // search recursively in all parent groups.
  if(location == Parents || location == ParentsAndCurrent || location ==All) {
    tmpGroup=getParentGroup();
    while(!tmpGroup.isNull()) {
      int nvarsp;
      ncmpiCheck(ncmpi_inq_num_rec_vars(tmpGroup.getId(), &nvarsp),__FILE__,__LINE__);
      nvars += nvarsp;
      // continue loop with the parent.
      tmpGroup=tmpGroup.getParentGroup();
    }
  }

  // search recursively in all child groups
  if(location == ChildrenAndCurrent || location == Children || location == All) {
    multimap<string,NcmpiGroup>::iterator it;
    multimap<string,NcmpiGroup> groups(getGroups());
    for (it=groups.begin();it!=groups.end();it++) {
      nvars += it->second.getRecVarCount(ChildrenAndCurrent);
    }
  }
  return nvars;
}

// Get the number of fixed-size variable NcmpiVar objects in this group.
int NcmpiGroup::getFixVarCount(NcmpiGroup::Location location) const {

  // search in current group.
  NcmpiGroup tmpGroup(*this); 
  int nvars=0;
  // search in current group
  if((location == ParentsAndCurrent || location == ChildrenAndCurrent || location == Current || location ==All) && !tmpGroup.isNull()) {
    ncmpiCheck(ncmpi_inq_num_fix_vars(tmpGroup.getId(), &nvars),__FILE__,__LINE__);
  }

  // search recursively in all parent groups.
  if(location == Parents || location == ParentsAndCurrent || location ==All) {
    tmpGroup=getParentGroup();
    while(!tmpGroup.isNull()) {
      int nvarsp;
      ncmpiCheck(ncmpi_inq_num_fix_vars(tmpGroup.getId(), &nvarsp),__FILE__,__LINE__);
      nvars += nvarsp;
      // continue loop with the parent.
      tmpGroup=tmpGroup.getParentGroup();
    }
  }

  // search recursively in all child groups
  if(location == ChildrenAndCurrent || location == Children || location == All) {
    multimap<string,NcmpiGroup>::iterator it;
    multimap<string,NcmpiGroup> groups(getGroups());
    for (it=groups.begin();it!=groups.end();it++) {
      nvars += it->second.getFixVarCount(ChildrenAndCurrent);
    }
  }
  return nvars;
}

// Get the size of record block, sum of single record of all record variables
MPI_Offset NcmpiGroup::getRecSize(NcmpiGroup::Location location) const {

  // search in current group.
  NcmpiGroup tmpGroup(*this); 
  MPI_Offset recsize=0;
  // search in current group
  if((location == ParentsAndCurrent || location == ChildrenAndCurrent || location == Current || location ==All) && !tmpGroup.isNull()) {
    ncmpiCheck(ncmpi_inq_recsize(tmpGroup.getId(), &recsize),__FILE__,__LINE__);
  }
  return recsize;
}

// Get the collection of NcmpiVar objects.
multimap<std::string,NcmpiVar> NcmpiGroup::getVars(NcmpiGroup::Location location) const {

  // create a container to hold the NcmpiVar's.
  multimap<string,NcmpiVar> ncmpiVars;

  // search in current group.
  NcmpiGroup tmpGroup(*this); 
  if((location == ParentsAndCurrent || location == ChildrenAndCurrent || location == Current || location ==All) && !tmpGroup.isNull()) {
    // get the number of variables.
    int varCount = getVarCount();
    if (varCount){
      // now get the name of each NcmpiVar object and populate the ncmpiVars container.
      // int* nvars=NULL;
      // vector<int> varids(varCount);
      // ncmpiCheck(ncmpi_inq_varids(myId, nvars,&varids[0]),__FILE__,__LINE__);
      for(int i=0; i<varCount;i++){
        // NcmpiVar tmpVar(*this,varids[i]);
        NcmpiVar tmpVar(*this,i);
        ncmpiVars.insert(pair<const string,NcmpiVar>(tmpVar.getName(),tmpVar));
      }
    }
  }

  
  // search recursively in all parent groups.
  if(location == Parents || location == ParentsAndCurrent || location ==All) {
    tmpGroup=getParentGroup();
    while(!tmpGroup.isNull()) {
      // get the number of variables
      int varCount = tmpGroup.getVarCount();
      if (varCount){
        // now get the name of each NcmpiVar object and populate the ncmpiVars container.
        int* nvars=NULL;
        vector<int> varids(varCount);
        ncmpiCheck(ncmpi_inq_varids(tmpGroup.getId(), nvars,&varids[0]),__FILE__,__LINE__);
        for(int i=0; i<varCount;i++){
          NcmpiVar tmpVar(tmpGroup,varids[i]);
          ncmpiVars.insert(pair<const string,NcmpiVar>(tmpVar.getName(),tmpVar));
        }
      }
      // continue loop with the parent.
      tmpGroup=tmpGroup.getParentGroup();
    }
  }

  // search recusively in all child groups.
  if(location == ChildrenAndCurrent || location == Children  || location == All ) {
    multimap<string,NcmpiGroup>::iterator it;
    multimap<string,NcmpiGroup> groups(getGroups());
    for (it=groups.begin();it!=groups.end();it++) {
      multimap<string,NcmpiVar> vars=it->second.getVars(ChildrenAndCurrent);
      ncmpiVars.insert(vars.begin(),vars.end());
    }
  }

  return ncmpiVars;
}
  

// Get all NcmpiVar objects with a given name.
set<NcmpiVar> NcmpiGroup::getVars(const string& name,NcmpiGroup::Location location) const {
  // get the set of ncmpiVars in this group and above.
  multimap<std::string,NcmpiVar> ncmpiVars(getVars(location));
  pair<multimap<string,NcmpiVar>::iterator,multimap<string,NcmpiVar>::iterator> ret;
  multimap<string,NcmpiVar>::iterator it;
  ret = ncmpiVars.equal_range(name);
  set<NcmpiVar> tmpVar;
  for (it=ret.first; it!=ret.second; ++it) {
    tmpVar.insert(it->second);
  }
  return tmpVar;
}



// Get the named NcmpiVar object.
NcmpiVar NcmpiGroup::getVar(const string& name,NcmpiGroup::Location location) const {
  multimap<std::string,NcmpiVar> ncmpiVars(getVars(location));
  pair<multimap<string,NcmpiVar>::iterator,multimap<string,NcmpiVar>::iterator> ret;
  ret = ncmpiVars.equal_range(name);
  if(ret.first == ret.second) 
    // no matching netCDF variable found so return null object.
    return NcmpiVar();
  else 
    return ret.first->second;
}

// Adds a new netCDF scalar variable.
NcmpiVar NcmpiGroup::addVar(const std::string& name, const NcmpiType& ncmpiType) const {
  return NcmpiGroup::addVar(name, ncmpiType, std::vector<NcmpiDim>());
}   

// Add a new netCDF variable.
NcmpiVar NcmpiGroup::addVar(const string& name, const string& typeName, const string& dimName) const {
  ncmpiCheckDefineMode(myId);
    
  // get an NcmpiType object with the given type name.
  NcmpiType tmpType(getType(typeName,NcmpiGroup::ParentsAndCurrent));
  if(tmpType.isNull()) throw NcNullType("Attempt to invoke NcmpiGroup::addVar failed: typeName must be defined in either the current group or a parent group",__FILE__,__LINE__);

  // get a NcmpiDim object with the given dimension name
  NcmpiDim tmpDim(getDim(dimName,NcmpiGroup::ParentsAndCurrent));
  if(tmpDim.isNull()) throw NcNullDim("Attempt to invoke NcmpiGroup::addVar failed: dimName must be defined in either the current group or a parent group",__FILE__,__LINE__);

  // finally define a new netCDF  variable
  int varId;
  int dimId(tmpDim.getId());
  ncmpiCheck(ncmpi_def_var(myId,name.c_str(),tmpType.getId(),1,&dimId,&varId),__FILE__,__LINE__);
  // return an NcmpiVar object for this new variable
  return NcmpiVar(*this,varId);
}


// Add a new netCDF variable.
NcmpiVar NcmpiGroup::addVar(const string& name, const NcmpiType& ncmpiType, const NcmpiDim& ncmpiDim) const {
  ncmpiCheckDefineMode(myId);
    
  // check NcmpiType object is valid
  if(ncmpiType.isNull()) throw NcNullType("Attempt to invoke NcmpiGroup::addVar with a Null NcmpiType object",__FILE__,__LINE__);
  NcmpiType tmpType(getType(ncmpiType.getName(),NcmpiGroup::ParentsAndCurrent));
  if(tmpType.isNull()) throw NcNullType("Attempt to invoke NcmpiGroup::addVar failed: NcmpiType must be defined in either the current group or a parent group",__FILE__,__LINE__);
  
  // check NcmpiDim object is valid
  if(ncmpiDim.isNull()) throw NcNullDim("Attempt to invoke NcmpiGroup::addVar with a Null NcmpiDim object",__FILE__,__LINE__);
  NcmpiDim tmpDim(getDim(ncmpiDim.getName(),NcmpiGroup::ParentsAndCurrent));
  if(tmpDim.isNull()) throw NcNullDim("Attempt to invoke NcmpiGroup::addVar failed: NcmpiDim must be defined in either the current group or a parent group",__FILE__,__LINE__);
  
  // finally define a new netCDF variable
  int varId;
  int dimId(tmpDim.getId());
  ncmpiCheck(ncmpi_def_var(myId,name.c_str(),tmpType.getId(),1,&dimId,&varId),__FILE__,__LINE__);
  // return an NcmpiVar object for this new variable
  return NcmpiVar(*this,varId);
}


// Add a new netCDF multi-dimensional variable.
NcmpiVar NcmpiGroup::addVar(const string& name, const string& typeName, const vector<string>& dimNames) const {
  ncmpiCheckDefineMode(myId);
    
  // get an NcmpiType object with the given name.
  NcmpiType tmpType(getType(typeName,NcmpiGroup::ParentsAndCurrent));
  if(tmpType.isNull()) throw NcNullType("Attempt to invoke NcmpiGroup::addVar failed: typeName must be defined in either the current group or a parent group",__FILE__,__LINE__);

  // get a set of NcmpiDim objects corresponding to the given dimension names.
  vector<int> dimIds;
  dimIds.reserve(dimNames.size());
  for (size_t i=0; i<dimNames.size();i++){
    NcmpiDim tmpDim(getDim(dimNames[i],NcmpiGroup::ParentsAndCurrent));
    if(tmpDim.isNull()) throw NcNullDim("Attempt to invoke NcmpiGroup::addVar failed: dimNames must be defined in either the current group or a parent group",__FILE__,__LINE__);
    dimIds.push_back(tmpDim.getId());
  }

  // finally define a new netCDF variable
  int varId;
  int *dimIdsPtr = dimIds.empty() ? 0 : &dimIds[0];
  ncmpiCheck(ncmpi_def_var(myId,name.c_str(),tmpType.getId(),dimIds.size(), dimIdsPtr,&varId),__FILE__,__LINE__);
  // return an NcmpiVar object for this new variable
  return NcmpiVar(*this,varId);
}

// Add a new netCDF multi-dimensional variable.
NcmpiVar NcmpiGroup::addVar(const string& name, const NcmpiType& ncmpiType, const vector<NcmpiDim>& ncmpiDimVector) const {
  ncmpiCheckDefineMode(myId);
    
  // check NcmpiType object is valid
  if(ncmpiType.isNull()) throw NcNullType("Attempt to invoke NcmpiGroup::addVar with a Null NcmpiType object",__FILE__,__LINE__);
  NcmpiType tmpType(getType(ncmpiType.getName(),NcmpiGroup::ParentsAndCurrent));
  if(tmpType.isNull()) throw NcNullType("Attempt to invoke NcmpiGroup::addVar failed: NcmpiType must be defined in either the current group or a parent group",__FILE__,__LINE__);
  
  // check NcmpiDim objects are valid
  vector<NcmpiDim>::const_iterator iter;
  vector<int> dimIds;
  dimIds.reserve(ncmpiDimVector.size());
  for (iter=ncmpiDimVector.begin();iter < ncmpiDimVector.end(); iter++) {
    if(iter->isNull()) throw NcNullDim("Attempt to invoke NcmpiGroup::addVar with a Null NcmpiDim object",__FILE__,__LINE__);
    NcmpiDim tmpDim(getDim(iter->getName(),NcmpiGroup::ParentsAndCurrent));
    if(tmpDim.isNull()) throw NcNullDim("Attempt to invoke NcmpiGroup::addVar failed: NcmpiDim must be defined in either the current group or a parent group",__FILE__,__LINE__);
    dimIds.push_back(tmpDim.getId());
  }

  // finally define a new netCDF variable
  int varId;
  int *dimIdsPtr = dimIds.empty() ? 0 : &dimIds[0];
  ncmpiCheck(ncmpi_def_var(myId,name.c_str(),tmpType.getId(),dimIds.size(), dimIdsPtr,&varId),__FILE__,__LINE__);
  // return an NcmpiVar object for this new variable
  return NcmpiVar(*this,varId);
}


// /////////////
// NcmpiAtt-related methods
// /////////////
  
// Get the number of group attributes.
int NcmpiGroup::getAttCount(NcmpiGroup::Location location) const {

  // search in current group.
  NcmpiGroup tmpGroup(*this); 
  int ngatts=0;
  // search in current group
  if((location == ParentsAndCurrent || location == ChildrenAndCurrent || location == Current || location ==All) && !tmpGroup.isNull()) {
    ncmpiCheck(ncmpi_inq_natts(tmpGroup.getId(), &ngatts),__FILE__,__LINE__);
  }

  // search recursively in all parent groups.
  if(location == Parents || location == ParentsAndCurrent || location ==All) {
    tmpGroup=getParentGroup();
    while(!tmpGroup.isNull()) {
      int ngattsp;
      ncmpiCheck(ncmpi_inq_natts(tmpGroup.getId(), &ngattsp),__FILE__,__LINE__);
      ngatts += ngattsp;
      // continue loop with the parent.
      tmpGroup=tmpGroup.getParentGroup();
    }
  }

  // search recursively in all child groups
  if(location == ChildrenAndCurrent || location == Children || location == All) {
    multimap<string,NcmpiGroup>::iterator it;
    multimap<string,NcmpiGroup> groups(getGroups());
    for (it=groups.begin();it!=groups.end();it++) {
      ngatts += it->second.getAttCount(ChildrenAndCurrent);
    }
  }

  return ngatts;
}
  
// Get the collection of NcmpiGroupAtt objects.
multimap<std::string,NcmpiGroupAtt> NcmpiGroup::getAtts(NcmpiGroup::Location location) const {

  // create a container to hold the NcmpiAtt's.
  multimap<string,NcmpiGroupAtt> ncmpiAtts;

  // search in current group.
  NcmpiGroup tmpGroup(*this); 
  if((location == ParentsAndCurrent || location == ChildrenAndCurrent || location == Current || location ==All) && !tmpGroup.isNull()) {
    // get the number of attributes
    int attCount = tmpGroup.getAttCount();
    // now get the name of each NcmpiAtt and populate the ncmpiAtts container.
    for(int i=0; i<attCount;i++){
      char charName[NC_MAX_NAME+1];
      ncmpiCheck(ncmpi_inq_attname(tmpGroup.getId(),NC_GLOBAL,i,charName),__FILE__,__LINE__);
      NcmpiGroupAtt tmpAtt(tmpGroup.getId(),i);
      ncmpiAtts.insert(pair<const string,NcmpiGroupAtt>(string(charName),tmpAtt));
    }
  }
  
  // search recursively in all parent groups.
  if(location == Parents || location == ParentsAndCurrent || location ==All) {
    tmpGroup=getParentGroup();
    while(!tmpGroup.isNull()) {
      // get the number of attributes
      int attCount = tmpGroup.getAttCount();
      // now get the name of each NcmpiAtt and populate the ncmpiAtts container.
      for(int i=0; i<attCount;i++){
        char charName[NC_MAX_NAME+1];
        ncmpiCheck(ncmpi_inq_attname(tmpGroup.getId(),NC_GLOBAL,i,charName),__FILE__,__LINE__);
        NcmpiGroupAtt tmpAtt(tmpGroup.getId(),i);
        ncmpiAtts.insert(pair<const string,NcmpiGroupAtt>(string(charName),tmpAtt));
      }
      // continue loop with the parent.
      tmpGroup=tmpGroup.getParentGroup();
    }
  }

  // search recusively in all child groups.
  if(location == ChildrenAndCurrent || location == Children  || location == All ) {
    multimap<string,NcmpiGroup>::iterator it;
    multimap<string,NcmpiGroup> groups(getGroups());
    for (it=groups.begin();it!=groups.end();it++) {
      multimap<string,NcmpiGroupAtt> atts=it->second.getAtts(ChildrenAndCurrent);
      ncmpiAtts.insert(atts.begin(),atts.end());
    }
  }

  return ncmpiAtts;
}
  
// Get the named NcmpiGroupAtt object.
NcmpiGroupAtt NcmpiGroup::getAtt(const std::string& name,NcmpiGroup::Location location) const {
  multimap<std::string,NcmpiGroupAtt> ncmpiAtts(getAtts(location));
  pair<multimap<string,NcmpiGroupAtt>::iterator,multimap<string,NcmpiGroupAtt>::iterator> ret;
  ret = ncmpiAtts.equal_range(name);
  if(ret.first == ret.second) 
    // no matching groupAttribute so return null object.
    return NcmpiGroupAtt();
  else 
    return ret.first->second;
}

// Get all NcmpiGroupAtt objects with a given name.
set<NcmpiGroupAtt> NcmpiGroup::getAtts(const string& name,NcmpiGroup::Location location) const {
  // get the set of ncmpiGroupAtts in this group and above.
  multimap<std::string,NcmpiGroupAtt> ncmpiAtts(getAtts(location));
  pair<multimap<string,NcmpiGroupAtt>::iterator,multimap<string,NcmpiGroupAtt>::iterator> ret;
  multimap<string,NcmpiGroupAtt>::iterator it;
  ret = ncmpiAtts.equal_range(name);
  set<NcmpiGroupAtt> tmpAtt;
  for (it=ret.first; it!=ret.second; ++it) {
    tmpAtt.insert(it->second);
  }
  return tmpAtt;
}




//  Creates a new NetCDF group attribute or if already exisiting replaces it.
NcmpiGroupAtt NcmpiGroup::putAtt(const string& name, const string& dataValues) const {
  ncmpiCheckDefineMode(myId);
  ncmpiCheck(ncmpi_put_att_text(myId,NC_GLOBAL,name.c_str(),dataValues.size(),dataValues.c_str()),__FILE__,__LINE__);
  // finally instantiate this attribute and return
  return getAtt(name);
}

//  Creates a new NetCDF group attribute or if already exisiting replaces it.
NcmpiGroupAtt NcmpiGroup::putAtt(const string& name, const NcmpiType& type, MPI_Offset len, const unsigned char* dataValues) const {
  ncmpiCheckDefineMode(myId);
  NcmpiType::ncmpiType typeClass(type.getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_put_att(myId,NC_GLOBAL,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_put_att_uchar(myId,NC_GLOBAL,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  // finally instantiate this attribute and return
  return getAtt(name);
}


//  Creates a new NetCDF group attribute or if already exisiting replaces it.
NcmpiGroupAtt NcmpiGroup::putAtt(const string& name, const NcmpiType& type, MPI_Offset len, const signed char* dataValues) const {
  ncmpiCheckDefineMode(myId);
  NcmpiType::ncmpiType typeClass(type.getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_put_att(myId,NC_GLOBAL,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_put_att_schar(myId,NC_GLOBAL,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  // finally instantiate this attribute and return
  return getAtt(name);
}


//  Creates a new NetCDF group attribute or if already exisiting replaces it.
NcmpiGroupAtt NcmpiGroup::putAtt(const string& name, const NcmpiType& type, short datumValue) const {
  ncmpiCheckDefineMode(myId);
  NcmpiType::ncmpiType typeClass(type.getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_put_att(myId,NC_GLOBAL,name.c_str(),type.getId(),1,&datumValue),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_put_att_short(myId,NC_GLOBAL,name.c_str(),type.getId(),1,&datumValue),__FILE__,__LINE__);
  // finally instantiate this attribute and return
  return getAtt(name);
}


//  Creates a new NetCDF group attribute or if already exisiting replaces it.
NcmpiGroupAtt NcmpiGroup::putAtt(const string& name, const NcmpiType& type, int datumValue) const {
  ncmpiCheckDefineMode(myId);
  NcmpiType::ncmpiType typeClass(type.getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_put_att(myId,NC_GLOBAL,name.c_str(),type.getId(),1,&datumValue),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_put_att_int(myId,NC_GLOBAL,name.c_str(),type.getId(),1,&datumValue),__FILE__,__LINE__);
  // finally instantiate this attribute and return
  return getAtt(name);
}

//  Creates a new NetCDF group attribute or if already exisiting replaces it.
NcmpiGroupAtt NcmpiGroup::putAtt(const string& name, const NcmpiType& type, long datumValue) const {
  ncmpiCheckDefineMode(myId);
  NcmpiType::ncmpiType typeClass(type.getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_put_att(myId,NC_GLOBAL,name.c_str(),type.getId(),1,&datumValue),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_put_att_long(myId,NC_GLOBAL,name.c_str(),type.getId(),1,&datumValue),__FILE__,__LINE__);
  // finally instantiate this attribute and return
  return getAtt(name);
}

//  Creates a new NetCDF group attribute or if already exisiting replaces it.
NcmpiGroupAtt NcmpiGroup::putAtt(const string& name, const NcmpiType& type, float datumValue) const {
  ncmpiCheckDefineMode(myId);
  NcmpiType::ncmpiType typeClass(type.getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_put_att(myId,NC_GLOBAL,name.c_str(),type.getId(),1,&datumValue),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_put_att_float(myId,NC_GLOBAL,name.c_str(),type.getId(),1,&datumValue),__FILE__,__LINE__);
  // finally instantiate this attribute and return
  return getAtt(name);
}


//  Creates a new NetCDF group attribute or if already exisiting replaces it.
NcmpiGroupAtt NcmpiGroup::putAtt(const string& name, const NcmpiType& type, double datumValue) const {
  ncmpiCheckDefineMode(myId);
  NcmpiType::ncmpiType typeClass(type.getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_put_att(myId,NC_GLOBAL,name.c_str(),type.getId(),1,&datumValue),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_put_att_double(myId,NC_GLOBAL,name.c_str(),type.getId(),1,&datumValue),__FILE__,__LINE__);
  // finally instantiate this attribute and return
  return getAtt(name);
}


//  Creates a new NetCDF group attribute or if already exisiting replaces it.
NcmpiGroupAtt NcmpiGroup::putAtt(const string& name, const NcmpiType& type, unsigned short datumValue) const {
  ncmpiCheckDefineMode(myId);
  NcmpiType::ncmpiType typeClass(type.getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_put_att(myId,NC_GLOBAL,name.c_str(),type.getId(),1,&datumValue),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_put_att_ushort(myId,NC_GLOBAL,name.c_str(),type.getId(),1,&datumValue),__FILE__,__LINE__);
  // finally instantiate this attribute and return
  return getAtt(name);
}

//  Creates a new NetCDF group attribute or if already exisiting replaces it.
NcmpiGroupAtt NcmpiGroup::putAtt(const string& name, const NcmpiType& type, unsigned int datumValue) const {
  ncmpiCheckDefineMode(myId);
  NcmpiType::ncmpiType typeClass(type.getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_put_att(myId,NC_GLOBAL,name.c_str(),type.getId(),1,&datumValue),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_put_att_uint(myId,NC_GLOBAL,name.c_str(),type.getId(),1,&datumValue),__FILE__,__LINE__);
  // finally instantiate this attribute and return
  return getAtt(name);
}

//  Creates a new NetCDF group attribute or if already exisiting replaces it.
NcmpiGroupAtt NcmpiGroup::putAtt(const string& name, const NcmpiType& type, long long datumValue) const {
  ncmpiCheckDefineMode(myId);
  NcmpiType::ncmpiType typeClass(type.getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_put_att(myId,NC_GLOBAL,name.c_str(),type.getId(),1,&datumValue),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_put_att_longlong(myId,NC_GLOBAL,name.c_str(),type.getId(),1,&datumValue),__FILE__,__LINE__);
  // finally instantiate this attribute and return
  return getAtt(name);
}


//  Creates a new NetCDF group attribute or if already exisiting replaces it.
NcmpiGroupAtt NcmpiGroup::putAtt(const string& name, const NcmpiType& type, unsigned long long datumValue) const {
  ncmpiCheckDefineMode(myId);
  NcmpiType::ncmpiType typeClass(type.getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_put_att(myId,NC_GLOBAL,name.c_str(),type.getId(),1,&datumValue),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_put_att_ulonglong(myId,NC_GLOBAL,name.c_str(),type.getId(),1,&datumValue),__FILE__,__LINE__);
  // finally instantiate this attribute and return
  return getAtt(name);
}


//  Creates a new NetCDF group attribute or if already exisiting replaces it.
NcmpiGroupAtt NcmpiGroup::putAtt(const string& name, const NcmpiType& type, MPI_Offset len, const short* dataValues) const {
  ncmpiCheckDefineMode(myId);
  NcmpiType::ncmpiType typeClass(type.getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_put_att(myId,NC_GLOBAL,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_put_att_short(myId,NC_GLOBAL,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  // finally instantiate this attribute and return
  return getAtt(name);
}


//  Creates a new NetCDF group attribute or if already exisiting replaces it.
NcmpiGroupAtt NcmpiGroup::putAtt(const string& name, const NcmpiType& type, MPI_Offset len, const int* dataValues) const {
  ncmpiCheckDefineMode(myId);
  NcmpiType::ncmpiType typeClass(type.getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_put_att(myId,NC_GLOBAL,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_put_att_int(myId,NC_GLOBAL,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  // finally instantiate this attribute and return
  return getAtt(name);
}

//  Creates a new NetCDF group attribute or if already exisiting replaces it.
NcmpiGroupAtt NcmpiGroup::putAtt(const string& name, const NcmpiType& type, MPI_Offset len, const long* dataValues) const {
  ncmpiCheckDefineMode(myId);
  NcmpiType::ncmpiType typeClass(type.getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_put_att(myId,NC_GLOBAL,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_put_att_long(myId,NC_GLOBAL,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  // finally instantiate this attribute and return
  return getAtt(name);
}

//  Creates a new NetCDF group attribute or if already exisiting replaces it.
NcmpiGroupAtt NcmpiGroup::putAtt(const string& name, const NcmpiType& type, MPI_Offset len, const float* dataValues) const {
  ncmpiCheckDefineMode(myId);
  NcmpiType::ncmpiType typeClass(type.getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_put_att(myId,NC_GLOBAL,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_put_att_float(myId,NC_GLOBAL,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  // finally instantiate this attribute and return
  return getAtt(name);
}


//  Creates a new NetCDF group attribute or if already exisiting replaces it.
NcmpiGroupAtt NcmpiGroup::putAtt(const string& name, const NcmpiType& type, MPI_Offset len, const double* dataValues) const {
  ncmpiCheckDefineMode(myId);
  NcmpiType::ncmpiType typeClass(type.getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_put_att(myId,NC_GLOBAL,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_put_att_double(myId,NC_GLOBAL,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  // finally instantiate this attribute and return
  return getAtt(name);
}


//  Creates a new NetCDF group attribute or if already exisiting replaces it.
NcmpiGroupAtt NcmpiGroup::putAtt(const string& name, const NcmpiType& type, MPI_Offset len, const unsigned short* dataValues) const {
  ncmpiCheckDefineMode(myId);
  NcmpiType::ncmpiType typeClass(type.getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_put_att(myId,NC_GLOBAL,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_put_att_ushort(myId,NC_GLOBAL,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  // finally instantiate this attribute and return
  return getAtt(name);
}

//  Creates a new NetCDF group attribute or if already exisiting replaces it.
NcmpiGroupAtt NcmpiGroup::putAtt(const string& name, const NcmpiType& type, MPI_Offset len, const unsigned int* dataValues) const {
  ncmpiCheckDefineMode(myId);
  NcmpiType::ncmpiType typeClass(type.getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_put_att(myId,NC_GLOBAL,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_put_att_uint(myId,NC_GLOBAL,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  // finally instantiate this attribute and return
  return getAtt(name);
}

//  Creates a new NetCDF group attribute or if already exisiting replaces it.
NcmpiGroupAtt NcmpiGroup::putAtt(const string& name, const NcmpiType& type, MPI_Offset len, const long long* dataValues) const {
  ncmpiCheckDefineMode(myId);
  NcmpiType::ncmpiType typeClass(type.getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_put_att(myId,NC_GLOBAL,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_put_att_longlong(myId,NC_GLOBAL,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  // finally instantiate this attribute and return
  return getAtt(name);
}


//  Creates a new NetCDF group attribute or if already exisiting replaces it.
NcmpiGroupAtt NcmpiGroup::putAtt(const string& name, const NcmpiType& type, MPI_Offset len, const unsigned long long* dataValues) const {
  ncmpiCheckDefineMode(myId);
  NcmpiType::ncmpiType typeClass(type.getTypeClass());
  if(typeClass == NcmpiType::ncmpi_VLEN || typeClass == NcmpiType::ncmpi_OPAQUE || typeClass == NcmpiType::ncmpi_ENUM || typeClass == NcmpiType::ncmpi_COMPOUND) 
    ncmpiCheck(ncmpi_put_att(myId,NC_GLOBAL,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  else
    ncmpiCheck(ncmpi_put_att_ulonglong(myId,NC_GLOBAL,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  // finally instantiate this attribute and return
  return getAtt(name);
}


//  Creates a new NetCDF group attribute or if already exisiting replaces it.
 NcmpiGroupAtt NcmpiGroup::putAtt(const string& name, const NcmpiType& type, MPI_Offset len, const void* dataValues) const {
  ncmpiCheckDefineMode(myId);
  ncmpiCheck(ncmpi_put_att(myId,NC_GLOBAL,name.c_str(),type.getId(),len,dataValues),__FILE__,__LINE__);
  // finally instantiate this attribute and return
  return getAtt(name);
}


  
// /////////////
// NcmpiDim-related methods
// /////////////
  
// Get the number of NcmpiDim objects.
int NcmpiGroup::getDimCount(NcmpiGroup::Location location) const {
  if(isNull()) throw NcNullGrp("Attempt to invoke NcmpiGroup::getDimCount on a Null group",__FILE__,__LINE__);

  // intialize counter
  int ndims=0;

  // search in current group
  if(location == Current || location == ParentsAndCurrent || location == ChildrenAndCurrent || location == All ) {
    int ndimsp;
    ncmpiCheck(ncmpi_inq_ndims(getId(), &ndimsp),__FILE__,__LINE__);
    ndims += ndimsp;
  }

  // search in parent groups.
  if(location == Parents || location == ParentsAndCurrent || location == All ) {
    multimap<string,NcmpiGroup>::iterator it;
    multimap<string,NcmpiGroup> groups(getGroups(ParentsGrps));
    for (it=groups.begin();it!=groups.end();it++) {
      ndims += it->second.getDimCount();
    }
  }
  
  // search in child groups.
  if(location == Children || location == ChildrenAndCurrent || location == All ) {
    multimap<string,NcmpiGroup>::iterator it;
    multimap<string,NcmpiGroup> groups(getGroups(AllChildrenGrps));
    for (it=groups.begin();it!=groups.end();it++) {
      ndims += it->second.getDimCount();
    }
  }
  return ndims;
}


// Get the set of NcmpiDim objects.
multimap<string,NcmpiDim> NcmpiGroup::getDims(NcmpiGroup::Location location) const {
  if(isNull()) throw NcNullGrp("Attempt to invoke NcmpiGroup::getDims on a Null group",__FILE__,__LINE__);
  // create a container to hold the NcmpiDim's.
  multimap<string,NcmpiDim> ncmpiDims;

  // search in current group
  if(location == Current || location == ParentsAndCurrent || location == ChildrenAndCurrent || location == All ) {
    int dimCount = getDimCount();
    if (dimCount){
      vector<int> dimids(dimCount);
      // ncmpiCheck(ncmpi_inq_dimids(getId(), &dimCount, &dimids[0], 0),__FILE__,__LINE__);
      ncmpiCheck(ncmpi_inq_ndims(getId(), &dimCount),__FILE__,__LINE__);
      // now get the name of each NcmpiDim and populate the nDims container.
      for(int i=0; i<dimCount;i++){
        dimids[i] = i;
        NcmpiDim tmpDim(*this,dimids[i]); 
        ncmpiDims.insert(pair<const string,NcmpiDim>(tmpDim.getName(),tmpDim));
      }
    }
  }

  // search in parent groups.
  if(location == Parents || location == ParentsAndCurrent || location == All ) {
    multimap<string,NcmpiGroup>::iterator it;
    multimap<string,NcmpiGroup> groups(getGroups(ParentsGrps));
    for (it=groups.begin();it!=groups.end();it++) {
      multimap<string,NcmpiDim> dimTmp(it->second.getDims());
      ncmpiDims.insert(dimTmp.begin(),dimTmp.end());
    }
  }

  // search in child groups (makes recursive calls).
  if(location == Children || location == ChildrenAndCurrent || location == All ) {
    multimap<string,NcmpiGroup>::iterator it;
    multimap<string,NcmpiGroup> groups(getGroups(AllChildrenGrps));
    for (it=groups.begin();it!=groups.end();it++) {
      multimap<string,NcmpiDim> dimTmp(it->second.getDims());
      ncmpiDims.insert(dimTmp.begin(),dimTmp.end());
    }
  }

  return ncmpiDims;
}



// Get the named NcmpiDim object.
NcmpiDim NcmpiGroup::getDim(const string& name,NcmpiGroup::Location location) const {
  if(isNull()) throw NcNullGrp("Attempt to invoke NcmpiGroup::getDim on a Null group",__FILE__,__LINE__);
  multimap<string,NcmpiDim> ncmpiDims(getDims(location));
  pair<multimap<string,NcmpiDim>::iterator,multimap<string,NcmpiDim>::iterator> ret;
  ret = ncmpiDims.equal_range(name);
  if(ret.first == ret.second) 
    return NcmpiDim(); // null group is returned
  else 
    return ret.first->second;
}


// Get all NcmpiDim objects with a given name.
set<NcmpiDim> NcmpiGroup::getDims(const string& name,NcmpiGroup::Location location) const {
  if(isNull()) throw NcNullGrp("Attempt to invoke NcmpiGroup::getDims on a Null group",__FILE__,__LINE__);
  // get the set of ncmpiDims in this group and above.
  multimap<string,NcmpiDim> ncmpiDims(getDims(location));
  pair<multimap<string,NcmpiDim>::iterator,multimap<string,NcmpiDim>::iterator> ret;
  multimap<string,NcmpiDim>::iterator it;
  ret = ncmpiDims.equal_range(name);
  set<NcmpiDim> tmpDim;
  for (it=ret.first; it!=ret.second; ++it) {
    tmpDim.insert(it->second);
  }
  return tmpDim;
}

// Add a new NcmpiDim object.
NcmpiDim NcmpiGroup::addDim(const string& name, MPI_Offset dimSize) const {
  ncmpiCheckDefineMode(myId);
  if(isNull()) throw NcNullGrp("Attempt to invoke NcmpiGroup::addDim on a Null group",__FILE__,__LINE__);
  int dimId;
  ncmpiCheck(ncmpi_def_dim(myId,name.c_str(),dimSize,&dimId),__FILE__,__LINE__);
  // finally return NcmpiDim object for this new variable
  return NcmpiDim(*this,dimId);
}

// Add a new NcmpiDim object with unlimited size..
NcmpiDim NcmpiGroup::addDim(const string& name) const {
  ncmpiCheckDefineMode(myId);
  if(isNull()) throw NcNullGrp("Attempt to invoke NcmpiGroup::addDim on a Null group",__FILE__,__LINE__);
  int dimId;
  ncmpiCheck(ncmpi_def_dim(myId,name.c_str(),NC_UNLIMITED,&dimId),__FILE__,__LINE__);
  // finally return NcmpiDim object for this new variable
  return NcmpiDim(*this,dimId);
}





// /////////////
// type-object related methods
// /////////////

// Gets the number of type objects.
int NcmpiGroup::getTypeCount(NcmpiGroup::Location location) const {

  if(isNull()) throw NcNullGrp("Attempt to invoke NcmpiGroup::getTypeCount on a Null group",__FILE__,__LINE__);

  // intialize counter
  int ntypes=0;

  // search in current group
  if(location == Current || location == ParentsAndCurrent || location == ChildrenAndCurrent || location == All ) {
    int ntypesp=0;
    int* typeidsp=NULL;
    ncmpiCheck(ncmpi_inq_typeids(getId(), &ntypesp,typeidsp),__FILE__,__LINE__);
    ntypes+= ntypesp;
  }

  // search in parent groups.
  if(location == Parents || location == ParentsAndCurrent || location == All ) {
    multimap<string,NcmpiGroup>::iterator it;
    multimap<string,NcmpiGroup> groups(getGroups(ParentsGrps));
    for (it=groups.begin();it!=groups.end();it++) {
      ntypes += it->second.getTypeCount();
    }
  }
  
  // search in child groups.
  if(location == Children || location == ChildrenAndCurrent || location == All ) {
    multimap<string,NcmpiGroup>::iterator it;
    multimap<string,NcmpiGroup> groups(getGroups(AllChildrenGrps));
    for (it=groups.begin();it!=groups.end();it++) {
      ntypes += it->second.getTypeCount();
    }
  }
  return ntypes;
}


  
// Gets the number of type objects with a given enumeration type.
int NcmpiGroup::getTypeCount(NcmpiType::ncmpiType enumType, NcmpiGroup::Location location) const {

  if(isNull()) throw NcNullGrp("Attempt to invoke NcmpiGroup::getTypeCount on a Null group",__FILE__,__LINE__);

  // intialize counter
  int ntypes=0;

  // search in current group
  if(location == Current || location == ParentsAndCurrent || location == ChildrenAndCurrent || location == All ) {
    int ntypesp=0;
    int* typeidsp=NULL;
    ncmpiCheck(ncmpi_inq_typeids(getId(), &ntypesp,typeidsp),__FILE__,__LINE__);
    if (ntypesp){
      vector<int> typeids(ntypesp);
      ncmpiCheck(ncmpi_inq_typeids(getId(), &ntypesp,&typeids[0]),__FILE__,__LINE__);
      for (int i=0; i<ntypesp;i++){
        NcmpiType tmpType(*this,typeids[i]);
        if(tmpType.getTypeClass() == enumType) ntypes++;
      }
    }
  }

  // search in parent groups.
  if(location == Parents || location == ParentsAndCurrent || location == All ) {
    multimap<string,NcmpiGroup>::iterator it;
    multimap<string,NcmpiGroup> groups(getGroups(ParentsGrps));
    for (it=groups.begin();it!=groups.end();it++) {
      ntypes += it->second.getTypeCount(enumType);
    }
  }
  
  // search in child groups.
  if(location == Children || location == ChildrenAndCurrent || location == All ) {
    multimap<string,NcmpiGroup>::iterator it;
    multimap<string,NcmpiGroup> groups(getGroups(AllChildrenGrps));
    for (it=groups.begin();it!=groups.end();it++) {
      ntypes += it->second.getTypeCount(enumType);
    }
  }
  return ntypes;
}


// Gets the collection of NcmpiType objects.
multimap<string,NcmpiType> NcmpiGroup::getTypes(NcmpiGroup::Location location) const {
  if(isNull()) throw NcNullGrp("Attempt to invoke NcmpiGroup::getTypes on a Null group",__FILE__,__LINE__);
  // create a container to hold the NcmpiType's.
  multimap<string,NcmpiType> ncmpiTypes;

  // search in current group
  if(location == Current || location == ParentsAndCurrent || location == ChildrenAndCurrent || location == All ) {
    int typeCount = getTypeCount();
    if (typeCount){
      vector<int> typeids(typeCount);
      ncmpiCheck(ncmpi_inq_typeids(getId(), &typeCount,&typeids[0]),__FILE__,__LINE__);
      // now get the name of each NcmpiType and populate the nTypes container.
      for(int i=0; i<typeCount;i++){
        NcmpiType tmpType(*this,typeids[i]); 
        ncmpiTypes.insert(pair<const string,NcmpiType>(tmpType.getName(),tmpType));
      }
    }
  }

  // search in parent groups.
  if(location == Parents || location == ParentsAndCurrent || location == All ) {
    multimap<string,NcmpiGroup>::iterator it;
    multimap<string,NcmpiGroup> groups(getGroups(ParentsGrps));
    for (it=groups.begin();it!=groups.end();it++) {
      multimap<string,NcmpiType> typeTmp(it->second.getTypes());
      ncmpiTypes.insert(typeTmp.begin(),typeTmp.end());
    }
  }

  // search in child groups (makes recursive calls).
  if(location == Children || location == ChildrenAndCurrent || location == All ) {
    multimap<string,NcmpiGroup>::iterator it;
    multimap<string,NcmpiGroup> groups(getGroups(AllChildrenGrps));
    for (it=groups.begin();it!=groups.end();it++) {
      multimap<string,NcmpiType> typeTmp(it->second.getTypes());
      ncmpiTypes.insert(typeTmp.begin(),typeTmp.end());
    }
  }

  return ncmpiTypes;
}


// Gets the collection of NcmpiType objects with a given name.
set<NcmpiType> NcmpiGroup::getTypes(const string& name, NcmpiGroup::Location location) const {
  if(isNull()) throw NcNullGrp("Attempt to invoke NcmpiGroup::getTypes on a Null group",__FILE__,__LINE__);
  // iterator for the multimap container.
  multimap<string,NcmpiType>::iterator it;
  // return argument of equal_range: iterators to lower and upper bounds of the range.
  pair<multimap<string,NcmpiType>::iterator,multimap<string,NcmpiType>::iterator> ret;
  // get the entire collection of types.
  multimap<string,NcmpiType> types(getTypes(location));
  // define STL set object to hold the result
  set<NcmpiType> tmpType;
  // get the set of NcmpiType objects with a given name
  ret=types.equal_range(name);
  for (it=ret.first;it!=ret.second;it++) {
    tmpType.insert(it->second);
  }
  return tmpType;
}


// Gets the collection of NcmpiType objects with a given data type.
set<NcmpiType> NcmpiGroup::getTypes(NcmpiType::ncmpiType enumType, NcmpiGroup::Location location) const {
  if(isNull()) throw NcNullGrp("Attempt to invoke NcmpiGroup::getTypes on a Null group",__FILE__,__LINE__);
  // iterator for the multimap container.
  multimap<string,NcmpiType>::iterator it;
  // get the entire collection of types.
  multimap<string,NcmpiType> types(getTypes(location));
  // define STL set object to hold the result
  set<NcmpiType> tmpType;
  // get the set of NcmpiType objects with a given data type
  for (it=types.begin();it!=types.end();it++) {
    if(it->second.getTypeClass() == enumType) {
      tmpType.insert(it->second);
    }
  }
  return(tmpType);
}


// Gets the collection of NcmpiType objects with a given name and data type.
set<NcmpiType> NcmpiGroup::getTypes(const string& name, NcmpiType::ncmpiType enumType, NcmpiGroup::Location location) const {
  if(isNull()) throw NcNullGrp("Attempt to invoke NcmpiGroup::getTypes on a Null group",__FILE__,__LINE__);
  // iterator for the multimap container.
  multimap<string,NcmpiType>::iterator it;
  // return argument of equal_range: iterators to lower and upper bounds of the range.
  pair<multimap<string,NcmpiType>::iterator,multimap<string,NcmpiType>::iterator> ret;
  // get the entire collection of types.
  multimap<string,NcmpiType> types(getTypes(location));
  // define STL set object to hold the result
  set<NcmpiType> tmpType;
  // get the set of NcmpiType objects with a given name
  ret=types.equal_range(name);
  for (it=ret.first;it!=ret.second;it++) {
    if((*it).second.getTypeClass() == enumType) {
      tmpType.insert(it->second);
    }
  }
  return(tmpType);
}


// Gets the NcmpiType object with a given name.
NcmpiType NcmpiGroup::getType(const string& name, NcmpiGroup::Location location) const {
  if(isNull()) throw NcNullGrp("Attempt to invoke NcmpiGroup::getType on a Null group",__FILE__,__LINE__);
  if(name ==  "byte"    ) return ncmpiByte;
  if(name ==  "ubyte"   ) return ncmpiUbyte;
  if(name ==  "char"    ) return ncmpiChar;
  if(name ==  "short"   ) return ncmpiShort;
  if(name ==  "ushort"  ) return ncmpiUshort;
  if(name ==  "int"     ) return ncmpiInt;
  if(name ==  "uint"    ) return ncmpiUint;  
  if(name ==  "int64"   ) return ncmpiInt64; 
  if(name ==  "uint64"  ) return ncmpiUint64;
  if(name ==  "float"   ) return ncmpiFloat;
  if(name ==  "double"  ) return ncmpiDouble;

  // this is a user defined type
  // iterator for the multimap container.
  multimap<string,NcmpiType>::iterator it;
  // return argument of equal_range: iterators to lower and upper bounds of the range.
  pair<multimap<string,NcmpiType>::iterator,multimap<string,NcmpiType>::iterator> ret;
  // get the entire collection of types.
  multimap<string,NcmpiType> types(getTypes(location));
  // define STL set object to hold the result
  set<NcmpiType> tmpType;
    // get the set of NcmpiType objects with a given name
  ret=types.equal_range(name);
  if(ret.first == ret.second) 
    return NcmpiType();
  else
    return ret.first->second;
}


// Adds a new netCDF Enum type.
NcmpiEnumType NcmpiGroup::addEnumType(const string& name,NcmpiEnumType::ncmpiEnumType baseType) const {
  ncmpiCheckDefineMode(myId);
  nc_type typeId;
  ncmpiCheck(ncmpi_def_enum(myId, baseType, name.c_str(), &typeId),__FILE__,__LINE__);
  NcmpiEnumType ncmpiTypeTmp(*this,name);
  return ncmpiTypeTmp;
}


// Adds a new netCDF Vlen type.
NcmpiVlenType NcmpiGroup::addVlenType(const string& name,NcmpiType& baseType) const {
  ncmpiCheckDefineMode(myId);
  nc_type typeId;
  ncmpiCheck(ncmpi_def_vlen(myId,  const_cast<char*>(name.c_str()),baseType.getId(),&typeId),__FILE__,__LINE__);
  NcmpiVlenType ncmpiTypeTmp(*this,name);
  return ncmpiTypeTmp;
}


// Adds a new netCDF Opaque type.
NcmpiOpaqueType NcmpiGroup::addOpaqueType(const string& name, MPI_Offset size) const {
  ncmpiCheckDefineMode(myId);
  nc_type typeId;
  ncmpiCheck(ncmpi_def_opaque(myId, size,const_cast<char*>(name.c_str()), &typeId),__FILE__,__LINE__);
  NcmpiOpaqueType ncmpiTypeTmp(*this,name);
  return ncmpiTypeTmp;
}
    
// Adds a new netCDF UserDefined type.
NcmpiCompoundType NcmpiGroup::addCompoundType(const string& name, MPI_Offset size) const {
  ncmpiCheckDefineMode(myId);
  nc_type typeId;
  ncmpiCheck(ncmpi_def_compound(myId, size,const_cast<char*>(name.c_str()),&typeId),__FILE__,__LINE__);
  NcmpiCompoundType ncmpiTypeTmp(*this,name);
  return ncmpiTypeTmp;
}
  
  
// Get the collection of coordinate variables.
map<string,NcmpiGroup> NcmpiGroup::getCoordVars(NcmpiGroup::Location location) const {
  map<string,NcmpiGroup> coordVars;

  // search in current group and parent groups.
  NcmpiGroup tmpGroup(*this); 
  multimap<string,NcmpiDim>::iterator itD;
  multimap<string,NcmpiVar>::iterator itV;
  while(1) {
    // get the collection of NcmpiDim objects defined in this group.
    multimap<string,NcmpiDim> dimTmp(tmpGroup.getDims());
    multimap<string,NcmpiVar> varTmp(tmpGroup.getVars());
    for (itD=dimTmp.begin();itD!=dimTmp.end();itD++) {
      string coordName(itD->first);
      itV = varTmp.find(coordName);
      if(itV != varTmp.end()) {
	coordVars.insert(pair<const string,NcmpiGroup>(string(coordName),tmpGroup));
      }
    }
    if(location != ParentsAndCurrent || location != All || tmpGroup.isRootGroup()) {
      break;
    }
    // continue loop with the parent.
    tmpGroup=tmpGroup.getParentGroup();
  }

  // search in child groups (makes recursive calls).
  if(location == ChildrenAndCurrent || location == All ) {
    multimap<string,NcmpiGroup>::iterator it;
    multimap<string,NcmpiGroup> groups(getGroups());
    for (it=groups.begin();it!=groups.end();it++) {
      map<string,NcmpiGroup> coordVarsTmp=getCoordVars(ChildrenAndCurrent);
      coordVars.insert(coordVarsTmp.begin(),coordVarsTmp.end());
    }
  }
  
  return coordVars;
}

// Get the NcmpiDim and NcmpiVar object pair for a named coordinate variables.
void NcmpiGroup::getCoordVar(string& coordVarName, NcmpiDim& ncmpiDim, NcmpiVar& ncmpiVar, NcmpiGroup::Location location) const {

  // search in current group and parent groups.
  multimap<string,NcmpiDim>::iterator itD;
  NcmpiGroup tmpGroup(*this); 
  multimap<string,NcmpiVar>::iterator itV;
  while(1) {
    // get the collection of NcmpiDim objects defined in this group.
    multimap<string,NcmpiDim> dimTmp(tmpGroup.getDims());
    multimap<string,NcmpiVar> varTmp(tmpGroup.getVars());
    itD=dimTmp.find(coordVarName);
    itV=varTmp.find(coordVarName);
    if(itD != dimTmp.end() && itV != varTmp.end()) {
      ncmpiDim=itD->second;
      ncmpiVar=itV->second;
      return;
    }
    if(location != ParentsAndCurrent || location != All || tmpGroup.isRootGroup()) {
      break;
    }
    // continue loop with the parent.
    tmpGroup=tmpGroup.getParentGroup();
  }

  // search in child groups (makes recursive calls).
  if(location == ChildrenAndCurrent || location == All ) {
    multimap<string,NcmpiGroup>::iterator it;
    multimap<string,NcmpiGroup> groups(getGroups());
    for (it=groups.begin();it!=groups.end();it++) {
      getCoordVar(coordVarName,ncmpiDim,ncmpiVar,ChildrenAndCurrent);
      if(!ncmpiDim.isNull()) break;
    }
  }

  if(ncmpiDim.isNull()) {
    // return null objects as no coordinates variables were obtained.
    NcmpiDim dimTmp;
    NcmpiVar varTmp;
    ncmpiDim=dimTmp;
    ncmpiVar=varTmp;
    return;
  }

}

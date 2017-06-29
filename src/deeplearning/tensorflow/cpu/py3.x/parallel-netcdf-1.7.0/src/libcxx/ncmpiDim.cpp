#include "ncmpiDim.h"
#include "ncmpiGroup.h"
#include "ncmpiCheck.h"
#include <algorithm>
using namespace std;


namespace PnetCDF {
  //  Global comparator operator ==============
  // comparator operator 
  bool operator<(const NcmpiDim& lhs,const NcmpiDim& rhs)
  {
    return false;
  }
  
  // comparator operator 
  bool operator>(const NcmpiDim& lhs,const NcmpiDim& rhs)
  {
    return true;
  }
}

using namespace PnetCDF;

// assignment operator
NcmpiDim& NcmpiDim::operator=(const NcmpiDim & rhs)
{
  nullObject = rhs.nullObject;
  myId = rhs.myId;
  groupId = rhs.groupId;
  return *this;
}

// The copy constructor.
NcmpiDim::NcmpiDim(const NcmpiDim& rhs): 
  nullObject(rhs.nullObject),
  myId(rhs.myId),
  groupId(rhs.groupId)
{}


// equivalence operator
bool NcmpiDim::operator==(const NcmpiDim& rhs) const
{
  if(nullObject) 
    return nullObject == rhs.nullObject;
  else
    return myId == rhs.myId && groupId == rhs.groupId;
}  

//  !=  operator
bool NcmpiDim::operator!=(const NcmpiDim & rhs) const
{
  return !(*this == rhs);
}  


// Gets parent group.
NcmpiGroup  NcmpiDim::getParentGroup() const {
  return NcmpiGroup(groupId);
}

// Constructor generates a null object.
NcmpiDim::NcmpiDim() : 
  nullObject(true),
  myId(-1),
  groupId(-1)
{}

// Constructor for a dimension (must already exist in the netCDF file.)
NcmpiDim::NcmpiDim(const NcmpiGroup& grp, int dimId) :
  nullObject(false)
{
  groupId = grp.getId();
  myId = dimId;
}

// gets the size of the dimension, for unlimited, this is the current number of records.
MPI_Offset NcmpiDim::getSize() const
{
  MPI_Offset dimSize;
  ncmpiCheck(ncmpi_inq_dimlen(groupId, myId, &dimSize),__FILE__,__LINE__);
  return dimSize;
}


// returns true if this dimension is unlimited.
bool NcmpiDim::isUnlimited() const
{
  int dimid;
  ncmpiCheck(ncmpi_inq_unlimdim(groupId,&dimid),__FILE__,__LINE__);
  return (myId == dimid);

#if 0
  int numlimdims;
  int* unlimdimidsp=NULL;
  // get the number of unlimited dimensions
  ncmpiCheck(ncmpi_inq_unlimdims(groupId,&numlimdims,unlimdimidsp),__FILE__,__LINE__);
  if (numlimdims){
	  // get all the unlimited dimension ids in this group
	  vector<int> unlimdimid(numlimdims);
	  ncmpiCheck(ncmpi_inq_unlimdims(groupId,&numlimdims,&unlimdimid[0]),__FILE__,__LINE__);
	  vector<int>::iterator it;
	  // now look to see if this dimension is unlimited
	  it = find(unlimdimid.begin(),unlimdimid.end(),myId);
	  return it != unlimdimid.end();
  }
  return false;
#endif
}


// gets the name of the dimension.
const string NcmpiDim::getName() const
{
  char dimName[NC_MAX_NAME+1];
  ncmpiCheck(ncmpi_inq_dimname(groupId, myId, dimName),__FILE__,__LINE__);
  return string(dimName);
}
  
// renames this dimension.
void NcmpiDim::rename(const string& name)
{
  ncmpiCheck(ncmpi_rename_dim(groupId, myId, name.c_str()),__FILE__,__LINE__);
}



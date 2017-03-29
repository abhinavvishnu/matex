#include "ncmpiUint.h"
#include <pnetcdf.h>
using namespace PnetCDF;

// create an instance of NcmpiUint  called PnetCDF::ncmpiUint
namespace PnetCDF {
  NcmpiUint ncmpiUint;
}

// constructor
NcmpiUint::NcmpiUint() : NcmpiType(NC_UINT){
}

NcmpiUint::~NcmpiUint() {
}


// equivalence operator
bool NcmpiUint::operator==(const NcmpiUint & rhs)    {
  // simply check the netCDF id.
  return myId == rhs.myId;
}  

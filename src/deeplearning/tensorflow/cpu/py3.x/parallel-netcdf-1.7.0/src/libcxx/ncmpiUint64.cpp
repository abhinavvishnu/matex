#include "ncmpiUint64.h"
#include <pnetcdf.h>
using namespace PnetCDF;

// create an instance of NcmpiUint64  called PnetCDF::ncmpiUint64
namespace PnetCDF {
  NcmpiUint64 ncmpiUint64;
}

// constructor
NcmpiUint64::NcmpiUint64() : NcmpiType(NC_UINT64){
}

NcmpiUint64::~NcmpiUint64() {
}


// equivalence operator
bool NcmpiUint64::operator==(const NcmpiUint64 & rhs)    {
  // simply check the netCDF id.
  return myId == rhs.myId;
}  

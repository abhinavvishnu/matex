#include "ncmpiInt64.h"
#include <pnetcdf.h>
using namespace PnetCDF;

// create an instance of NcmpiInt64  called PnetCDF::ncmpiInt64
namespace PnetCDF {
  NcmpiInt64 ncmpiInt64;
}

// constructor
NcmpiInt64::NcmpiInt64() : NcmpiType(NC_INT64){
}

NcmpiInt64::~NcmpiInt64() {
}


// equivalence operator
bool NcmpiInt64::operator==(const NcmpiInt64 & rhs)    {
  // simply check the netCDF id.
  return myId == rhs.myId;
}  

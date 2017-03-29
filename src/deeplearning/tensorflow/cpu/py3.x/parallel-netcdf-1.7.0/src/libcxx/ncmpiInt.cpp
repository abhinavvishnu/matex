#include "ncmpiInt.h"
#include <pnetcdf.h>
using namespace PnetCDF;

// create an instance of NcmpiInt  called PnetCDF::ncmpiInt
namespace PnetCDF {
  NcmpiInt ncmpiInt;
}

// constructor
NcmpiInt::NcmpiInt() : NcmpiType(NC_INT){
}

NcmpiInt::~NcmpiInt() {
}


// equivalence operator
bool NcmpiInt::operator==(const NcmpiInt & rhs)    {
  // simply check the netCDF id.
  return myId == rhs.myId;
}  

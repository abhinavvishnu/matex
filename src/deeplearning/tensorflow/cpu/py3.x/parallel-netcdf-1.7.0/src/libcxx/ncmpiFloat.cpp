#include "ncmpiFloat.h"
#include <pnetcdf.h>
using namespace PnetCDF;

// create an instance of NcmpiFloat  called PnetCDF::ncmpiFloat
namespace PnetCDF {
  NcmpiFloat ncmpiFloat;
}

// constructor
NcmpiFloat::NcmpiFloat() : NcmpiType(NC_FLOAT){
}

NcmpiFloat::~NcmpiFloat() {
}


// equivalence operator
bool NcmpiFloat::operator==(const NcmpiFloat & rhs)    {
  // simply check the netCDF id.
  return myId == rhs.myId;
}  

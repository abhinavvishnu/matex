#include "ncmpiShort.h"
#include <pnetcdf.h>
using namespace PnetCDF;

// create an instance of NcmpiShort  called PnetCDF::ncmpiShort
namespace PnetCDF {
  NcmpiShort ncmpiShort;
}

// constructor
NcmpiShort::NcmpiShort() : NcmpiType(NC_SHORT){
}

NcmpiShort::~NcmpiShort() {
}


// equivalence operator
bool NcmpiShort::operator==(const NcmpiShort & rhs)    {
  // simply check the netCDF id.
  return myId == rhs.myId;
}  

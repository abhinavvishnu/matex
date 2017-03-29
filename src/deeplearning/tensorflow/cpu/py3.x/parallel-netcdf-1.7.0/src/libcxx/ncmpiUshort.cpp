#include "ncmpiUshort.h"
#include <pnetcdf.h>
using namespace PnetCDF;

// create an instance of NcmpiUshort  called PnetCDF::ncmpiUshort
namespace PnetCDF {
  NcmpiUshort ncmpiUshort;
}

// constructor
NcmpiUshort::NcmpiUshort() : NcmpiType(NC_USHORT){
}

NcmpiUshort::~NcmpiUshort() {
}


// equivalence operator
bool NcmpiUshort::operator==(const NcmpiUshort & rhs)    {
  // simply check the netCDF id.
  return myId == rhs.myId;
}  

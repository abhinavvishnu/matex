#include "ncmpiDouble.h"
#include <pnetcdf.h>
using namespace PnetCDF;

// create an instance of NcmpiDouble  called PnetCDF::ncmpiDouble
namespace PnetCDF {
  NcmpiDouble ncmpiDouble;
}

// constructor
NcmpiDouble::NcmpiDouble() : NcmpiType(NC_DOUBLE){
}

NcmpiDouble::~NcmpiDouble() {
}


// equivalence operator
bool NcmpiDouble::operator==(const NcmpiDouble & rhs)    {
  // simply check the netCDF id.
  return myId == rhs.myId;
}  

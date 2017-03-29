#include "ncmpiUbyte.h"
#include <pnetcdf.h>
using namespace PnetCDF;

// create an instance of NcmpiUbyte  called PnetCDF::ncmpiUbyte
namespace PnetCDF {
  NcmpiUbyte ncmpiUbyte;
}

// constructor
NcmpiUbyte::NcmpiUbyte() : NcmpiType(NC_UBYTE){
}

NcmpiUbyte::~NcmpiUbyte() {
}


// equivalence operator
bool NcmpiUbyte::operator==(const NcmpiUbyte & rhs)    {
  // simply check the netCDF id.
  return myId == rhs.myId;
}  

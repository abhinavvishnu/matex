#include "ncmpiChar.h"
#include <pnetcdf.h>
using namespace PnetCDF;

// create an instance of NcmpiChar  called PnetCDF::ncmpiChar
namespace PnetCDF {
  NcmpiChar ncmpiChar;
}

// constructor
NcmpiChar::NcmpiChar() : NcmpiType(NC_CHAR){
}

NcmpiChar::~NcmpiChar() {
}


// equivalence operator
bool NcmpiChar::operator==(const NcmpiChar & rhs)    {
  // simply check the netCDF id.
  return myId == rhs.myId;
}  

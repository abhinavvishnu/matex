#include "ncmpiByte.h"
#include <pnetcdf.h>
using namespace PnetCDF;

// create an instance of NcmpiByte  called PnetCDF::ncmpiByte
namespace PnetCDF {
  NcmpiByte ncmpiByte;
}

// constructor
NcmpiByte::NcmpiByte() : NcmpiType(NC_BYTE){
}

NcmpiByte::~NcmpiByte() {
}

int NcmpiByte::sizeoff(){char a;return sizeof(a);};


// equivalence operator
bool NcmpiByte::operator==(const NcmpiByte & rhs)    {
  // simply check the netCDF id.
  return myId == rhs.myId;
}  

#include <string>
#include "ncmpiType.h"

#ifndef NcmpiVlenTypeClass
#define NcmpiVlenTypeClass


namespace PnetCDF
{
  class NcmpiGroup;  // forward declaration.

  /*! Class represents a netCDF VLEN type */
  class NcmpiVlenType : public NcmpiType
  {
  public:

    /*! Constructor generates a \ref isNull "null object". */
    NcmpiVlenType();

    /*! 
      Constructor.
      The vlen Type must already exist in the netCDF file. New netCDF vlen types can be added 
      using NcmpiGroup::addNcmpiVlenType();
      \param grp        The parent group where this type is defined.
      \param name       Name of new type.
    */
    NcmpiVlenType(const NcmpiGroup& grp, const std::string& name);

    /*! 
      Constructor.
      Constructs from the base type NcmpiType object. Will throw an exception if the NcmpiType is not the base of a Vlen type.
      \param ncmpiType     A Nctype object.
    */
    NcmpiVlenType(const NcmpiType& ncmpiType);

    /*! assignment operator */
    NcmpiVlenType& operator=(const NcmpiVlenType& rhs);
      
    /*! 
      Assignment operator.
      This assigns from the base type NcmpiType object. Will throw an exception if the NcmpiType is not the base of a Vlen type.
    */
    NcmpiVlenType& operator=(const NcmpiType& rhs);
      
    /*! The copy constructor. */
    NcmpiVlenType(const NcmpiVlenType& rhs);
      
    ~NcmpiVlenType(){;}
      
    /*! Returns the base type. */
    NcmpiType  getBaseType() const;
      
  };
  
}

#endif

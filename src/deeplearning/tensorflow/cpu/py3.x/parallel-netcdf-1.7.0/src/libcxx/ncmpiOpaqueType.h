#include <string>
#include "ncmpiType.h"

#ifndef NcmpiOpaqueTypeClass
#define NcmpiOpaqueTypeClass


namespace PnetCDF
{
  class NcmpiGroup;  // forward declaration.

  /*! Class represents a netCDF opaque type */
  class NcmpiOpaqueType : public NcmpiType
  {
  public:

    /*! Constructor generates a \ref isNull "null object". */
    NcmpiOpaqueType();

    /*! 
      Constructor.
      The opaque Type must already exist in the netCDF file. New netCDF opaque types #
      can be added using NcmpiGroup::addNcmpiOpaqueType();
      \param grp        The parent group where this type is defined.
      \param name       Name of new type.
    */
    NcmpiOpaqueType(const NcmpiGroup& grp, const std::string& name);

    /*! 
      Constructor.
      Constructs from the base type NcmpiType object. Will throw an exception if the NcmpiType is not the base of a Opaque type.
      \param ncmpiType     A Nctype object.
    */
    NcmpiOpaqueType(const NcmpiType& ncmpiType);

    /*! assignment operator */
    NcmpiOpaqueType& operator=(const NcmpiOpaqueType& rhs);
      
    /*! 
      Assignment operator.
      This assigns from the base type NcmpiType object. Will throw an exception if the NcmpiType is not the base of an Opaque type.
    */
    NcmpiOpaqueType& operator=(const NcmpiType& rhs);
      
    /*! The copy constructor.*/
    NcmpiOpaqueType(const NcmpiOpaqueType& rhs);
      
    /*!  destructor */
    ~NcmpiOpaqueType(){;}

    /*! Returns the size of the opaque type in bytes. */
    MPI_Offset  getTypeSize() const;

  };
  
}

#endif

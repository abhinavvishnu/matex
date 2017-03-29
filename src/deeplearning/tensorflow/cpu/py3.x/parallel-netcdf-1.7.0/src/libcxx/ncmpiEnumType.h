#include <string>
#include "ncmpiType.h"
#include "ncmpiCheck.h"

#include <pnetcdf.h>
#include "ncmpi_notyet.h"

#ifndef NcmpiEnumTypeClass
#define NcmpiEnumTypeClass


namespace PnetCDF
{
  class NcmpiGroup;  // forward declaration.

  /*! Class represents a netCDF enum type */
  class NcmpiEnumType : public NcmpiType
    {
    public:
      
      /*! List of NetCDF-4 Enumeration types.*/
      enum ncmpiEnumType	{
	ncmpi_BYTE     = NC_BYTE, 	//!< signed 1 byte integer
	ncmpi_SHORT    = NC_SHORT, 	//!< signed 2 byte integer
	ncmpi_INT      = NC_INT,	//!< signed 4 byte integer
	ncmpi_UBYTE    = NC_UBYTE,	//!< unsigned 1 byte int
	ncmpi_USHORT   = NC_USHORT,	//!< unsigned 2-byte int
	ncmpi_UINT     = NC_UINT,	//!< unsigned 4-byte int
	ncmpi_INT64    = NC_INT64,	//!< signed 8-byte int
	ncmpi_UINT64   = NC_UINT64	//!< unsigned 8-byte int
      };
      
      /*! Constructor generates a \ref isNull "null object". */
      NcmpiEnumType();

      /*! 
	Constructor.
	The enum Type must already exist in the netCDF file. New netCDF enum types can 
	be added using NcmpiGroup::addNcmpiEnumType();
	\param grp        The parent group where this type is defined.
	\param name       Name of new type.
      */
      NcmpiEnumType(const NcmpiGroup& grp, const std::string& name);

      /*! 
	Constructor.
	Constructs from the base type NcmpiType object. Will throw an exception if the NcmpiType is not the base of an Enum type.
	\param ncmpiType     A Nctype object.
      */
      NcmpiEnumType(const NcmpiType& ncmpiType);

      /*! assignment operator */
      NcmpiEnumType& operator=(const NcmpiEnumType& rhs);
      
      /*! 
	Assignment operator.
       This assigns from the base type NcmpiType object. Will throw an exception if the NcmpiType is not the base of an Enum type.
      */
      NcmpiEnumType& operator=(const NcmpiType& rhs);
      
      /*! The copy constructor. */
      NcmpiEnumType(const NcmpiEnumType& rhs);
      
      /*! Destructor */
      ~NcmpiEnumType(){}
      
      
      /*! 
	Adds a new member to this NcmpiEnumType type.
	\param name         Name for this new Enum memebr.
	\param memberValue  Member value, must be of the correct NcmpiType.
      */
      template <class T> void addMember(const std::string& name, T memberValue)
      {
	ncmpiCheck(ncmpi_insert_enum(groupId, myId, name.c_str(), (void*) &memberValue),__FILE__,__LINE__);
      }

      /*! Returns number of members in this NcmpiEnumType object. */
      MPI_Offset  getMemberCount() const;
      
      /*! Returns the member name for the given zero-based index. */
      std::string  getMemberNameFromIndex(int index) const;

      /*! Returns the member name for the given NcmpiEnumType value. */
      template <class T>  std::string  getMemberNameFromValue(const T memberValue) const {
	char charName[NC_MAX_NAME+1];
	ncmpiCheck(ncmpi_inq_enum_ident(groupId,myId,static_cast<long long>(memberValue),charName),__FILE__,__LINE__);
	return std::string(charName);
      }
	
      /*! 
	Returns the value of a member with the given zero-based index.
	\param name         Name for this new Enum member.
	\param memberValue  Member value, returned by this routine.
      */
      template <class T> void getMemberValue(int index, T& memberValue) const
	{
	  char* charName=NULL;
	  ncmpiCheck(ncmpi_inq_enum_member(groupId,myId,index,charName,&memberValue),__FILE__,__LINE__);
	}

      /*! Returns the base type. */
      NcmpiType  getBaseType() const;
      
  };
  
}

#endif

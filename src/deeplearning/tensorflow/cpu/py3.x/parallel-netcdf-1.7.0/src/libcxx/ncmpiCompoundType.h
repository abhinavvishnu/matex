#include <string>
#include <vector>
#include "ncmpiType.h"

#ifndef NcmpiCompoundTypeClass
#define NcmpiCompoundTypeClass


namespace PnetCDF
{
  class NcmpiGroup;  // forward declaration.

  /*!
    Class represents a netCDF compound type
  */
  class NcmpiCompoundType : public NcmpiType
  {
  public:

    /*! Constructor generates a \ref isNull "null object". */
    NcmpiCompoundType();

    /*!
      Constructor.
      The compound Type must already exist in the netCDF file. New netCDF compound types can be
      added using NcmpiGroup::addNcmpiCompoundType();
      \param grp        The parent group where this type is defined.
      \param name       Name of new type.
    */
    NcmpiCompoundType(const NcmpiGroup& grp, const std::string& name);

    /*!
      Constructor.
      Constructs from the base type NcmpiType object. Will throw an exception if the NcmpiType is not the base of a Compound type.
      \param ncmpiType     A Nctype object.
    */
    NcmpiCompoundType(const NcmpiType& ncmpiType);

    /*! assignment operator */
    NcmpiCompoundType& operator=(const NcmpiCompoundType& rhs);

    /*!
      Assignment operator.
      This assigns from the base type NcmpiType object. Will throw an exception if the NcmpiType is not the base of a Compound type.
    */
    NcmpiCompoundType& operator=(const NcmpiType& rhs);

    /*! The copy constructor. */
    NcmpiCompoundType(const NcmpiCompoundType& rhs);

    /*! equivalence operator */
    bool operator==(const NcmpiCompoundType & rhs);

    /*! destructor */
    ~NcmpiCompoundType(){;}


    /*!
      Adds a named field.
      \param memName       Name of new field.
      \param newMemberType The type of the new member.
      \param offset        Offset of this member in bytes, obtained by a call to offsetof. For example
the offset of a member "mem4" in structure struct1 is: offsetof(struct1,mem4).
    */
    void addMember(const std::string& memName, const NcmpiType& newMemberType,MPI_Offset offset);

    /*!
      Adds a named array field.
      \param memName       Name of new field.
      \param newMemberType The type of the new member.
      \param offset        Offset of this member in bytes, obtained by a call to offsetof. For example
                           the offset of a member "mem4" in structure struct1 is: offsetof(struct1,mem4).
      \param shape         The shape of the array field.
    */
    void addMember(const std::string& memName, const NcmpiType& newMemberType, MPI_Offset offset, const std::vector<int>& shape);


    /*! Returns number of members in this NcmpiCompoundType object. */
    MPI_Offset  getMemberCount() const;

    /*! Returns a NcmpiType object for a single member. */
    NcmpiType getMember(int memberIndex) const;

    /*! Returns the offset of the member with given index. */
    MPI_Offset getMemberOffset(const int index) const;

    /*!
      Returns the number of dimensions of a member with the given index.
      \param Index of member (numbering starts at zero).
      \return The number of dimensions of the field. Non-array fields have 0 dimensions.
    */
    int getMemberDimCount(int memberIndex) const;


    /*!
      Returns the shape of a given member.
      \param Index of member (numbering starts at zero).
      \return The size of the dimensions of the field. Non-array fields have 0 dimensions.
    */
    std::vector<int> getMemberShape(int memberIndex) const;

  };

}


#endif

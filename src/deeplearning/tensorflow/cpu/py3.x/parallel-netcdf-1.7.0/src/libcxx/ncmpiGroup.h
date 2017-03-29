#include <string>
#include <vector>
#include <set>
#include <map>
#include "ncmpiType.h"
#include "ncmpiEnumType.h"
#include "ncmpiGroupAtt.h"



#ifndef NcmpiGroupClass
#define NcmpiGroupClass


namespace PnetCDF
{
  class NcmpiVar;          // forward declaration.
  class NcmpiDim;          // forward declaration.
  class NcmpiVlenType;     // forward declaration.
  class NcmpiCompoundType; // forward declaration.
  class NcmpiOpaqueType;   // forward declaration.

  /*! Class represents a netCDF group. */
  class NcmpiGroup
  {	
    
  public:
    
    /*!
      The enumeration list contains the options for selecting groups (used for returned set of NcmpiGroup objects).
    */
    enum GroupLocation
      {
	ChildrenGrps,              //!< Select from the set of children in the current group.
	ParentsGrps,               //!< Select from set of parent groups (excludes the current group).
	ChildrenOfChildrenGrps,    //!< Select from set of all children of children in the current group.
	AllChildrenGrps,           //!< Select from set of all children of the current group and beneath.
	ParentsAndCurrentGrps,     //!< Select from set of parent groups(includes the current group).
	AllGrps                    //!< Select from set of parent groups, current groups and all the children beneath.
      };
    
    /*!
      The enumeration list contains the options for selecting groups.
    */
    enum Location
      {
	Current,            //!< Select from contents of current group.
	Parents,            //!< Select from contents of parents groups.
	Children,           //!< Select from contents of children groups.
	ParentsAndCurrent,  //!< Select from contents of current and parents groups.
	ChildrenAndCurrent, //!< Select from contents of current and child groups.
	All                 //!< Select from contents of current, parents and child groups.
      };
    

    /*! assignment operator  */
    NcmpiGroup& operator=(const NcmpiGroup& rhs);
    
    /*! Constructor generates a \ref isNull "null object". */
    NcmpiGroup();

    //* constructor */
    NcmpiGroup(int groupId);

    /*! The copy constructor. */
    NcmpiGroup(const NcmpiGroup& rhs);

    /*! destructor  */
    virtual ~NcmpiGroup();
    
    /*! equivalence operator */
    bool operator==(const NcmpiGroup& rhs) const;     
    
    /*!  != operator */
    bool operator!=(const NcmpiGroup& rhs) const;     

    /*! comparator operator  */
    friend bool operator<(const NcmpiGroup& lhs,const NcmpiGroup& rhs);
    
    /*! comparator operator  */
    friend bool operator>(const NcmpiGroup& lhs,const NcmpiGroup& rhs);
    
    // /////////////
    // NcmpiGroup-related methods
    // /////////////
    
    /*! Gets the group name. */
    /*!
      Method will throw an PnetCDF::exceptions::NcNullgrp exception if the group is null (ie NcmpiGroup::isNull()=true).
      \param fullName If true then the full name is returned with subgroups separated by a forward slash "/" (default is false)
      \return         The group name.
    */
    std::string getName(bool fullName=false) const;

    /*! 
      Gets the parent group.
      Method will throw an PnetCDF::exceptions::NcNullgrp exception if the  group is null (ie NcmpiGroup::isNull()=true).
      If the current root is the parent group, then return a null group.
    */
    NcmpiGroup getParentGroup() const ;

    /*! 
      Gets the group id.
      Method will throw an PnetCDF::exceptions::NcNullgrp exception if the group is null (ie NcmpiGroup::isNull()=true).
    */
    int  getId() const;
    
    /*! 
      Gets the number of  NcmpiGroup objects.
      Method will throw an PnetCDF::exceptions::NcNullgrp exception if the group is null (ie NcmpiGroup::isNull()=true).
      \param location Enumeration type controlling the groups to search.
      \return         Number of groups.
    */
    int getGroupCount(NcmpiGroup::GroupLocation location=ChildrenGrps) const;

    /*! 
      Gets the collection of NcmpiGroup objects.
      Method will throw an PnetCDF::exceptions::NcNullgrp exception if the group is null (ie NcmpiGroup::isNull()=true).
      \param location Enumeration type controlling the groups to search.
      \return         A STL multimap object, containing pairs of <attribute name, NcmpiGroup object> entities.
    */
    std::multimap<std::string,NcmpiGroup> getGroups(NcmpiGroup::GroupLocation location=ChildrenGrps) const;


    /*! 
      Gets NcmpiGroup objects with a given name.
      Method will throw an PnetCDF::exceptions::NcNullgrp exception if the group is null (ie NcmpiGroup::isNull()=true).
      \param name     Name of group.
      \param location Enumeration type controlling the groups to search.
      \return         Set of NcmpiGroup objects with given name.
    */
    std::set<NcmpiGroup> getGroups(const std::string& name,NcmpiGroup::GroupLocation location=ChildrenGrps) const;

    /*!
      Gets the named child NcmpiGroup object.
      Method will throw an PnetCDF::exceptions::NcNullgrp exception if the group is null (ie NcmpiGroup::isNull()=true).
      \param name  Group name.
      \param location   Enumeration type controlling the groups to search.
      \return      An NcmpiGroup object. If there are multiple objects indentied with the same name, 
      the object closest to the current group is returned. If no valid object is found , 
      a \ref NcmpiGroup::isNull "null node" is returned.
    */
    NcmpiGroup getGroup(const std::string& name,NcmpiGroup::GroupLocation location=ChildrenGrps) const;

    /*! 
      Adds a new child netCDF group object.
      Method will throw an PnetCDF::exceptions::NcNullgrp exception if the group is null (ie NcmpiGroup::isNull()=true).
      \param   name     Variable name.
      \return  NcmpiGroup  The NcmpiGroup object for this new netCDF group.
    */
    NcmpiGroup addGroup(const std::string& name) const;


    /*! Returns true if this object is null (i.e. it has no contents); otherwise returns false. */
    bool isNull() const  {return nullObject;}

    /*! Returns true if this is the root group, otherwise returns false. */
    bool isRootGroup() const;

    // /////////////
    // NcmpiVar-related accessors
    // /////////////

    /*! 
      Gets the number of NcmpiVar objects in this group.
      \param location Enumeration type controlling the groups to search.
      \return         Number of variables.
    */
    int getVarCount(NcmpiGroup::Location location=Current) const;

    /*! 
      Gets the number of record variable NcmpiVar objects in this group.
      \param location Enumeration type controlling the groups to search.
      \return         Number of record variables.
    */
    int getRecVarCount(NcmpiGroup::Location location=Current) const;

    /*! 
      Gets the size of record block, i.e. the sume of single records of all
      the record variables.
      \param location Enumeration type controlling the groups to search.
      \return         size of record bock.
    */
    MPI_Offset getRecSize(NcmpiGroup::Location location=Current) const;

    /*! 
      Gets the number of fixed-size variable NcmpiVar objects in this group.
      \param location Enumeration type controlling the groups to search.
      \return         Number of fixed-size variables.
    */
    int getFixVarCount(NcmpiGroup::Location location=Current) const;

    /*!
      Get the collection of NcmpiVar objects.
      \param location Enumeration type controlling the groups to search.
      \return         A STL multimap object, containing pairs of <attribute name, NcmpiVar object> entities.
    */
   std::multimap<std::string,NcmpiVar> getVars(NcmpiGroup::Location location=Current) const;

   /*! 
     Gets all NcmpiVar objects with a given name.
      \param name     Name of attribute
      \param location Enumeration type controlling the groups to search.
      \return         Set of NcmpiVar objects.
    */
    std::set<NcmpiVar> getVars(const std::string& name,NcmpiGroup::Location location=Current) const;
    
    /*! 
      Gets the named NcmpiVar object..
      \param name     Variable name.
      \param location Enumeration type controlling the groups to search.
      \return         A NcmpiVar object. If there are multiple objects indentied with the 
      same name, the object closest  to the current group is returned.  
      If no valid object is found , a \ref NcmpiVar::isNull "null node" is returned.
     */
    NcmpiVar getVar(const std::string& name,NcmpiGroup::Location location=Current) const;

    /*! 
      Adds a new netCDF scalar variable.
      The NcmpiType must be non-null, and be defined in either the current group or a parent group.
      An NcNullType exception is thrown if the NcmpiType object is invalid.
      \param    name     Variable name.
      \param   typeName  Type name.
      \return            The NcmpiVar object for this new netCDF variable.
    */
    NcmpiVar addVar(const std::string& name, const NcmpiType& ncmpiType) const;

    /*!
      Adds a new netCDF variable.
      The NcmpiType and NcmpiDim objects must be non-null, and be defined in either the current group or a parent group.
      An NcNullType exception is thrown if the NcmpiType object is invalid.
      An NcNullDim exception is thrown if the NcmpiDim object is invalid.
      \param    name     Variable name.
      \param   typeName  Type name.
      \param   dimName   Dimension name.
      \return            The NcmpiVar object for this new netCDF variable.
    */
    NcmpiVar addVar(const std::string& name, const std::string& typeName, const std::string& dimName) const;

    /*! 
      Adds a new netCDF variable.
      The NcmpiType and NcmpiDim objects must be non-null, and be defined in either the current group or a parent group.
      An NcNullType exception is thrown if the NcmpiType object is invalid.
      An NcNullDim exception is thrown if the NcmpiDim object is invalid.
      \param    name      Variable name.
      \param    ncmpiType    NcmpiType object.
      \param    ncmpiDim     NcmpiDim object.
      \return             The NcmpiVar object for this new netCDF variable.
    */
    NcmpiVar addVar(const std::string& name, const NcmpiType& ncmpiType, const NcmpiDim& ncmpiDim) const;

    /*! 
      Adds a new netCDF multi-dimensional variable.
      The NcmpiType and NcmpiDim objects must be non-null, and be defined in either the current group or a parent group.
      An NcNullType exception is thrown if the NcmpiType object is invalid.
      An NcNullDim exception is thrown if the NcmpiDim object is invalid.
      \param   name     Variable name.
      \param   typeName Type name.
      \param   dimNames Vector of dimension names.
      \return           The NcmpiVar object for this new netCDF variable.
    */
    NcmpiVar addVar(const std::string& name, const std::string& typeName, const std::vector<std::string>& dimNames) const;
    

    /*! 
      Adds a new multi-dimensional netCDF variable.
      The NcmpiType and NcmpiDim objects must be non-null, and be defined in either the current group or a parent group.
      An NcNullType exception is thrown if the NcmpiType object is invalid.
      An NcNullDim exception is thrown if any of the the NcmpiDim objects are invalid.
      \param    name        Variable name.
      \param    ncmpiType      NcmpiType object.
      \param    ncmpiDimvector Vector of NcmpiDim objects.
      \return               The NcmpiVar object for this new netCDF variable.
    */
    NcmpiVar addVar(const std::string& name, const NcmpiType& ncmpiType, const std::vector<NcmpiDim>& ncmpiDimVector) const;

    // /////////////
    // NcmpiGroupAtt-related methods
    // /////////////

    /*! 
      Gets the number of group attributes.
      \param location Enumeration type controlling the groups to search.
      \return         Number of attributes.
    */
    int getAttCount(NcmpiGroup::Location location=Current) const;

    /*! 
      Gets the collection of NcmpiGroupAtt objects.
      \param location Enumeration type controlling the groups to search.
      \return         A STL multimap object, containing pairs of <attribute name, NcmpiGroupAtt object> entities.
    */
    std::multimap<std::string,NcmpiGroupAtt> getAtts(NcmpiGroup::Location location=Current) const;
    
    /*! 
    Gets all NcmpiGroupAtt objects with a given name.
      \param name     Name of attribute
      \param location Enumeration type controlling the groups to search.
      \return         Set of NcmpiGroupAtt objects.
    */
    std::set<NcmpiGroupAtt> getAtts(const std::string& name,NcmpiGroup::Location location=Current) const;
    
    /*! 
      Gets the named NcmpiGroupAtt object.
      \param name     Name of attribute
      \param location Enumeration type controlling the groups to search.
      \return         A NcmpiGroupAtt object. If there are multiple objects indentied with the 
      same name, the object closest  to the current group is returned.  If no valid object is found , 
      a \ref NcmpiGroupAtt::isNull "null node" is returned.
    */
    NcmpiGroupAtt getAtt(const std::string& name,NcmpiGroup::Location location=Current) const;
    

    /*! \overload
     */ 
    NcmpiGroupAtt putAtt(const std::string& name, MPI_Offset len, const char** dataValues) const ;

    /*! \overload
     */ 
    NcmpiGroupAtt putAtt(const std::string& name, const std::string& dataValues) const ;
    /*! \overload
     */ 
    NcmpiGroupAtt putAtt(const std::string& name, const NcmpiType& type, short datumValue) const ;
    /*! \overload
     */ 
    NcmpiGroupAtt putAtt(const std::string& name, const NcmpiType& type, int datumValue) const ;
    /*! \overload
     */ 
    NcmpiGroupAtt putAtt(const std::string& name, const NcmpiType& type, long datumValue) const ;
    /*! \overload
     */ 
    NcmpiGroupAtt putAtt(const std::string& name, const NcmpiType& type, float datumValue) const ;
    /*! \overload
     */ 
    NcmpiGroupAtt putAtt(const std::string& name, const NcmpiType& type, double datumValue) const ;
    /*! \overload
     */ 
    NcmpiGroupAtt putAtt(const std::string& name, const NcmpiType& type, unsigned short datumValue) const ;
    /*! \overload
     */ 
    NcmpiGroupAtt putAtt(const std::string& name, const NcmpiType& type, unsigned int datumValue) const ;
    /*! \overload
     */ 
    NcmpiGroupAtt putAtt(const std::string& name, const NcmpiType& type, unsigned long long datumValue) const ;
    /*! \overload
     */ 
    NcmpiGroupAtt putAtt(const std::string& name, const NcmpiType& type, long long datumValue) const ;
    /*! \overload
     */ 
    NcmpiGroupAtt putAtt(const std::string& name, const NcmpiType& type, MPI_Offset len, const unsigned char* dataValues) const ;
    /*! \overload
     */ 
    NcmpiGroupAtt putAtt(const std::string& name, const NcmpiType& type, MPI_Offset len, const signed char* dataValues) const ;
    /*! \overload
     */ 
    NcmpiGroupAtt putAtt(const std::string& name, const NcmpiType& type, MPI_Offset len, const short* dataValues) const ;
    /*! \overload
     */ 
    NcmpiGroupAtt putAtt(const std::string& name, const NcmpiType& type, MPI_Offset len, const int* dataValues) const ;
    /*! \overload
     */ 
    NcmpiGroupAtt putAtt(const std::string& name, const NcmpiType& type, MPI_Offset len, const long* dataValues) const ;
    /*! \overload
     */ 
    NcmpiGroupAtt putAtt(const std::string& name, const NcmpiType& type, MPI_Offset len, const float* dataValues) const ;
    /*! \overload
     */ 
    NcmpiGroupAtt putAtt(const std::string& name, const NcmpiType& type, MPI_Offset len, const double* dataValues) const ;
    /*! \overload
     */ 
    NcmpiGroupAtt putAtt(const std::string& name, const NcmpiType& type, MPI_Offset len, const unsigned short* dataValues) const ;
    /*! \overload
     */ 
    NcmpiGroupAtt putAtt(const std::string& name, const NcmpiType& type, MPI_Offset len, const unsigned int* dataValues) const ;
    /*! \overload
     */ 
    NcmpiGroupAtt putAtt(const std::string& name, const NcmpiType& type, MPI_Offset len, const unsigned long long* dataValues) const ;
    /*! \overload
     */ 
    NcmpiGroupAtt putAtt(const std::string& name, const NcmpiType& type, MPI_Offset len, const long long* dataValues) const ;
    /*! 
      Creates a new NetCDF group attribute or if already exisiting replaces it.
      If you are writing a _Fill_Value_ attribute, and will tell the HDF5 layer to use 
      the specified fill value for that variable. 
      \par 
      Although it's possible to create attributes of all types, text and double attributes are adequate for most purposes. 
      \param name        Name of attribute.
      \param type    The attribute type.
      \param len         The length of the attribute (number of Nctype repeats).
      \param dataValues  Data Values to put into the new attribute.
      If the type of data values differs from the netCDF variable type, type conversion will occur. 
      (However, no type conversion is carried out for variables using the user-defined data types:
      ncmpi_Vlen, ncmpi_Opaque, ncmpi_Compound and ncmpi_Enum.)
      \return            The NcmpiGroupAtt object for this new netCDF attribute.
    */
    NcmpiGroupAtt putAtt(const std::string& name, const NcmpiType& type, MPI_Offset len, const void* dataValues) const ;



    // /////////////
    // NcmpiDim-related methods
    // /////////////

    /*! 
      Gets the number of NcmpiDim objects.
      \param location Enumeration type controlling the groups to search.
      \return         Number of dimensions.
    */
    int getDimCount(NcmpiGroup::Location location=Current) const;
    
    /*! 
      Gets the collection of NcmpiDim objects.
      \param location Enumeration type controlling the groups to search.
      \return         A STL multimap object, containing pairs of <attribute name, NcmpiDim object> entities.
    */
    std::multimap<std::string,NcmpiDim> getDims(NcmpiGroup::Location location=Current) const;
    
    /*! 
      Gets NcmpiDim objects with a given name.
      \param name     Name of dimension.
      \param location Enumeration type controlling the groups to search.
      \return         Set of NcmpiDim objects with given name.
    */
    std::set<NcmpiDim> getDims(const std::string& name,NcmpiGroup::Location location=Current) const;

    /*! 
      Gets the named NcmpiDim object.
      \param name       Name of dimension.
      \param location   Enumeration type controlling the groups to search.
      \return           An NcmpiDim object. If there are multiple objects indentied with the same name, 
      the object closest to the current group is returned. If no valid object is found , a \ref NcmpiDim::isNull "null node" is returned.
    */
    NcmpiDim getDim(const std::string& name,NcmpiGroup::Location location=Current) const;

    /*! 
      Adds a new netCDF dimension.
      \param The name of new dimension.
      \param Length of dimension; that is, number of values for this dimension as an index to variables 
      that use it.
      \return   The NcmpiDim object for this new netCDF dimension.
    */
    NcmpiDim addDim(const std::string& name, MPI_Offset dimSize) const;

    /*! 
      Adds a new unlimited netCDF dimension.
      \param The name of new dimension.
      \return   The NcmpiDim object for this new netCDF dimension.
    */
    NcmpiDim addDim(const std::string& name) const;

    // /////////////
    // NcmpiType-related methods
    // /////////////

    /*! 
      Gets the number of type objects.
      \param location Enumeration type controlling the groups to search.
      \return         Number of types.
    */
    int getTypeCount(NcmpiGroup::Location location=Current) const;
    
    
    /*! 
      Gets the number of type objects with a given enumeration type.
      \param enumType The enumeration value of the object type.
      \param location Enumeration type controlling the groups to search.
      \return         Number of types of the given enumeration type.
    */
    int getTypeCount(NcmpiType::ncmpiType enumType, NcmpiGroup::Location location=Current) const;
    
    
    /*! 
      Gets the collection of NcmpiType objects.
      \param location Enumeration type controlling the groups to search.
      \return         A STL multimap object, on return contains pairs of <Type name, NcmpiType object> entities.
                      For atomic types, the type returned is the CDL name.
    */
    std::multimap<std::string,NcmpiType> getTypes(NcmpiGroup::Location location=Current) const;

  
    /*! 
      Gets the collection of NcmpiType objects with a given name.
      \param name     Name of type. For atomic types, the CDL name is expected. This is consistent with the 
                         string returned from NcmpiType::getName().
      \param location Enumeration type controlling the groups to search.
      \return         Set of  NcmpiType objects.
    */
    std::set<NcmpiType> getTypes(const std::string& name, NcmpiGroup::Location location=Current) const;

    /*! 
      Gets the collection of NcmpiType objects with a given data type.
      \param enumType Enumeration type specifying the data type.
      \param location Enumeration type controlling the groups to search.
      \return         Set of Nctype objects.
    */
    std::set<NcmpiType> getTypes(NcmpiType::ncmpiType enumType, NcmpiGroup::Location location=Current) const;


    /*! 
      Gets the collection of NcmpiType objects with a given name and data type.
      \param name     Name of type. For atomic types, the CDL name is expected. This is consistent with the 
                         string returned from NcmpiType::getName().
      \param enumType Enumeration type specifying the data type.
      \param location Enumeration type controlling the groups to search.
      \return         Set of Nctype objects.
    */
    std::set<NcmpiType> getTypes(const std::string& name, NcmpiType::ncmpiType enumType, NcmpiGroup::Location location=Current) const;


    /*! 
      Gets the NcmpiType object with a given name.
      \param name     Name of type. For atomic types, the CDL name is expected. This is consistent with the 
                         string returned from NcmpiType::getName().
      \param location Enumeration type controlling the groups to search.
      \return         NcmpiType object. If there are multiple objects indentied with the same name, 
      the object closest to the current group is returned.  If no valid object is found , a \ref NcmpiType::isNull "null node" is returned.

    */
    NcmpiType getType(const std::string& name, NcmpiGroup::Location location=Current) const;
    
    
    /*! 
      Adds a new netCDF enum type.
      \param name        Name of type. For atomic types, the CDL name is expected. This is consistent with the 
                         string returned from NcmpiType::getName().
      \param enumType    The enumeration value of the object type.
      \return            The NcmpiEnumType object for this new netCDF enum type.
    */
    NcmpiEnumType addEnumType(const std::string& name,NcmpiEnumType::ncmpiEnumType basetype) const;


    /*! 
      Adds a new netCDF Vlen type.
      \param name        Name of type.
      \param basetype    A NcmpiType object to be used for the basetype.
      \return            The NcmpiVlenType object for this new netCDF vlen type.
    */
    NcmpiVlenType addVlenType(const std::string& name,NcmpiType& basetype) const;


    /*! 
      Adds a new netCDF Opaque type.
      \param name     Name of type.
      \param size     The size of the new type in bytes.
      \return         The NcmpiOpaqueType object for this new netCDF opaque type..
    */
    NcmpiOpaqueType addOpaqueType(const std::string& name, MPI_Offset size) const;


    /*! 
      Adds a new netCDF UserDefined type.
      \param name     Name of type.
      \param size     The size of the new type in bytes.
      \return         The new NcmpiCompoundType object for this new netCDF userDefined type.
    */
    NcmpiCompoundType addCompoundType(const std::string& name, MPI_Offset size) const; 


    /*! 
      Gets a collection of  coordinate variables.
      Coordinate variable have  an NcmpiDim and NcmpiVar object with the same name defined in the same group.
      \par
      The method returns STL map object containing a coordinate variables in the current group  and optionally
      in the parent and child groups. It is expected that within each group, the names of dimensions are unique and
      the the names of variables are unique. However, if this is not the case, this method will still work correctly.
      
      \param location Enumeration type controlling the groups to search.
      \return         The NcmpiVar dimension variable. If no valid object is found , a \ref NcmpiVar::isNull "null node" is returned.
    */
    std::map<std::string,NcmpiGroup> getCoordVars(NcmpiGroup::Location location=Current) const;

    /*! 
      Gets the NcmpiDim and NcmpiVar object pair for a named coordinate variable.
      Coordinate variable have  an NcmpiDim and NcmpiVar object with the same name defined in the same group.
      \par
      The method returns two objects for the named coordinate variable. The method searches first in the current 
      group and optionally in the parent and child group and returns the first instance found.
      \param location Enumeration type controlling the groups to search.
      \return         The set of names of dimension variables. 
    */
    void getCoordVar(std::string& coordVarName, NcmpiDim& ncmpiDim, NcmpiVar& ncmpiVar, NcmpiGroup::Location location=Current) const;


  protected:
    
    /*! assignment operator  */
    /* NcmpiGroup& operator=(const NcmpiGroup& rhs); */
    
    bool nullObject;

    int myId;	
    
  };

}
#endif

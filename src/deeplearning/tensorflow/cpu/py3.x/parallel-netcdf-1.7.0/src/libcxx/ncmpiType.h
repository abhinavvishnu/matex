#include <string>
#include <pnetcdf.h>

#ifndef NcmpiTypeClass
#define NcmpiTypeClass


namespace PnetCDF
{
  class NcmpiGroup; // forward declaration to avoid cyclic reference.

  /*! Base class inherited by NcOpaque, NcVlen, NcCompound and NcEnum classes. */
  class NcmpiType
  {	
    
  public:
    
    /*!
      List of netCDF types that can be represented.
      The enumeration list contains the complete set of netCDF variable types. In addition, the type NC_TYPE
      is included. This enables the user to instantiate a netCDF type object without explcitly needing to know
      it precise type.
    */
    enum ncmpiType
    {
      ncmpi_BYTE     = NC_BYTE, 	//!< signed 1 byte integer
      ncmpi_CHAR     = NC_CHAR,		//!< ISO/ASCII character
      ncmpi_SHORT    = NC_SHORT, 	//!< signed 2 byte integer
      ncmpi_INT      = NC_INT,		//!< signed 4 byte integer
      ncmpi_FLOAT    = NC_FLOAT, 	//!< single precision floating point number
      ncmpi_DOUBLE   = NC_DOUBLE, 	//!< double precision floating point number
      ncmpi_UBYTE    = NC_UBYTE,	//!< unsigned 1 byte int
      ncmpi_USHORT   = NC_USHORT,	//!< unsigned 2-byte int
      ncmpi_UINT     = NC_UINT,		//!< unsigned 4-byte int
      ncmpi_INT64    = NC_INT64,	//!< signed 8-byte int
      ncmpi_UINT64   = NC_UINT64,	//!< unsigned 8-byte int

      // PnetCDF does not support types below
      ncmpi_STRING   = NC_STRING, 	//!< string
      ncmpi_VLEN     = NC_VLEN,   	//!< "NcVlen type"
      ncmpi_OPAQUE   = NC_OPAQUE, 	//!< "NcOpaque type"
      ncmpi_ENUM     = NC_ENUM, 	//!< "NcEnum type"
      ncmpi_COMPOUND = NC_COMPOUND	//!< "NcCompound type"
    };
    
    /*! Constructor generates a \ref isNull "null object". */
    NcmpiType();

    /*! 
      Constructor for a non-global type.
      This object describes the "essential" information for all netCDF types required by NcmpiVar, NcmpiAtt objects.
      New netCDF types can be added using the appropriate "add" method in the NcmpiGroup object.
      \param grp    Parent NcmpiGroup object.
      \param name   Name of this type.
    */
    NcmpiType(const PnetCDF::NcmpiGroup& grp, const std::string& name);


    /*! 
      Constructor for a non-global type.
      This object describes the "essential" information for all netCDF types required by NcmpiVar, NcmpiAtt objects.
      New netCDF types can be added using the appropriate "add" method in the NcmpiGroup object.
      \param grp    Parent NcmpiGroup object.
      \param id     type id
    */
    NcmpiType(const PnetCDF::NcmpiGroup& grp, nc_type id);

    /*! 
      Constructor for a global type
      This object describes the "essential" information for a netCDF global type.
      \param id     type id
    */
    NcmpiType(nc_type id);

    /*! The copy constructor. */
    NcmpiType(const NcmpiType& rhs);

    /*! destructor  */
    virtual ~NcmpiType() {}
    
    /*! equivalence operator */
    bool operator==(const NcmpiType&) const;     
    
    /*!  != operator */
    bool operator!=(const NcmpiType &) const;

    // accessors to private data.
    /*! The netCDF Id of this type. */
    nc_type getId() const {return myId;} 

    /*! Gets parent group. For an atomic type, returns a Null object.*/
    PnetCDF::NcmpiGroup getParentGroup() const;

    /*! 
      The name of this type. For atomic types, the CDL type names are returned. These are as follows: 
        - NcmpiByte   String returned is "byte".
        - NcmpiUbyte  String returned is "ubyte".
        - NcmpiChar   String returned is "char".   
        - NcmpiShort  String returned is "short".  
        - NcmpiUshort String returned is "ushort". 
        - NcmpiInt    String returned is "int".    
        - NcmpiUint   String returned is "uint".   
        - NcmpiInt64  String returned is "int64".  
        - NcmpiUint64 String returned is "uint64". 
        - NcmpiFloat  String returned is "float".  
        - NcmpiDouble String returned is "double". 
     */
    std::string getName() const;                        

    /*! 
      The size in bytes. 
      This function will work on any type, including atomic and any user defined type, whether 
      compound, opaque, enumeration, or variable length array. 
     */
    MPI_Offset getSize() const;                         

    /*! 
      The type class returned as enumeration type.
      Valid for all types, whether atomic or user-defined. User-defined types are returned as one of the following
      enumeration types: ncmpi_VLEN, ncmpi_OPAQUE, ncmpi_ENUM, or ncmpi_COMPOUND. 
     */
    ncmpiType getTypeClass() const;                       

    /*! 
      Return a string containing the name of the enumerated type.  (ie one of the following strings:
      "ncmpi_BYTE", "ncmpi_CHAR", "ncmpi_SHORT", "ncmpi_INT", "ncmpi_FLOAT", "ncmpi_DOUBLE", "ncmpi_UBYTE", "ncmpi_USHORT", 
      "ncmpi_UINT", "ncmpi_INT64", "ncmpi_UINT64", "ncmpi_STRING", "ncmpi_VLEN", "ncmpi_OPAQUE", "ncmpi_ENUM", "ncmpi_COMPOUND"
     */
    std::string getTypeClassName() const;
    
    /*! Returns true if this object is null (i.e. it has no contents); otherwise returns false. */
    bool isNull() const  {return nullObject;}

    /*! comparator operator  */
    friend bool operator<(const NcmpiType& lhs,const NcmpiType& rhs);
    
    /*! comparator operator  */
    friend bool operator>(const NcmpiType& lhs,const NcmpiType& rhs);
    
  protected:

    /*! assignment operator  */
    NcmpiType& operator=(const NcmpiType& rhs);
    
    bool nullObject;

    /*! the type Id */
    nc_type myId;
    
    /*! the group Id */
    int groupId;

  };

}
#endif


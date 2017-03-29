#include <exception>
#include <string>
#include <iostream>

#ifndef NcmpiExceptionClasses
#define NcmpiExceptionClasses

namespace PnetCDF
{

  //!  Exception classes.
  /*!
    These exceptions are thrown if the netCDF-4 API encounters an error.
  */
  namespace exceptions
  {

    /*! 
      Base object is thrown if a netCDF exception is encountered.
      An unsatisfactory return from a call to one of the netCDF C-routines 
      generates an exception using an object inheriting this class.  All other netCDF-related
      errors  including those originating in the C++ binding, generates an NcmpiException.
    */
    class NcmpiException : public std::exception {
    public:
      //NcmpiException(const string& complaint,const char* fileName,int lineNumber);
      NcmpiException(const char* complaint,const char* fileName,int lineNumber);
      NcmpiException(int errorCode, const char* complaint,const char* fileName,int lineNumber);
      NcmpiException(const NcmpiException& e) throw();
      NcmpiException& operator=(const NcmpiException& e) throw();
      virtual ~NcmpiException() throw();
      const char* what() const throw();
      int errorCode() const throw();
    private:
      std::string* what_msg;
      int ec;
    };


    /*! Thrown if the specified netCDF ID does not refer to an open netCDF dataset. */
    class NcBadId : public NcmpiException
    {
    public:
      NcBadId(const char* complaint,const char* file,int line);
    };

    /*! Thrown if too many netcdf files are open. */
    class NcNFile : public NcmpiException
    {
    public:
      NcNFile(const char* complaint,const char* file,int line);
    };

    /*! Thrown if, having set NC_NOCLOBBER, the specified dataset already exists. */
    class NcExist : public NcmpiException
    {
    public:
      NcExist(const char* complaint,const char* file,int line);
    };

    /*! Thrown if not a netCDF id.  */
    class NcInvalidArg : public NcmpiException
    {
    public:
      NcInvalidArg(const char* complaint,const char* file,int line);
    };

    /*! Thrown if invalid argument. */
    class NcInvalidWrite : public NcmpiException
    {
    public:
      NcInvalidWrite(const char* complaint,const char* file,int line);
    };

    /*! Thrown if operation not allowed in data mode. */
    class NcNotInDefineMode : public NcmpiException
    {
    public:
      NcNotInDefineMode(const char* complaint,const char* file,int line);
    };

    /*! Thrown if operation not allowed in defined mode. */
    class NcInDefineMode : public NcmpiException
    {
    public:
      NcInDefineMode(const char* complaint,const char* file,int line);
    };

    /*! 
      Index exceeds dimension bound.
      Exception may  be generated during operations to get or put  netCDF variable data.
      The exception is thrown if the specified indices were out of range for the rank of the 
      specified variable. For example, a negative index or an index that is larger than 
      the corresponding dimension length will cause an error.
    */
    class NcInvalidCoords : public NcmpiException
    {
    public:
      NcInvalidCoords(const char* complaint,const char* file,int line);
    };

    /*! Thrown if NC_MAX_DIMS is exceeded. */
    class NcMaxDims : public NcmpiException
    {
    public:
      NcMaxDims(const char* complaint,const char* file,int line);
    };

    /*! Thrown if string match to name is in use. */
    class NcNameInUse : public NcmpiException
    {
    public:
      NcNameInUse(const char* complaint,const char* file,int line);
    };

    /*! Thrown if attribute is not found. */
    class NcNotAtt : public NcmpiException
    {
    public:
      NcNotAtt(const char* complaint,const char* file,int line);
    };

    /*! Thrown if Nc_MAX_ATTRS is exceeded. */
    class NcMaxAtts : public NcmpiException
    {
    public:
      NcMaxAtts(const char* complaint,const char* file,int line);
    };

    /*! Thrown if not a valid netCDF data type. */
    class NcBadType : public NcmpiException
    {
    public:
      NcBadType(const char* complaint,const char* file,int line);
    };

    /*! Thrown if an invalid dimension id or name. */
    class NcBadDim : public NcmpiException
    {
    public:
      NcBadDim(const char* complaint,const char* file,int line);
    };

    /*! Thrown if Nc_UNLIMITED is in the wrong index. */
    class NcUnlimPos : public NcmpiException
    {
    public:
      NcUnlimPos(const char* complaint,const char* file,int line);
    };

    /*! Thrown if NC_MAX_VARS is exceeded. */
    class NcMaxVars : public NcmpiException
    {
    public:
      NcMaxVars(const char* complaint,const char* file,int line);
    };

    /*! Thrown if variable is not found. */
    class NcNotVar : public NcmpiException
    {
    public:
      NcNotVar(const char* complaint,const char* file,int line);
    };

    /*! Thrown if the action is prohibited on the NC_GLOBAL varid. */
    class NcGlobal : public NcmpiException
    {
    public:
      NcGlobal(const char* complaint,const char* file,int line);
    };

    /*! Thrown if not a netCDF file. */
    class NcNotNCF : public NcmpiException
    {
    public:
      NcNotNCF(const char* complaint,const char* file,int line);
    };

    /*! Thrown if in FORTRAN, string is too short. */
    class NcSts : public NcmpiException
    {
    public:
      NcSts(const char* complaint,const char* file,int line);
    };

    /*! Thrown if NC_MAX_NAME is exceeded. */
    class NcMaxName : public NcmpiException
    {
    public:
      NcMaxName(const char* complaint,const char* file,int line);
    };

    /*! Thrown if NC_UNLIMITED size is already in use. */
    class NcUnlimit : public NcmpiException
    {
    public:
      NcUnlimit(const char* complaint,const char* file,int line);
    };

    /*! Thrown if ncmpi_rec op when there are no record vars. */
    class NcNoRecVars : public NcmpiException
    {
    public:
      NcNoRecVars(const char* complaint,const char* file,int line);
    };

    /*! Thrown if attempt to convert between text and numbers. */
    class NcmpiChar : public NcmpiException
    {
    public:
      NcmpiChar(const char* complaint,const char* file,int line);
    };

    /*! Thrown if edge+start exceeds dimension bound. */
    class NcEdge : public NcmpiException
    {
    public:
      NcEdge(const char* complaint,const char* file,int line);
    };

    /*! Thrown if illegal stride. */
    class NcStride : public NcmpiException
    {
    public:
      NcStride(const char* complaint,const char* file,int line);
    };

    /*! Thrown if attribute or variable name contains illegal characters. */
    class NcBadName : public NcmpiException
    {
    public:
      NcBadName(const char* complaint,const char* file,int line);
    };

    /*! Thrown if math result not representable. */
    class NcRange : public NcmpiException
    {
    public:
      NcRange(const char* complaint,const char* file,int line);
    };

    /*! Thrown if memory allocation (malloc) failure. */
    class NcNoMem : public NcmpiException
    {
    public:
      NcNoMem(const char* complaint,const char* file,int line);
    };

    /*! Thrown if one or more variable sizes violate format constraints */
    class NcmpiVarSize : public NcmpiException
    {
    public:
      NcmpiVarSize(const char* complaint,const char* file,int line);
    };

    /*! Thrown if invalid dimension size. */
    class NcmpiDimSize : public NcmpiException
    {
    public:
      NcmpiDimSize(const char* complaint,const char* file,int line);
    };

    /*! Thrown if file likely truncated or possibly corrupted. */
    class NcTrunc : public NcmpiException
    {
    public:
      NcTrunc(const char* complaint,const char* file,int line);
    };

    /*! Thrown if an error was reported by the HDF5 layer. */
    class NcHdfErr : public NcmpiException
    {
    public:
      NcHdfErr(const char* complaint,const char* file,int line);
    };

    /*! Thrown if cannot read. */
    class NcCantRead : public NcmpiException
    {
    public:
      NcCantRead(const char* complaint,const char* file,int line);
    };

    /*! Thrown if cannot write. */
    class NcCantWrite : public NcmpiException
    {
    public:
      NcCantWrite(const char* complaint,const char* file,int line);
    };

    /*! Thrown if cannot create. */
    class NcCantCreate : public NcmpiException
    {
    public:
      NcCantCreate(const char* complaint,const char* file,int line);
    };

    /*! Thrown if file meta. */
    class NcmpiFileMeta : public NcmpiException
    {
    public:
      NcmpiFileMeta(const char* complaint,const char* file,int line);
    };

    /*! Thrown if dim meta. */
    class NcmpiDimMeta : public NcmpiException
    {
    public:
      NcmpiDimMeta(const char* complaint,const char* file,int line);
    };

    /*! Thrown if attribute meta. */
    class NcmpiAttMeta : public NcmpiException
    {
    public:
      NcmpiAttMeta(const char* complaint,const char* file,int line);
    };

    /*! Thrown if variable meta. */
    class NcmpiVarMeta : public NcmpiException
    {
    public:
      NcmpiVarMeta(const char* complaint,const char* file,int line);
    };

    /*! Thrown if no compound. */
    class NcNoCompound : public NcmpiException
    {
    public:
      NcNoCompound(const char* complaint,const char* file,int line);
    };

    /*! Thrown if attribute exists. */
    class NcmpiAttExists : public NcmpiException
    {
    public:
      NcmpiAttExists(const char* complaint,const char* file,int line);
    };

    /*! Thrown if attempting netcdf-4 operation on netcdf-3 file. */
    class NcNotNc4 : public NcmpiException
    {
    public:
      NcNotNc4(const char* complaint,const char* file,int line);
    };

    /*! Thrown if attempting netcdf-4 operation on strict nc3 netcdf-4 file. */
    class NcStrictNc3 : public NcmpiException
    {
    public:
      NcStrictNc3(const char* complaint,const char* file,int line);
    };

    /*! Thrown if bad group id. */
    class NcBadGroupId : public NcmpiException
    {
    public:
      NcBadGroupId(const char* complaint,const char* file,int line);
    };

    /*! Thrown if bad type id. */
    class NcBadTypeId : public NcmpiException
    {
    public:
      NcBadTypeId(const char* complaint,const char* file,int line);
    };

    /*! Thrown if bad field id. */
    class NcBadFieldId : public NcmpiException
    {
    public:
      NcBadFieldId(const char* complaint,const char* file,int line);
    };

    /*! Thrown if cannot find the field id. */
    class NcUnknownName : public NcmpiException
    {
    public:
      NcUnknownName(const char* complaint,const char* file,int line);
    };

    /*! Thrown if cannot return a netCDF group. */
    class NcEnoGrp : public NcmpiException
    {
    public:
      NcEnoGrp(const char* complaint,const char* file,int line);
    };

    /*! 
      Thrown if the requested operation is on a NULL group.
    
      This exception is thrown if an operation on a NcmpiGroup object is requested which is empty. To test if the object is empty used NcmpiGroup::isNull()
     */
    class NcNullGrp : public NcmpiException
    {
    public:
      NcNullGrp(const char* complaint,const char* file,int line);
    };

    /*! 
      Thrown if the requested operation is on a NULL type.
    
      This exception is thrown if an operation on a NcmpiType object is requested which is empty. To test if the object is empty used NcmpiType::isNull()
     */
    class NcNullType : public NcmpiException
    {
    public:
      NcNullType(const char* complaint,const char* file,int line);
    };

    /*! 
      Thrown if the requested operation is on a NULL dimension.
    
      This exception is thrown if an operation on a NcmpiDim object is requested which is empty. To test if the object is empty used NcmpiDim::isNull()
     */
    class NcNullDim : public NcmpiException
    {
    public:
      NcNullDim(const char* complaint,const char* file,int line);
    };

    /*! 
      Thrown if an operation to set the chunking, endianness, fill of a NcmpiVar object is issued after a 
      call to NcmpiVar::getVar or NcmpiVar::putVar has been made.
    */
    class NcElateDef : public NcmpiException
    {
    public:
      NcElateDef(const char* complaint,const char* file,int line);
    };

  }

}

#endif


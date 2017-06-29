#include <exception>
#include <string>
#include <typeinfo>
#include <map>
#include <vector>
#include <pnetcdf.h>
#include "ncmpiVarAtt.h"
#include "ncmpiGroup.h"
#include "ncmpiByte.h"
#include "ncmpiUbyte.h"
#include "ncmpiChar.h"
#include "ncmpiShort.h"
#include "ncmpiUshort.h"
#include "ncmpiInt.h"
#include "ncmpiUint.h"
#include "ncmpiInt64.h"
#include "ncmpiUint64.h"
#include "ncmpiFloat.h"
#include "ncmpiDouble.h"

#ifndef NcmpiVarClass
#define NcmpiVarClass

namespace PnetCDF
{
  //  class NcmpiGroup;  // forward declaration.
  class NcmpiDim;    // forward declaration.
  //  class NcmpiVarAtt; // forward declaration.
  class NcmpiType;   // forward declaration.

  /*! Class represents a netCDF variable. */
  class NcmpiVar
  {
  public:
    
    /*! Used for chunking specifications (see NcmpiVar::setChunking,  NcmpiVar::getChunkingParameters). */
    enum ChunkMode
      {
	/*!
	  Chunked storage is used for this variable.
	*/
	ncmpi_CHUNKED    = NC_CHUNKED,   
	/*! Contiguous storage is used for this variable. Variables with one or more unlimited 
	  dimensions cannot use contiguous storage. If contiguous storage is turned on, the 
	  chunkSizes parameter is ignored. 
	*/
	ncmpi_CONTIGUOUS = NC_CONTIGUOUS
      };

    /*!
      Used to specifying the endianess of the data, (see NcmpiVar::setEndianness, NcmpiVar::getEndianness). By default this is NC_ENDIAN_NATIVE.
    */
    enum EndianMode
      {
	ncmpi_ENDIAN_NATIVE = NC_ENDIAN_NATIVE, //!< Native endian.
	ncmpi_ENDIAN_LITTLE = NC_ENDIAN_LITTLE, //!< Little endian.
	ncmpi_ENDIAN_BIG    = NC_ENDIAN_BIG     //!< Big endian.
      };

    /*! Used for checksum specification (see NcmpiVar::setChecksum, NcmpiVar::getChecksum). */
    enum ChecksumMode
      {
	ncmpi_NOCHECKSUM = NC_NOCHECKSUM, //!< No checksum (the default).
	ncmpi_FLETCHER32 = NC_FLETCHER32  //!< Selects the Fletcher32 checksum filter. 
      };

    /*! destructor */
    ~NcmpiVar(){};
      
    /*! Constructor generates a \ref isNull "null object". */
    NcmpiVar ();

    /*! Constructor for a variable .
    
      The variable must already exist in the netCDF file. New netCDF variables can be added using NcmpiGroup::addNcmpiVar();
      \param grp    Parent NcmpiGroup object.
      \param varId  Id of the is NcmpiVar object.
    */
    NcmpiVar (const NcmpiGroup& grp, const int& varId);

    /*! assignment operator  */
    NcmpiVar& operator =(const NcmpiVar& rhs);
      
    /*! equivalence operator */
    bool operator==(const NcmpiVar& rhs) const;     
      
    /*!  != operator */
    bool operator!=(const NcmpiVar& rhs) const;     

    /*! The copy constructor. */
    NcmpiVar(const NcmpiVar& ncmpiVar);
      
    /*! Name of this NcmpiVar object.*/
    std::string getName() const;

    /*! Gets parent group. */
    NcmpiGroup  getParentGroup() const;

    /*! Returns the variable type. */
    NcmpiType getType() const;              
      
      
    /*! Rename the variable. */
    void rename( const std::string& newname ) const;
      

    /*! Get the variable id. */
    int  getId() const;
    
    /*! Returns true if this object variable is not defined. */
    bool isNull() const  {return nullObject;}

    /*! comparator operator  */
    friend bool operator<(const NcmpiVar& lhs,const NcmpiVar& rhs);
    
    /*! comparator operator  */
    friend bool operator>(const NcmpiVar& lhs,const NcmpiVar& rhs);
    
    /////////////////
      
    // Information about Dimensions
      
    /////////////////
      
    /*! The the number of dimensions. */
    int getDimCount() const ;

    /*! Gets the i'th NcmpiDim object. */
    NcmpiDim getDim(int i) const;

    /*! Gets the set of NcmpiDim objects. */
    std::vector<NcmpiDim> getDims() const;

    /////////////////
      
    // Information about Attributes
      
    /////////////////
      
    /*! Gets the number of attributes. */
    int getAttCount() const;
      
    /*! Gets attribute by name */
    NcmpiVarAtt getAtt(const std::string& name) const;

    /*! Gets the set of attributes. */
    std::map<std::string,NcmpiVarAtt> getAtts() const;




    /////////////////////////


    /*! \overload
     */ 
    NcmpiVarAtt putAtt(const std::string& name, const std::string& dataValues) const ;

    /*! \overload
     */ 
    NcmpiVarAtt putAtt(const std::string& name, const NcmpiType& type, MPI_Offset len, const unsigned char* dataValues) const ;
    /*! \overload
     */ 
    NcmpiVarAtt putAtt(const std::string& name, const NcmpiType& type, MPI_Offset len, const signed char* dataValues) const ;
    /*! \overload
     */ 
    NcmpiVarAtt putAtt(const std::string& name, const NcmpiType& type, short datumValue) const ;
    /*! \overload
     */ 
    NcmpiVarAtt putAtt(const std::string& name, const NcmpiType& type, int datumValue) const ;
    /*! \overload
     */ 
    NcmpiVarAtt putAtt(const std::string& name, const NcmpiType& type, long datumValue) const ;
    /*! \overload
     */ 
    NcmpiVarAtt putAtt(const std::string& name, const NcmpiType& type, float datumValue) const ;
    /*! \overload
     */ 
    NcmpiVarAtt putAtt(const std::string& name, const NcmpiType& type, double datumValue) const ;
    /*! \overload
     */ 
    NcmpiVarAtt putAtt(const std::string& name, const NcmpiType& type, unsigned short datumValue) const ;
    /*! \overload
     */ 
    NcmpiVarAtt putAtt(const std::string& name, const NcmpiType& type, unsigned int datumValue) const ;
    /*! \overload
     */ 
    NcmpiVarAtt putAtt(const std::string& name, const NcmpiType& type, unsigned long long datumValue) const ;
    /*! \overload
     */ 
    NcmpiVarAtt putAtt(const std::string& name, const NcmpiType& type, long long datumValue) const ;
    /*! \overload
     */ 
    NcmpiVarAtt putAtt(const std::string& name, const NcmpiType& type, MPI_Offset len, const short* dataValues) const ;
    /*! \overload
     */ 
    NcmpiVarAtt putAtt(const std::string& name, const NcmpiType& type, MPI_Offset len, const int* dataValues) const ;
    /*! \overload
     */ 
    NcmpiVarAtt putAtt(const std::string& name, const NcmpiType& type, MPI_Offset len, const long* dataValues) const ;
    /*! \overload
     */ 
    NcmpiVarAtt putAtt(const std::string& name, const NcmpiType& type, MPI_Offset len, const float* dataValues) const ;
    /*! \overload
     */ 
    NcmpiVarAtt putAtt(const std::string& name, const NcmpiType& type, MPI_Offset len, const double* dataValues) const ;
    /*! \overload
     */ 
    NcmpiVarAtt putAtt(const std::string& name, const NcmpiType& type, MPI_Offset len, const unsigned short* dataValues) const ;
    /*! \overload
     */ 
    NcmpiVarAtt putAtt(const std::string& name, const NcmpiType& type, MPI_Offset len, const unsigned int* dataValues) const ;
    /*! \overload
     */ 
    NcmpiVarAtt putAtt(const std::string& name, const NcmpiType& type, MPI_Offset len, const unsigned long long* dataValues) const ;
    /*! \overload
     */ 
    NcmpiVarAtt putAtt(const std::string& name, const NcmpiType& type, MPI_Offset len, const long long* dataValues) const ;
    /*! 
      Creates a new variable attribute or if already exisiting replaces it.
      If you are writing a _Fill_Value_ attribute, and will tell the HDF5 layer to use 
      the specified fill value for that variable. 
      \par 
      Although it's possible to create attributes of all types, text and double attributes are adequate for most purposes. 
      \param name        Name of attribute.
      \param type        The attribute type. 
      \param len         The length of the attribute (number of Nctype repeats).
      \param dataValues  Data Values to put into the new attribute.
      If the type of data values differs from the netCDF variable type, type conversion will occur. 
      (However, no type conversion is carried out for variables using the user-defined data types:
      ncmpi_Vlen, ncmpi_Opaque, ncmpi_Compound and ncmpi_Enum.)
      \return            The NcmpiVarAtt object for this new netCDF attribute.
    */
    NcmpiVarAtt putAtt(const std::string& name, const NcmpiType& type, MPI_Offset len, const void* dataValues) const ;



    ////////////////////
      
    // Chunking details
      
    ////////////////////
      
    /*! Sets chunking parameters.
      \param chunkMode   Enumeration type. Allowable parameters are: "ncmpi_CONTIGUOUS", "ncmpi_CHUNKED"
      \param chunksizes  Shape of chunking, used if ChunkMode=ncmpi_CHUNKED.
    */
    void setChunking(ChunkMode chunkMode, std::vector<MPI_Offset>& chunksizes) const;
      
    /*! Gets the chunking parameters
      \param chunkMode   On return contains either: "ncmpi_CONTIGUOUS" or "ncmpi_CHUNKED"
      \param chunksizes  On return contains shape of chunking, used if ChunkMode=ncmpi_CHUNKED.
    */
    void getChunkingParameters(ChunkMode& chunkMode, std::vector<MPI_Offset>& chunkSizes) const;
      
      
      
    ////////////////////
      
    // Fill details
      
    ////////////////////
      
    // Sets the fill parameters
    /*!
      \overload
    */
    void setFill(bool fillMode, void *fillValue=NULL) const;

    /*!
      This is an overloaded member function, provided for convenience. 
      It differs from the above function in what argument(s) it accepts. 
      The function can be used for any type, including user-defined types. 
      \param fillMode   Setting to true, turns on fill mode.
      \param fillValue  Pointer to fill value.
      Must be the same type as the variable. Ignored if fillMode=.false.
    */
    // void setFill(bool fillMode,const void* fillValue=NULL) const;

    /*! Sets the fill parameters
      \param fillMode   Setting to true, turns on fill mode.
      \param fillValue  Fill value for the variable. 
      Must be the same type as the variable. Ignored if fillMode=.false.
    */
/*
    template<class T>
      void setFill(bool fillMode, T fillValue) const
      {
	ncmpiCheck(ncmpi_def_var_fill(groupId,myId,static_cast<int> (!fillMode),&fillValue),__FILE__,__LINE__);
      }
*/

    
    
    /*!
      This is an overloaded member function, provided for convenience. 
      It differs from the above function in what argument(s) it accepts. 
      The function can be used for any type, including user-defined types. 
      \param fillMode   On return set to true  if fill mode is enabled.
      \param fillValue  On return containts a pointer to fill value.
      Must be the same type as the variable. Ignored if fillMode=.false.
    */
    void getFillModeParameters(bool& fillMode, void* fillValue=NULL) const;
    

    /*! Gets the fill parameters
      \param On return set to true  if fill mode is enabled.
      \param On return  is set to the fill value.
    */
    template <class T> void getFillModeParameters(bool& fillMode,T& fillValue) const{
       int fillModeInt;
      ncmpiCheck(ncmpi_inq_var_fill(groupId,myId,&fillModeInt,&fillValue),__FILE__,__LINE__);
      fillMode= static_cast<bool> (fillModeInt == 0);
    }
  

    /* fill variable */
    void fillRec(MPI_Offset recno) const;
      
    ////////////////////
      
    // Compression details
      
    ////////////////////
      
      
    /*! Sets the compression parameters
      \param enableShuffleFilter Set to true to turn on shuffle filter.
      \param enableDeflateFilter Set to true to turn on deflate filter.
      \param deflateLevel        The deflate level, must be 0 and 9.
    */
    void setCompression(bool enableShuffleFilter, bool enableDeflateFilter, int deflateLevel) const;
      
    /*! Gets the compression parameters
      \param enableShuffleFilter  On return set to true if the shuffle filter is enabled.
      \param enableDeflateFilter  On return set to true if the deflate filter is enabled.
      \param deflateLevel         On return set to the deflate level.
    */
    void getCompressionParameters(bool& shuffleFilterEnabled, bool& deflateFilterEnabled, int& deflateLevel) const;
      
      
      
    ////////////////////
      
    // Endianness details
      
    ////////////////////
      
      
    /*! Sets the endianness of the variable.
      \param Endianness enumeration type. Allowable parameters are: "ncmpi_ENDIAN_NATIVE" (the default), "ncmpi_ENDIAN_LITTLE", "ncmpi_ENDIAN_BIG"
    */
    void setEndianness(EndianMode endianMode) const;
      
    /*! Gets the endianness of the variable.
      \return Endianness enumeration type. Allowable parameters are: "ncmpi_ENDIAN_NATIVE" (the default), "ncmpi_ENDIAN_LITTLE", "ncmpi_ENDIAN_BIG"
    */
    EndianMode getEndianness() const;
      
      
      
    ////////////////////
      
    // Checksum details
      
    ////////////////////
      
      
    /*! Sets the checksum parameters of a variable.
      \param ChecksumMode Enumeration type. Allowable parameters are: "ncmpi_NOCHECKSUM", "ncmpi_FLETCHER32".
    */
    void setChecksum(ChecksumMode checksumMode) const;
      
    /*! Gets the checksum parameters of the variable.
      \return ChecksumMode Enumeration type. Allowable parameters are: "ncmpi_NOCHECKSUM", "ncmpi_FLETCHER32".
    */
    ChecksumMode getChecksum() const;
    
    
    
    ////////////////////
    
    //  data  reading
    
    ////////////////////
      
      

    // Reads the entire data into the netCDF variable. (independent data mode)
    /*!
      This is an overloaded member function, provided for convenience. 
      It differs from the above function in what argument(s) it accepts. 
      In addition, no data conversion is carried out. This means that 
      the type of the data in memory must match the type of the variable.
    */
    // void getVar(void* dataValues) const;
    /*! \overload
     */ 
    void getVar(char* dataValues) const;
    /*! \overload
     */ 
    void getVar(unsigned char* dataValues) const;
    /*! \overload
     */ 
    void getVar(signed char* dataValues) const;
    /*! \overload
     */ 
    void getVar(short* dataValues) const;
    /*! \overload
     */ 
    void getVar(int* dataValues) const;
    /*! \overload
     */ 
    void getVar(long* dataValues) const;
    /*! \overload
     */ 
    void getVar(float* dataValues) const;
    /*! \overload
     */ 
    void getVar(double* dataValues) const;
    /*! \overload
     */ 
    void getVar(unsigned short* dataValues) const;
    /*! \overload
     */ 
    void getVar(unsigned int* dataValues) const;
    /*! \overload
     */ 
    void getVar(unsigned long long* dataValues) const;
    /*! 
      Reads the entire data from an netCDF variable.
      This is the simplest interface to use for reading the value of a scalar variable 
      or when all the values of a multidimensional variable can be read at once. The values 
      are read into consecutive locations with the last dimension varying fastest. 
      
      Take care when using the simplest forms of this interface with record variables when you 
      don't specify how many records are to be read. If you try to read all the values of a 
      record variable into an array but there are more records in the file than you assume, 
      more data will be read than you expect, which may cause a segmentation violation.

      \param dataValues Pointer to the location into which the data value is read. If the type of 
      data value differs from the netCDF variable type, type conversion will occur.
      (However, no type conversion is carried out for variables using the user-defined data types:
      ncmpi_Vlen, ncmpi_Opaque, ncmpi_Compound and ncmpi_Enum.)
    */
    void getVar(long long* dataValues) const;

    void getVar(void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const;


    //////////////////////

    // Reads a single datum value from a variable of an open netCDF dataset.
    /*!
      This is an overloaded member function, provided for convenience. 
      It differs from the above function in what argument(s) it accepts. 
      In addition, no data conversion is carried out. This means that 
      the type of the data in memory must match the type of the variable.
    */
    // void getVar(const std::vector<MPI_Offset>& index, void* dataumValue) const;
    /*! \overload
     */ 
    void getVar(const std::vector<MPI_Offset>& index, char* dataumValue) const;
    /*! \overload
     */ 
    void getVar(const std::vector<MPI_Offset>& index, unsigned char* dataumValue) const;
    /*! \overload
     */ 
    void getVar(const std::vector<MPI_Offset>& index, signed char* dataumValue) const;
    /*! \overload
     */ 
    void getVar(const std::vector<MPI_Offset>& index, short* dataumValue) const;
    /*! \overload
     */ 
    void getVar(const std::vector<MPI_Offset>& index, int* dataumValue) const;
    /*! \overload
     */ 
    void getVar(const std::vector<MPI_Offset>& index, long* dataumValue) const;
    /*! \overload
     */ 
    void getVar(const std::vector<MPI_Offset>& index, float* dataumValue) const;
    /*! \overload
     */ 
    void getVar(const std::vector<MPI_Offset>& index, double* dataumValue) const;
    /*! \overload
     */ 
    void getVar(const std::vector<MPI_Offset>& index, unsigned short* dataumValue) const;
    /*! \overload
     */ 
    void getVar(const std::vector<MPI_Offset>& index, unsigned int* dataumValue) const;
    /*! \overload
     */ 
    void getVar(const std::vector<MPI_Offset>& index, unsigned long long* dataumValue) const;
    /*! Reads a single datum value from a variable of an open netCDF dataset.
      The value is converted from the external data type of the variable, if necessary.

      \param index       Vector specifying the index of the data value to be read. 
      The indices are relative to 0, so for example, the first data value of a two-dimensional 
      variable would have index (0,0). The elements of index must correspond to the variable's dimensions. 
      Hence, if the variable is a record variable, the first index is the record number.

      \param datumValue Pointer to the location into which the data value is read. If the type of 
      data value differs from the netCDF variable type, type conversion will occur.
      (However, no type conversion is carried out for variables using the user-defined data types:
      ncmpi_Vlen, ncmpi_Opaque, ncmpi_Compound and ncmpi_Enum.)
    */
    void getVar(const std::vector<MPI_Offset>& index, long long* dataumValue) const;

    void getVar(const std::vector<MPI_Offset>& index, void* dataumValue, MPI_Offset bufcount, MPI_Datatype buftype) const;

    //////////////////////

    // Reads an array of values from a netCDF variable of an open netCDF dataset. 
    /*!
      This is an overloaded member function, provided for convenience. 
      It differs from the above function in what argument(s) it accepts. 
      In addition, no data conversion is carried out. This means that 
      the type of the data in memory must match the type of the variable.
    */
    // void getVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count, void* dataValues) const;
    /*! \overload
     */ 
    void getVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count, char* dataValues) const;
    /*! \overload
     */ 
    void getVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count, unsigned char* dataValues) const;
    /*! \overload
     */ 
    void getVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count, signed char* dataValues) const;
    /*! \overload
     */ 
    void getVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count, short* dataValues) const;
    /*! \overload
     */ 
    void getVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count, int* dataValues) const;
    /*! \overload
     */ 
    void getVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count, long* dataValues) const;
    /*! \overload
     */ 
    void getVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count, float* dataValues) const;
    /*! \overload
     */ 
    void getVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count, double* dataValues) const;
    /*! \overload
     */ 
    void getVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count, unsigned short* dataValues) const;
    /*! \overload
     */ 
    void getVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count, unsigned int* dataValues) const;
    /*! \overload
     */ 
    void getVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count, unsigned long long* dataValues) const;
    /*! 
      Reads an array of values from a netCDF variable of an open netCDF dataset. 
      The array is specified by giving a corner and a vector of edge lengths. 
      The values are read into consecutive locations with the last dimension varying fastest.

      \param start
      Vector specifying the index in the variable where the first of the data values will be read. 
      The indices are relative to 0, so for example, the first data value of a variable would have index (0, 0, ... , 0). 
      The length of start must be the same as the number of dimensions of the specified variable. 
      The elements of start correspond, in order, to the variable's dimensions. Hence, if the variable is a record variable, 
      the first index would correspond to the starting record number for reading the data values.

      \param count
      Vector specifying the edge lengths along each dimension of the block of data values to be read. 
      To read a single value, for example, specify count as (1, 1, ... , 1). The length of count is the number of 
      dimensions of the specified variable. The elements of count correspond, in order, to the variable's dimensions. 
      Hence, if the variable is a record variable, the first element of count corresponds to a count of the number of records to read.
      Note: setting any element of the count array to zero causes the function to exit without error, and without doing anything. 

      \param dataValues Pointer to the location into which the data value is read. If the type of 
      data value differs from the netCDF variable type, type conversion will occur.
      (However, no type conversion is carried out for variables using the user-defined data types:
      ncmpi_Vlen, ncmpi_Opaque, ncmpi_Compound and ncmpi_Enum.)
    */
    void getVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count, long long* dataValues) const;

    void getVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count, void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const;

    //////////////////////

    // Reads a subsampled (strided) array section of values from a netCDF variable.
    /*!
      This is an overloaded member function, provided for convenience. 
      It differs from the above function in what argument(s) it accepts. 
      In addition, no data conversion is carried out. This means that 
      the type of the data in memory must match the type of the variable.
    */
    // void getVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, void* dataValues) const;
    /*! \overload
     */ 
    void getVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, char* dataValues) const;
    /*! \overload
     */ 
    void getVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, unsigned char* dataValues) const;
    /*! \overload
     */ 
    void getVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, signed char* dataValues) const;
    /*! \overload
     */ 
    void getVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, short* dataValues) const;
    /*! \overload
     */ 
    void getVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, int* dataValues) const;
    /*! \overload
     */ 
    void getVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, long* dataValues) const;
    /*! \overload
     */ 
    void getVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, float* dataValues) const;
    /*! \overload
     */ 
    void getVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, double* dataValues) const;
    /*! \overload
     */ 
    void getVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, unsigned short* dataValues) const;
    /*! \overload
     */ 
    void getVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, unsigned int* dataValues) const;
    /*! \overload
     */ 
    void getVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, unsigned long long* dataValues) const;
    /*! 
      Reads a subsampled (strided) array section of values from a netCDF variable.
      The subsampled array section is specified by giving a corner, a vector of edge lengths, and a stride vector. 
      The values are read with the last dimension of the netCDF variable varying fastest.

      \param start
      Vector specifying the index in the variable where the first of the data values will be read. 
      The indices are relative to 0, so for example, the first data value of a variable would have index (0, 0, ... , 0). 
      The length of start must be the same as the number of dimensions of the specified variable. 
      The elements of start correspond, in order, to the variable's dimensions. Hence, if the variable is a record variable, 
      the first index would correspond to the starting record number for reading the data values.

      \param count
      Vector specifying the edge lengths along each dimension of the block of data values to be read. 
      To read a single value, for example, specify count as (1, 1, ... , 1). The length of count is the number of 
      dimensions of the specified variable. The elements of count correspond, in order, to the variable's dimensions. 
      Hence, if the variable is a record variable, the first element of count corresponds to a count of the number of records to read.
      Note: setting any element of the count array to zero causes the function to exit without error, and without doing anything. 

      \param stride
      Vector specifying the interval between selected indices. The elements of the stride vector correspond, in order, 
      to the variable's dimensions. A value of 1 accesses adjacent values of the netCDF variable in the corresponding 
      dimension; a value of 2 accesses every other value of the netCDF variable in the corresponding dimension; and so 
      on. A NULL stride argument is treated as (1, 1, ... , 1). 

      \param dataValues Pointer to the location into which the data value is read. If the type of 
      data value differs from the netCDF variable type, type conversion will occur.
      (However, no type conversion is carried out for variables using the user-defined data types:
      ncmpi_Vlen, ncmpi_Opaque, ncmpi_Compound and ncmpi_Enum.)
    */
    void getVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, long long* dataValues) const;

    void getVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const;


    //////////////////////

    // Reads a mapped array section of values from a netCDF variable.
    /*!
      This is an overloaded member function, provided for convenience. 
      It differs from the above function in what argument(s) it accepts. 
      In addition, no data conversion is carried out. This means that 
      the type of the data in memory must match the type of the variable.
    */
    // void getVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, const std::vector<MPI_Offset>& imap, void* dataValues) const;
    /*! \overload
     */ 
    void getVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, const std::vector<MPI_Offset>& imap, char* dataValues) const;
    /*! \overload
     */ 
    void getVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, const std::vector<MPI_Offset>& imap, unsigned char* dataValues) const;
    /*! \overload
     */ 
    void getVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, const std::vector<MPI_Offset>& imap, signed char* dataValues) const;
    /*! \overload
     */ 
    void getVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, const std::vector<MPI_Offset>& imap, short* dataValues) const;
    /*! \overload
     */ 
    void getVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, const std::vector<MPI_Offset>& imap, int* dataValues) const;
    /*! \overload
     */ 
    void getVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, const std::vector<MPI_Offset>& imap, long* dataValues) const;
    /*! \overload
     */ 
    void getVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, const std::vector<MPI_Offset>& imap, float* dataValues) const;
    /*! \overload
     */ 
    void getVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, const std::vector<MPI_Offset>& imap, double* dataValues) const;
    /*! \overload
     */ 
    void getVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, const std::vector<MPI_Offset>& imap, unsigned short* dataValues) const;
    /*! \overload
     */ 
    void getVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, const std::vector<MPI_Offset>& imap, unsigned int* dataValues) const;
    /*! \overload
     */ 
    void getVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, const std::vector<MPI_Offset>& imap, unsigned long long* dataValues) const;
    /*! 
      Reads a mapped array section of values from a netCDF variable.
      The mapped array section is specified by giving a corner, a vector of edge lengths, a stride vector, and an 
      index mapping vector. The index mapping vector is a vector of integers that specifies the mapping between the 
      dimensions of a netCDF variable and the in-memory structure of the internal data array. No assumptions are 
      made about the ordering or length of the dimensions of the data array. 

      \param start
      Vector specifying the index in the variable where the first of the data values will be read. 
      The indices are relative to 0, so for example, the first data value of a variable would have index (0, 0, ... , 0). 
      The length of start must be the same as the number of dimensions of the specified variable. 
      The elements of start correspond, in order, to the variable's dimensions. Hence, if the variable is a record variable, 
      the first index would correspond to the starting record number for reading the data values.

      \param count
      Vector specifying the edge lengths along each dimension of the block of data values to be read. 
      To read a single value, for example, specify count as (1, 1, ... , 1). The length of count is the number of 
      dimensions of the specified variable. The elements of count correspond, in order, to the variable's dimensions. 
      Hence, if the variable is a record variable, the first element of count corresponds to a count of the number of records to read.
      Note: setting any element of the count array to zero causes the function to exit without error, and without doing anything. 

      \param stride
      Vector specifying the interval between selected indices. The elements of the stride vector correspond, in order, 
      to the variable's dimensions. A value of 1 accesses adjacent values of the netCDF variable in the corresponding 
      dimension; a value of 2 accesses every other value of the netCDF variable in the corresponding dimension; and so 
      on. A NULL stride argument is treated as (1, 1, ... , 1). 

      \param imap
      Vector of integers that specifies the mapping between the dimensions of a netCDF variable and the in-memory 
      structure of the internal data array. imap[0] gives the distance between elements of the internal array corresponding 
      to the most slowly varying dimension of the netCDF variable. imap[n-1] (where n is the rank of the netCDF variable) 
      gives the distance between elements of the internal array corresponding to the most rapidly varying dimension of the 
      netCDF variable. Intervening imap elements correspond to other dimensions of the netCDF variable in the obvious way. 
      Distances between elements are specified in type-independent units of elements (the distance between internal elements 
      that occupy adjacent memory locations is 1 and not the element's byte-length as in netCDF 2). 

      \param dataValues Pointer to the location into which the data value is read. If the type of 
      data value differs from the netCDF variable type, type conversion will occur.
      (However, no type conversion is carried out for variables using the user-defined data types:
      ncmpi_Vlen, ncmpi_Opaque, ncmpi_Compound and ncmpi_Enum.)
    */
    void getVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, const std::vector<MPI_Offset>& imap, long long* dataValues) const;

    void getVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, const std::vector<MPI_Offset>& imap, void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const;

    //////////////////////

    // Reads the entire data into the netCDF variable. (collective data mode)
    /*!
      This is an overloaded member function, provided for convenience. 
      It differs from the above function in what argument(s) it accepts. 
      In addition, no data conversion is carried out. This means that 
      the type of the data in memory must match the type of the variable.
    */
    // void getVar_all(void* dataValues) const;
    /*! \overload
     */ 
    void getVar_all(char* dataValues) const;
    /*! \overload
     */ 
    void getVar_all(unsigned char* dataValues) const;
    /*! \overload
     */ 
    void getVar_all(signed char* dataValues) const;
    /*! \overload
     */ 
    void getVar_all(short* dataValues) const;
    /*! \overload
     */ 
    void getVar_all(int* dataValues) const;
    /*! \overload
     */ 
    void getVar_all(long* dataValues) const;
    /*! \overload
     */ 
    void getVar_all(float* dataValues) const;
    /*! \overload
     */ 
    void getVar_all(double* dataValues) const;
    /*! \overload
     */ 
    void getVar_all(unsigned short* dataValues) const;
    /*! \overload
     */ 
    void getVar_all(unsigned int* dataValues) const;
    /*! \overload
     */ 
    void getVar_all(unsigned long long* dataValues) const;
    /*! 
      Reads the entire data from an netCDF variable.
      This is the simplest interface to use for reading the value of a scalar variable 
      or when all the values of a multidimensional variable can be read at once. The values 
      are read into consecutive locations with the last dimension varying fastest. 
      
      Take care when using the simplest forms of this interface with record variables when you 
      don't specify how many records are to be read. If you try to read all the values of a 
      record variable into an array but there are more records in the file than you assume, 
      more data will be read than you expect, which may cause a segmentation violation.

      \param dataValues Pointer to the location into which the data value is read. If the type of 
      data value differs from the netCDF variable type, type conversion will occur.
      (However, no type conversion is carried out for variables using the user-defined data types:
      ncmpi_Vlen, ncmpi_Opaque, ncmpi_Compound and ncmpi_Enum.)
    */
    void getVar_all(long long* dataValues) const;

    void getVar_all(void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const;

    ////////////////////
    
    // Reads a single datum value from a variable of an open netCDF dataset.
    /*!
      This is an overloaded member function, provided for convenience. 
      It differs from the above function in what argument(s) it accepts. 
      In addition, no data conversion is carried out. This means that 
      the type of the data in memory must match the type of the variable.
    */
    // void getVar_all(const std::vector<MPI_Offset>& index, void* dataumValue) const;
    /*! \overload
     */ 
    void getVar_all(const std::vector<MPI_Offset>& index, char* dataumValue) const;
    /*! \overload
     */ 
    void getVar_all(const std::vector<MPI_Offset>& index, unsigned char* dataumValue) const;
    /*! \overload
     */ 
    void getVar_all(const std::vector<MPI_Offset>& index, signed char* dataumValue) const;
    /*! \overload
     */ 
    void getVar_all(const std::vector<MPI_Offset>& index, short* dataumValue) const;
    /*! \overload
     */ 
    void getVar_all(const std::vector<MPI_Offset>& index, int* dataumValue) const;
    /*! \overload
     */ 
    void getVar_all(const std::vector<MPI_Offset>& index, long* dataumValue) const;
    /*! \overload
     */ 
    void getVar_all(const std::vector<MPI_Offset>& index, float* dataumValue) const;
    /*! \overload
     */ 
    void getVar_all(const std::vector<MPI_Offset>& index, double* dataumValue) const;
    /*! \overload
     */ 
    void getVar_all(const std::vector<MPI_Offset>& index, unsigned short* dataumValue) const;
    /*! \overload
     */ 
    void getVar_all(const std::vector<MPI_Offset>& index, unsigned int* dataumValue) const;
    /*! \overload
     */ 
    void getVar_all(const std::vector<MPI_Offset>& index, unsigned long long* dataumValue) const;
    /*! Reads a single datum value from a variable of an open netCDF dataset.
      The value is converted from the external data type of the variable, if necessary.

      \param index       Vector specifying the index of the data value to be read. 
      The indices are relative to 0, so for example, the first data value of a two-dimensional 
      variable would have index (0,0). The elements of index must correspond to the variable's dimensions. 
      Hence, if the variable is a record variable, the first index is the record number.

      \param datumValue Pointer to the location into which the data value is read. If the type of 
      data value differs from the netCDF variable type, type conversion will occur.
      (However, no type conversion is carried out for variables using the user-defined data types:
      ncmpi_Vlen, ncmpi_Opaque, ncmpi_Compound and ncmpi_Enum.)
    */
    void getVar_all(const std::vector<MPI_Offset>& index, long long* dataumValue) const;

    void getVar_all(const std::vector<MPI_Offset>& index, void* dataumValue, MPI_Offset bufcount, MPI_Datatype buftype) const;

    //////////////////////

    // Reads an array of values from a netCDF variable of an open netCDF dataset. 
    /*!
      This is an overloaded member function, provided for convenience. 
      It differs from the above function in what argument(s) it accepts. 
      In addition, no data conversion is carried out. This means that 
      the type of the data in memory must match the type of the variable.
    */
    // void getVar_all(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count, void* dataValues) const;
    /*! \overload
     */ 
    void getVar_all(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count, char* dataValues) const;
    /*! \overload
     */ 
    void getVar_all(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count, unsigned char* dataValues) const;
    /*! \overload
     */ 
    void getVar_all(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count, signed char* dataValues) const;
    /*! \overload
     */ 
    void getVar_all(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count, short* dataValues) const;
    /*! \overload
     */ 
    void getVar_all(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count, int* dataValues) const;
    /*! \overload
     */ 
    void getVar_all(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count, long* dataValues) const;
    /*! \overload
     */ 
    void getVar_all(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count, float* dataValues) const;
    /*! \overload
     */ 
    void getVar_all(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count, double* dataValues) const;
    /*! \overload
     */ 
    void getVar_all(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count, unsigned short* dataValues) const;
    /*! \overload
     */ 
    void getVar_all(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count, unsigned int* dataValues) const;
    /*! \overload
     */ 
    void getVar_all(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count, unsigned long long* dataValues) const;
    /*! 
      Reads an array of values from a netCDF variable of an open netCDF dataset. 
      The array is specified by giving a corner and a vector of edge lengths. 
      The values are read into consecutive locations with the last dimension varying fastest.

      \param start
      Vector specifying the index in the variable where the first of the data values will be read. 
      The indices are relative to 0, so for example, the first data value of a variable would have index (0, 0, ... , 0). 
      The length of start must be the same as the number of dimensions of the specified variable. 
      The elements of start correspond, in order, to the variable's dimensions. Hence, if the variable is a record variable, 
      the first index would correspond to the starting record number for reading the data values.

      \param count
      Vector specifying the edge lengths along each dimension of the block of data values to be read. 
      To read a single value, for example, specify count as (1, 1, ... , 1). The length of count is the number of 
      dimensions of the specified variable. The elements of count correspond, in order, to the variable's dimensions. 
      Hence, if the variable is a record variable, the first element of count corresponds to a count of the number of records to read.
      Note: setting any element of the count array to zero causes the function to exit without error, and without doing anything. 

      \param dataValues Pointer to the location into which the data value is read. If the type of 
      data value differs from the netCDF variable type, type conversion will occur.
      (However, no type conversion is carried out for variables using the user-defined data types:
      ncmpi_Vlen, ncmpi_Opaque, ncmpi_Compound and ncmpi_Enum.)
    */
    void getVar_all(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count, long long* dataValues) const;

    void getVar_all(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count, void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const;

    //////////////////////

    // Reads a subsampled (strided) array section of values from a netCDF variable.
    /*!
      This is an overloaded member function, provided for convenience. 
      It differs from the above function in what argument(s) it accepts. 
      In addition, no data conversion is carried out. This means that 
      the type of the data in memory must match the type of the variable.
    */
    // void getVar_all(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, void* dataValues) const;
    /*! \overload
     */ 
    void getVar_all(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, char* dataValues) const;
    /*! \overload
     */ 
    void getVar_all(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, unsigned char* dataValues) const;
    /*! \overload
     */ 
    void getVar_all(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, signed char* dataValues) const;
    /*! \overload
     */ 
    void getVar_all(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, short* dataValues) const;
    /*! \overload
     */ 
    void getVar_all(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, int* dataValues) const;
    /*! \overload
     */ 
    void getVar_all(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, long* dataValues) const;
    /*! \overload
     */ 
    void getVar_all(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, float* dataValues) const;
    /*! \overload
     */ 
    void getVar_all(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, double* dataValues) const;
    /*! \overload
     */ 
    void getVar_all(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, unsigned short* dataValues) const;
    /*! \overload
     */ 
    void getVar_all(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, unsigned int* dataValues) const;
    /*! \overload
     */ 
    void getVar_all(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, unsigned long long* dataValues) const;
    /*! 
      Reads a subsampled (strided) array section of values from a netCDF variable.
      The subsampled array section is specified by giving a corner, a vector of edge lengths, and a stride vector. 
      The values are read with the last dimension of the netCDF variable varying fastest.

      \param start
      Vector specifying the index in the variable where the first of the data values will be read. 
      The indices are relative to 0, so for example, the first data value of a variable would have index (0, 0, ... , 0). 
      The length of start must be the same as the number of dimensions of the specified variable. 
      The elements of start correspond, in order, to the variable's dimensions. Hence, if the variable is a record variable, 
      the first index would correspond to the starting record number for reading the data values.

      \param count
      Vector specifying the edge lengths along each dimension of the block of data values to be read. 
      To read a single value, for example, specify count as (1, 1, ... , 1). The length of count is the number of 
      dimensions of the specified variable. The elements of count correspond, in order, to the variable's dimensions. 
      Hence, if the variable is a record variable, the first element of count corresponds to a count of the number of records to read.
      Note: setting any element of the count array to zero causes the function to exit without error, and without doing anything. 

      \param stride
      Vector specifying the interval between selected indices. The elements of the stride vector correspond, in order, 
      to the variable's dimensions. A value of 1 accesses adjacent values of the netCDF variable in the corresponding 
      dimension; a value of 2 accesses every other value of the netCDF variable in the corresponding dimension; and so 
      on. A NULL stride argument is treated as (1, 1, ... , 1). 

      \param dataValues Pointer to the location into which the data value is read. If the type of 
      data value differs from the netCDF variable type, type conversion will occur.
      (However, no type conversion is carried out for variables using the user-defined data types:
      ncmpi_Vlen, ncmpi_Opaque, ncmpi_Compound and ncmpi_Enum.)
    */
    void getVar_all(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, long long* dataValues) const;

    void getVar_all(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const;


    //////////////////////

    // Reads a mapped array section of values from a netCDF variable.
    /*!
      This is an overloaded member function, provided for convenience. 
      It differs from the above function in what argument(s) it accepts. 
      In addition, no data conversion is carried out. This means that 
      the type of the data in memory must match the type of the variable.
    */
    // void getVar_all(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, const std::vector<MPI_Offset>& imap, void* dataValues) const;
    /*! \overload
     */ 
    void getVar_all(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, const std::vector<MPI_Offset>& imap, char* dataValues) const;
    /*! \overload
     */ 
    void getVar_all(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, const std::vector<MPI_Offset>& imap, unsigned char* dataValues) const;
    /*! \overload
     */ 
    void getVar_all(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, const std::vector<MPI_Offset>& imap, signed char* dataValues) const;
    /*! \overload
     */ 
    void getVar_all(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, const std::vector<MPI_Offset>& imap, short* dataValues) const;
    /*! \overload
     */ 
    void getVar_all(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, const std::vector<MPI_Offset>& imap, int* dataValues) const;
    /*! \overload
     */ 
    void getVar_all(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, const std::vector<MPI_Offset>& imap, long* dataValues) const;
    /*! \overload
     */ 
    void getVar_all(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, const std::vector<MPI_Offset>& imap, float* dataValues) const;
    /*! \overload
     */ 
    void getVar_all(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, const std::vector<MPI_Offset>& imap, double* dataValues) const;
    /*! \overload
     */ 
    void getVar_all(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, const std::vector<MPI_Offset>& imap, unsigned short* dataValues) const;
    /*! \overload
     */ 
    void getVar_all(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, const std::vector<MPI_Offset>& imap, unsigned int* dataValues) const;
    /*! \overload
     */ 
    void getVar_all(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, const std::vector<MPI_Offset>& imap, unsigned long long* dataValues) const;
    /*! 
      Reads a mapped array section of values from a netCDF variable.
      The mapped array section is specified by giving a corner, a vector of edge lengths, a stride vector, and an 
      index mapping vector. The index mapping vector is a vector of integers that specifies the mapping between the 
      dimensions of a netCDF variable and the in-memory structure of the internal data array. No assumptions are 
      made about the ordering or length of the dimensions of the data array. 

      \param start
      Vector specifying the index in the variable where the first of the data values will be read. 
      The indices are relative to 0, so for example, the first data value of a variable would have index (0, 0, ... , 0). 
      The length of start must be the same as the number of dimensions of the specified variable. 
      The elements of start correspond, in order, to the variable's dimensions. Hence, if the variable is a record variable, 
      the first index would correspond to the starting record number for reading the data values.

      \param count
      Vector specifying the edge lengths along each dimension of the block of data values to be read. 
      To read a single value, for example, specify count as (1, 1, ... , 1). The length of count is the number of 
      dimensions of the specified variable. The elements of count correspond, in order, to the variable's dimensions. 
      Hence, if the variable is a record variable, the first element of count corresponds to a count of the number of records to read.
      Note: setting any element of the count array to zero causes the function to exit without error, and without doing anything. 

      \param stride
      Vector specifying the interval between selected indices. The elements of the stride vector correspond, in order, 
      to the variable's dimensions. A value of 1 accesses adjacent values of the netCDF variable in the corresponding 
      dimension; a value of 2 accesses every other value of the netCDF variable in the corresponding dimension; and so 
      on. A NULL stride argument is treated as (1, 1, ... , 1). 

      \param imap
      Vector of integers that specifies the mapping between the dimensions of a netCDF variable and the in-memory 
      structure of the internal data array. imap[0] gives the distance between elements of the internal array corresponding 
      to the most slowly varying dimension of the netCDF variable. imap[n-1] (where n is the rank of the netCDF variable) 
      gives the distance between elements of the internal array corresponding to the most rapidly varying dimension of the 
      netCDF variable. Intervening imap elements correspond to other dimensions of the netCDF variable in the obvious way. 
      Distances between elements are specified in type-independent units of elements (the distance between internal elements 
      that occupy adjacent memory locations is 1 and not the element's byte-length as in netCDF 2). 

      \param dataValues Pointer to the location into which the data value is read. If the type of 
      data value differs from the netCDF variable type, type conversion will occur.
      (However, no type conversion is carried out for variables using the user-defined data types:
      ncmpi_Vlen, ncmpi_Opaque, ncmpi_Compound and ncmpi_Enum.)
    */
    void getVar_all(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, const std::vector<MPI_Offset>& imap, long long* dataValues) const;

    void getVar_all(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, const std::vector<MPI_Offset>& imap, void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const;


   
    ////////////////////

    //  Nonblocking data reading

    ////////////////////

    // Reads the entire data into the netCDF variable.
    /*!
      This is an overloaded member function, provided for convenience. 
      It differs from the above function in what argument(s) it accepts. 
      In addition, no data conversion is carried out. This means that 
      the type of the data in memory must match the type of the variable.
    */
    // void igetVar(void* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVar(char* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVar(unsigned char* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVar(signed char* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVar(short* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVar(int* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVar(long* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVar(float* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVar(double* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVar(unsigned short* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVar(unsigned int* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVar(unsigned long long* dataValues, int *req) const;
    /*! 
      Reads the entire data from an netCDF variable.
      This is the simplest interface to use for reading the value of a scalar variable 
      or when all the values of a multidimensional variable can be read at once. The values 
      are read into consecutive locations with the last dimension varying fastest. 
      
      Take care when using the simplest forms of this interface with record variables when you 
      don't specify how many records are to be read. If you try to read all the values of a 
      record variable into an array but there are more records in the file than you assume, 
      more data will be read than you expect, which may cause a segmentation violation.

      \param dataValues Pointer to the location into which the data value is read. If the type of 
      data value differs from the netCDF variable type, type conversion will occur.
      (However, no type conversion is carried out for variables using the user-defined data types:
      ncmpi_Vlen, ncmpi_Opaque, ncmpi_Compound and ncmpi_Enum.)
    */
    void igetVar(long long* dataValues, int *req) const;

    void igetVar(void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype, int *req) const;


    //////////////////////

    // Reads a single datum value from a variable of an open netCDF dataset.
    /*!
      This is an overloaded member function, provided for convenience. 
      It differs from the above function in what argument(s) it accepts. 
      In addition, no data conversion is carried out. This means that 
      the type of the data in memory must match the type of the variable.
    */
    // void igetVar(const std::vector<MPI_Offset>& index, void* dataumValue, int *req) const;
    /*! \overload
     */ 
    void igetVar(const std::vector<MPI_Offset>& index, char* dataumValue, int *req) const;
    /*! \overload
     */ 
    void igetVar(const std::vector<MPI_Offset>& index, unsigned char* dataumValue, int *req) const;
    /*! \overload
     */ 
    void igetVar(const std::vector<MPI_Offset>& index, signed char* dataumValue, int *req) const;
    /*! \overload
     */ 
    void igetVar(const std::vector<MPI_Offset>& index, short* dataumValue, int *req) const;
    /*! \overload
     */ 
    void igetVar(const std::vector<MPI_Offset>& index, int* dataumValue, int *req) const;
    /*! \overload
     */ 
    void igetVar(const std::vector<MPI_Offset>& index, long* dataumValue, int *req) const;
    /*! \overload
     */ 
    void igetVar(const std::vector<MPI_Offset>& index, float* dataumValue, int *req) const;
    /*! \overload
     */ 
    void igetVar(const std::vector<MPI_Offset>& index, double* dataumValue, int *req) const;
    /*! \overload
     */ 
    void igetVar(const std::vector<MPI_Offset>& index, unsigned short* dataumValue, int *req) const;
    /*! \overload
     */ 
    void igetVar(const std::vector<MPI_Offset>& index, unsigned int* dataumValue, int *req) const;
    /*! \overload
     */ 
    void igetVar(const std::vector<MPI_Offset>& index, unsigned long long* dataumValue, int *req) const;
    /*! Reads a single datum value from a variable of an open netCDF dataset.
      The value is converted from the external data type of the variable, if necessary.

      \param index       Vector specifying the index of the data value to be read. 
      The indices are relative to 0, so for example, the first data value of a two-dimensional 
      variable would have index (0,0). The elements of index must correspond to the variable's dimensions. 
      Hence, if the variable is a record variable, the first index is the record number.

      \param datumValue Pointer to the location into which the data value is read. If the type of 
      data value differs from the netCDF variable type, type conversion will occur.
      (However, no type conversion is carried out for variables using the user-defined data types:
      ncmpi_Vlen, ncmpi_Opaque, ncmpi_Compound and ncmpi_Enum.)
    */
    void igetVar(const std::vector<MPI_Offset>& index, long long* dataumValue, int *req) const;

    void igetVar(const std::vector<MPI_Offset>& index, void* dataumValue, MPI_Offset bufcount, MPI_Datatype buftype, int *req) const;

    //////////////////////

    // Reads an array of values from a netCDF variable of an open netCDF dataset. 
    /*!
      This is an overloaded member function, provided for convenience. 
      It differs from the above function in what argument(s) it accepts. 
      In addition, no data conversion is carried out. This means that 
      the type of the data in memory must match the type of the variable.
    */
    // void igetVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count, void* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count, char* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count, unsigned char* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count, signed char* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count, short* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count, int* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count, long* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count, float* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count, double* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count, unsigned short* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count, unsigned int* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count, unsigned long long* dataValues, int *req) const;
    /*! 
      Reads an array of values from a netCDF variable of an open netCDF dataset. 
      The array is specified by giving a corner and a vector of edge lengths. 
      The values are read into consecutive locations with the last dimension varying fastest.

      \param start
      Vector specifying the index in the variable where the first of the data values will be read. 
      The indices are relative to 0, so for example, the first data value of a variable would have index (0, 0, ... , 0). 
      The length of start must be the same as the number of dimensions of the specified variable. 
      The elements of start correspond, in order, to the variable's dimensions. Hence, if the variable is a record variable, 
      the first index would correspond to the starting record number for reading the data values.

      \param count
      Vector specifying the edge lengths along each dimension of the block of data values to be read. 
      To read a single value, for example, specify count as (1, 1, ... , 1). The length of count is the number of 
      dimensions of the specified variable. The elements of count correspond, in order, to the variable's dimensions. 
      Hence, if the variable is a record variable, the first element of count corresponds to a count of the number of records to read.
      Note: setting any element of the count array to zero causes the function to exit without error, and without doing anything. 

      \param dataValues Pointer to the location into which the data value is read. If the type of 
      data value differs from the netCDF variable type, type conversion will occur.
      (However, no type conversion is carried out for variables using the user-defined data types:
      ncmpi_Vlen, ncmpi_Opaque, ncmpi_Compound and ncmpi_Enum.)
    */
    void igetVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count, long long* dataValues, int *req) const;

    void igetVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count, void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype, int *req) const;

    //////////////////////

    // Reads a subsampled (strided) array section of values from a netCDF variable.
    /*!
      This is an overloaded member function, provided for convenience. 
      It differs from the above function in what argument(s) it accepts. 
      In addition, no data conversion is carried out. This means that 
      the type of the data in memory must match the type of the variable.
    */
    // void igetVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, void* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, char* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, unsigned char* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, signed char* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, short* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, int* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, long* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, float* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, double* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, unsigned short* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, unsigned int* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, unsigned long long* dataValues, int *req) const;
    /*! 
      Reads a subsampled (strided) array section of values from a netCDF variable.
      The subsampled array section is specified by giving a corner, a vector of edge lengths, and a stride vector. 
      The values are read with the last dimension of the netCDF variable varying fastest.

      \param start
      Vector specifying the index in the variable where the first of the data values will be read. 
      The indices are relative to 0, so for example, the first data value of a variable would have index (0, 0, ... , 0). 
      The length of start must be the same as the number of dimensions of the specified variable. 
      The elements of start correspond, in order, to the variable's dimensions. Hence, if the variable is a record variable, 
      the first index would correspond to the starting record number for reading the data values.

      \param count
      Vector specifying the edge lengths along each dimension of the block of data values to be read. 
      To read a single value, for example, specify count as (1, 1, ... , 1). The length of count is the number of 
      dimensions of the specified variable. The elements of count correspond, in order, to the variable's dimensions. 
      Hence, if the variable is a record variable, the first element of count corresponds to a count of the number of records to read.
      Note: setting any element of the count array to zero causes the function to exit without error, and without doing anything. 

      \param stride
      Vector specifying the interval between selected indices. The elements of the stride vector correspond, in order, 
      to the variable's dimensions. A value of 1 accesses adjacent values of the netCDF variable in the corresponding 
      dimension; a value of 2 accesses every other value of the netCDF variable in the corresponding dimension; and so 
      on. A NULL stride argument is treated as (1, 1, ... , 1). 

      \param dataValues Pointer to the location into which the data value is read. If the type of 
      data value differs from the netCDF variable type, type conversion will occur.
      (However, no type conversion is carried out for variables using the user-defined data types:
      ncmpi_Vlen, ncmpi_Opaque, ncmpi_Compound and ncmpi_Enum.)
    */
    void igetVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, long long* dataValues, int *req) const;

    void igetVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype, int *req) const;


    //////////////////////

    // Reads a mapped array section of values from a netCDF variable.
    /*!
      This is an overloaded member function, provided for convenience. 
      It differs from the above function in what argument(s) it accepts. 
      In addition, no data conversion is carried out. This means that 
      the type of the data in memory must match the type of the variable.
    */
    // void igetVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, const std::vector<MPI_Offset>& imap, void* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, const std::vector<MPI_Offset>& imap, char* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, const std::vector<MPI_Offset>& imap, unsigned char* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, const std::vector<MPI_Offset>& imap, signed char* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, const std::vector<MPI_Offset>& imap, short* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, const std::vector<MPI_Offset>& imap, int* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, const std::vector<MPI_Offset>& imap, long* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, const std::vector<MPI_Offset>& imap, float* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, const std::vector<MPI_Offset>& imap, double* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, const std::vector<MPI_Offset>& imap, unsigned short* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, const std::vector<MPI_Offset>& imap, unsigned int* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, const std::vector<MPI_Offset>& imap, unsigned long long* dataValues, int *req) const;
    /*! 
      Reads a mapped array section of values from a netCDF variable.
      The mapped array section is specified by giving a corner, a vector of edge lengths, a stride vector, and an 
      index mapping vector. The index mapping vector is a vector of integers that specifies the mapping between the 
      dimensions of a netCDF variable and the in-memory structure of the internal data array. No assumptions are 
      made about the ordering or length of the dimensions of the data array. 

      \param start
      Vector specifying the index in the variable where the first of the data values will be read. 
      The indices are relative to 0, so for example, the first data value of a variable would have index (0, 0, ... , 0). 
      The length of start must be the same as the number of dimensions of the specified variable. 
      The elements of start correspond, in order, to the variable's dimensions. Hence, if the variable is a record variable, 
      the first index would correspond to the starting record number for reading the data values.

      \param count
      Vector specifying the edge lengths along each dimension of the block of data values to be read. 
      To read a single value, for example, specify count as (1, 1, ... , 1). The length of count is the number of 
      dimensions of the specified variable. The elements of count correspond, in order, to the variable's dimensions. 
      Hence, if the variable is a record variable, the first element of count corresponds to a count of the number of records to read.
      Note: setting any element of the count array to zero causes the function to exit without error, and without doing anything. 

      \param stride
      Vector specifying the interval between selected indices. The elements of the stride vector correspond, in order, 
      to the variable's dimensions. A value of 1 accesses adjacent values of the netCDF variable in the corresponding 
      dimension; a value of 2 accesses every other value of the netCDF variable in the corresponding dimension; and so 
      on. A NULL stride argument is treated as (1, 1, ... , 1). 

      \param imap
      Vector of integers that specifies the mapping between the dimensions of a netCDF variable and the in-memory 
      structure of the internal data array. imap[0] gives the distance between elements of the internal array corresponding 
      to the most slowly varying dimension of the netCDF variable. imap[n-1] (where n is the rank of the netCDF variable) 
      gives the distance between elements of the internal array corresponding to the most rapidly varying dimension of the 
      netCDF variable. Intervening imap elements correspond to other dimensions of the netCDF variable in the obvious way. 
      Distances between elements are specified in type-independent units of elements (the distance between internal elements 
      that occupy adjacent memory locations is 1 and not the element's byte-length as in netCDF 2). 

      \param dataValues Pointer to the location into which the data value is read. If the type of 
      data value differs from the netCDF variable type, type conversion will occur.
      (However, no type conversion is carried out for variables using the user-defined data types:
      ncmpi_Vlen, ncmpi_Opaque, ncmpi_Compound and ncmpi_Enum.)
    */
    void igetVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, const std::vector<MPI_Offset>& imap, long long* dataValues, int *req) const;

    void igetVar(const std::vector<MPI_Offset>& start, const std::vector<MPI_Offset>& count,  const std::vector<MPI_Offset>& stride, const std::vector<MPI_Offset>& imap, void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype, int *req) const;

    //////////////////////

    // Reads a list of subarrays of values from a netCDF variable of an open netCDF dataset. (independent I/O APIs)
    /*!
      This is an overloaded member function, provided for convenience. 
      It differs from the above function in what argument(s) it accepts. 
      In addition, no data conversion is carried out. This means that 
      the type of the data in memory must match the type of the variable.
    */
    // void getVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], void* dataValues) const;
    /*! \overload
     */ 
    void getVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], char* dataValues) const;
    /*! \overload
     */ 
    void getVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], unsigned char* dataValues) const;
    /*! \overload
     */ 
    void getVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], signed char* dataValues) const;
    /*! \overload
     */ 
    void getVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], short* dataValues) const;
    /*! \overload
     */ 
    void getVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], int* dataValues) const;
    /*! \overload
     */ 
    void getVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], long* dataValues) const;
    /*! \overload
     */ 
    void getVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], float* dataValues) const;
    /*! \overload
     */ 
    void getVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], double* dataValues) const;
    /*! \overload
     */ 
    void getVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], unsigned short* dataValues) const;
    /*! \overload
     */ 
    void getVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], unsigned int* dataValues) const;
    /*! \overload
     */ 
    void getVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], unsigned long long* dataValues) const;
    /*! 
      Reads a list of subarrays from a netCDF variable of an open netCDF dataset. 
      Each subarray i is specified by giving a corner (starts[i][*]) and a vector of edge lengths (counts[i][*]). 
      The values are read into consecutive locations with the last dimension varying fastest.

      \param num
      Number of subarrays.

      \param starts
      2D array of size [num][ndims] where num is the number of subarrays to get and ndims if the number of dimensions of the specified variable.
      Each subarray i is specified by starts[i][*], the first of the data values will be read, and counts[i][*], edge lengths of the subarray.
      The indices are relative to 0, so for example, the first data value of a variable would have index (0, 0, ... , 0). 
      The length of the second dimension of starts and counts must be the same as the number of dimensions of the specified variable. 
      The elements of starts's second dimension correspond, in order, to the variable's dimensions. Hence, if the variable is a record variable, 
      the first element starts[*][0] would correspond to the starting record number for reading the data values.

      \param counts
      2D array of size [num][ndims] where num is the number of subarrays to get and ndims if the number of dimensions of the specified variable.
      To read a single value, for example, specify count as (1, 1, ... , 1). The length of each count[i] is the number of 
      dimensions of the specified variable. The elements of each count[i] correspond, in order, to the variable's dimensions. 
      Hence, if the variable is a record variable, the first element of count[i] corresponds to a count of the number of records to read.
      Note: setting any element of the count array to zero causes the function to exit without error, and without doing anything. 

      \param dataValues Pointer to the location into which the data value is read. If the type of 
      data value differs from the netCDF variable type, type conversion will occur.
      (However, no type conversion is carried out for variables using the user-defined data types:
      ncmpi_Vlen, ncmpi_Opaque, ncmpi_Compound and ncmpi_Enum.)
    */
    void getVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], long long* dataValues) const;

    void getVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const;

    // Reads a list of subarrays of values from a netCDF variable of an open netCDF dataset. (collective I/O APIs)
    /*!
      This is an overloaded member function, provided for convenience. 
      It differs from the above function in what argument(s) it accepts. 
      In addition, no data conversion is carried out. This means that 
      the type of the data in memory must match the type of the variable.
    */
    // void getVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], void* dataValues) const;
    /*! \overload
     */ 
    void getVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], char* dataValues) const;
    /*! \overload
     */ 
    void getVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], unsigned char* dataValues) const;
    /*! \overload
     */ 
    void getVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], signed char* dataValues) const;
    /*! \overload
     */ 
    void getVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], short* dataValues) const;
    /*! \overload
     */ 
    void getVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], int* dataValues) const;
    /*! \overload
     */ 
    void getVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], long* dataValues) const;
    /*! \overload
     */ 
    void getVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], float* dataValues) const;
    /*! \overload
     */ 
    void getVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], double* dataValues) const;
    /*! \overload
     */ 
    void getVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], unsigned short* dataValues) const;
    /*! \overload
     */ 
    void getVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], unsigned int* dataValues) const;
    /*! \overload
     */ 
    void getVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], unsigned long long* dataValues) const;
    /*! 
      Reads a list of subarrays from a netCDF variable of an open netCDF dataset. 
      Each subarray i is specified by giving a corner (starts[i][*]) and a vector of edge lengths (counts[i][*]). 
      The values are read into consecutive locations with the last dimension varying fastest.

      \param num
      Number of subarrays.

      \param starts
      2D array of size [num][ndims] where num is the number of subarrays to get and ndims if the number of dimensions of the specified variable.
      Each subarray i is specified by starts[i][*], the first of the data values will be read, and counts[i][*], edge lengths of the subarray.
      The indices are relative to 0, so for example, the first data value of a variable would have index (0, 0, ... , 0). 
      The length of the second dimension of starts and counts must be the same as the number of dimensions of the specified variable. 
      The elements of starts's second dimension correspond, in order, to the variable's dimensions. Hence, if the variable is a record variable, 
      the first element starts[*][0] would correspond to the starting record number for reading the data values.

      \param counts
      2D array of size [num][ndims] where num is the number of subarrays to get and ndims if the number of dimensions of the specified variable.
      To read a single value, for example, specify count as (1, 1, ... , 1). The length of each count[i] is the number of 
      dimensions of the specified variable. The elements of each count[i] correspond, in order, to the variable's dimensions. 
      Hence, if the variable is a record variable, the first element of count[i] corresponds to a count of the number of records to read.
      Note: setting any element of the count array to zero causes the function to exit without error, and without doing anything. 

      \param dataValues Pointer to the location into which the data value is read. If the type of 
      data value differs from the netCDF variable type, type conversion will occur.
      (However, no type conversion is carried out for variables using the user-defined data types:
      ncmpi_Vlen, ncmpi_Opaque, ncmpi_Compound and ncmpi_Enum.)
    */
    void getVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], long long* dataValues) const;

    void getVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const;

    /* vard APIs take filetype and buftype */
    void getVard    (MPI_Datatype filetype, void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const;
    void getVard_all(MPI_Datatype filetype, void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const;

    //////////////////////

    // Nonblocking reads a list of subarrays of values from a netCDF variable of an open netCDF dataset.
    /*!
      This is an overloaded member function, provided for convenience. 
      It differs from the above function in what argument(s) it accepts. 
      In addition, no data conversion is carried out. This means that 
      the type of the data in memory must match the type of the variable.
    */
    // void igetVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], void* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], char* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], unsigned char* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], signed char* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], short* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], int* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], long* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], float* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], double* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], unsigned short* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], unsigned int* dataValues, int *req) const;
    /*! \overload
     */ 
    void igetVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], unsigned long long* dataValues, int *req) const;
    /*! 
      Reads a list of subarrays from a netCDF variable of an open netCDF dataset. 
      Each subarray i is specified by giving a corner (starts[i][*]) and a vector of edge lengths (counts[i][*]). 
      The values are read into consecutive locations with the last dimension varying fastest.

      \param num
      Number of subarrays.

      \param starts
      2D array of size [num][ndims] where num is the number of subarrays to get and ndims if the number of dimensions of the specified variable.
      Each subarray i is specified by starts[i][*], the first of the data values will be read, and counts[i][*], edge lengths of the subarray.
      The indices are relative to 0, so for example, the first data value of a variable would have index (0, 0, ... , 0). 
      The length of the second dimension of starts and counts must be the same as the number of dimensions of the specified variable. 
      The elements of starts's second dimension correspond, in order, to the variable's dimensions. Hence, if the variable is a record variable, 
      the first element starts[*][0] would correspond to the starting record number for reading the data values.

      \param counts
      2D array of size [num][ndims] where num is the number of subarrays to get and ndims if the number of dimensions of the specified variable.
      To read a single value, for example, specify count as (1, 1, ... , 1). The length of each count[i] is the number of 
      dimensions of the specified variable. The elements of each count[i] correspond, in order, to the variable's dimensions. 
      Hence, if the variable is a record variable, the first element of count[i] corresponds to a count of the number of records to read.
      Note: setting any element of the count array to zero causes the function to exit without error, and without doing anything. 

      \param dataValues Pointer to the location into which the data value is read. If the type of 
      data value differs from the netCDF variable type, type conversion will occur.
      (However, no type conversion is carried out for variables using the user-defined data types:
      ncmpi_Vlen, ncmpi_Opaque, ncmpi_Compound and ncmpi_Enum.)
    */
    void igetVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], long long* dataValues, int *req) const;

    void igetVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype, int *req) const;

    //////////////////////


    //  data writing (independent data mode)

    ////////////////////
   
    // Writes the entire data into the netCDF variable.
    /*!
      This is an overloaded member function, provided for convenience. 
      It differs from the above function in what argument(s) it accepts. 
      In addition, no data conversion is carried out. This means that 
      the type of the data in memory must match the type of the variable.
    */
    // void putVar(const void* dataValues) const;
    /*! \overload
     */ 
    void putVar(const char* dataValues) const;
    /*!  \overload
    */
    void putVar(const unsigned char* dataValues) const;
    /*!  \overload
    */
    void putVar(const signed char* dataValues) const;
    /*!  \overload
    */
    void putVar(const short* dataValues) const;
    /*!  \overload
    */
    void putVar(const int* dataValues) const;
    /*!  \overload
    */
    void putVar(const long* dataValues) const;
    /*!  \overload
    */
    void putVar(const float* dataValues) const;
    /*!  \overload
    */
    void putVar(const double* dataValues) const;
    /*!  \overload
    */
    void putVar(const unsigned short* dataValues) const;
    /*!  \overload
    */
    void putVar(const unsigned int* dataValues) const;
    /*!  \overload
    */
    void putVar(const unsigned long long* dataValues) const;
    /*! 
      Writes the entire data into the netCDF variable.
      This is the simplest interface to use for writing a value in a scalar variable 
      or whenever all the values of a multidimensional variable can all be 
      written at once. The values to be written are associated with the 
      netCDF variable by assuming that the last dimension of the netCDF 
      variable varies fastest in the C interface. 
      
      Take care when using the simplest forms of this interface with 
      record variables when you don't specify how many records are to be 
      written. If you try to write all the values of a record variable 
      into a netCDF file that has no record data yet (hence has 0 records), 
      nothing will be written. Similarly, if you try to write all of a record 
      variable but there are more records in the file than you assume, more data 
      may be written to the file than you supply, which may result in a 
      segmentation violation.
      
      \param dataValues The data values. The order in which the data will be written to the netCDF variable is with the last 
      dimension of the specified variable varying fastest. If the type of data values differs from the netCDF variable type, type conversion will occur. 
      (However, no type conversion is carried out for variables using the user-defined data types:
      ncmpi_Vlen, ncmpi_Opaque, ncmpi_Compound and ncmpi_Enum.)
    */
    void putVar(const long long* dataValues) const;

    void putVar(const void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const;

    /////////////////////////

    // Writes a single datum into the netCDF variable.
    /*!
      This is an overloaded member function, provided for convenience. 
      It differs from the above function in what argument(s) it accepts. 
      In addition, no data conversion is carried out. This means that 
      the type of the data in memory must match the type of the variable.
    */
    // void putVar(const std::vector<MPI_Offset>& index, const void* dataumValue) const;
    /*! \overload
     */ 
    void putVar(const std::vector<MPI_Offset>& index, const std::string& dataumValue) const;
    /*!  \overload
    */
    void putVar(const std::vector<MPI_Offset>& index, const unsigned char* dataumValue) const;
    /*!  \overload
    */
    void putVar(const std::vector<MPI_Offset>& index, const signed char* dataumValue) const;
    /*!  \overload
    */
    void putVar(const std::vector<MPI_Offset>& index, const short dataumValue) const;
    /*!  \overload
    */
    void putVar(const std::vector<MPI_Offset>& index, const int dataumValue) const;
    /*!  \overload
    */
    void putVar(const std::vector<MPI_Offset>& index, const long dataumValue) const;
    /*!  \overload
    */
    void putVar(const std::vector<MPI_Offset>& index, const float dataumValue) const;
    /*!  \overload
    */
    void putVar(const std::vector<MPI_Offset>& index, const double dataumValue) const;
    /*!  \overload
    */
    void putVar(const std::vector<MPI_Offset>& index, const unsigned short dataumValue) const;
    /*!  \overload
    */
    void putVar(const std::vector<MPI_Offset>& index, const unsigned int dataumValue) const;
    /*!  \overload
    */
    void putVar(const std::vector<MPI_Offset>& index, const unsigned long long dataumValue) const;
    /*! 
      Writes a single datum into the netCDF variable.

      \param index      Vector specifying the index where the data values will be written. The indices are relative to 0, so for example, 
      the first data value of a two-dimensional variable would have index (0,0). The elements of index must correspond to the variable's dimensions. 
      Hence, if the variable uses the unlimited dimension, the first index would correspond to the unlimited dimension.

      \param datumValue The data value. If the type of data values differs from the netCDF variable type, type conversion will occur.
      (However, no type conversion is carried out for variables using the user-defined data types:
      ncmpi_Vlen, ncmpi_Opaque, ncmpi_Compound and ncmpi_Enum.)
    */
    void putVar(const std::vector<MPI_Offset>& index, const long long dataumValue) const;

    void putVar(const std::vector<MPI_Offset>& index, const void* dataumValue, MPI_Offset bufcount, MPI_Datatype buftype) const;

    /////////////////////////

    // Writes an array of values into the netCDF variable.
    /*!
      This is an overloaded member function, provided for convenience. 
      It differs from the above function in what argument(s) it accepts. 
      In addition, no data conversion is carried out. This means that 
      the type of the data in memory must match the type of the variable.
    */
    // void putVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const void* dataValues) const;
    /*! \overload
     */ 
    void putVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const char* dataValues) const;
    /*!  \overload
    */
    void putVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const unsigned char* dataValues) const;
    /*!  \overload
    */
    void putVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const signed char* dataValues) const;
    /*!  \overload
    */
    void putVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const short* dataValues) const;
    /*!  \overload
    */
    void putVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const int* dataValues) const;
    /*!  \overload
    */
    void putVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const long* dataValues) const;
    /*!  \overload
    */
    void putVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const float* dataValues) const;
    /*!  \overload
    */
    void putVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const double* dataValues) const;
    /*!  \overload
    */
    void putVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const unsigned short* dataValues) const;
    /*!  \overload
    */
    void putVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const unsigned int* dataValues) const;
    /*!  \overload
    */
    void putVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const unsigned long long* dataValues) const;
    /*! 
      Writes an array of values into the netCDF variable.
      The portion of the netCDF variable to write is specified by giving a corner and a vector of edge lengths 
      that refer to an array section of the netCDF variable. The values to be written are associated with 
      the netCDF variable by assuming that the last dimension of the netCDF variable varies fastest. 

      \param startp  Vector specifying the index where the first data values will be written.  The indices are relative to 0, so for 
      example, the first data value of a variable would have index (0, 0, ... , 0). The elements of start correspond, in order, to the 
      variable's dimensions. Hence, if the variable is a record variable, the first index corresponds to the starting record number for writing the data values.

      \param countp  Vector specifying the number of indices selected along each dimension. 
      To write a single value, for example, specify count as (1, 1, ... , 1). The elements of 
      count correspond, in order, to the variable's dimensions. Hence, if the variable is a record 
      variable, the first element of count corresponds to a count of the number of records to write. Note: setting any element 
      of the count array to zero causes the function to exit without error, and without doing anything. 

      \param dataValues The data values. The order in which the data will be written to the netCDF variable is with the last 
      dimension of the specified variable varying fastest. If the type of data values differs from the netCDF variable 
      type, type conversion will occur. (However, no type conversion is 
      carried out for variables using the user-defined data types:
      ncmpi_Vlen, ncmpi_Opaque, ncmpi_Compound and ncmpi_Enum.)
    */
    void putVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const long long* dataValues) const;
    void putVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const;

    ////////////////

    // Writes a set of subsampled array values into the netCDF variable.
    /*!
      This is an overloaded member function, provided for convenience. 
      It differs from the above function in what argument(s) it accepts. 
      In addition, no data conversion is carried out. This means that 
      the type of the data in memory must match the type of the variable.
    */
    // void putVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const void* dataValues) const;
    /*! \overload
     */ 
    void putVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const char* dataValues) const;
    /*!  \overload
    */
    void putVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const unsigned char* dataValues) const;
    /*!  \overload
    */
    void putVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const signed char* dataValues) const;
    /*!  \overload
    */
    void putVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const short* dataValues) const;
    /*!  \overload
    */
    void putVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const int* dataValues) const;
    /*!  \overload
    */
    void putVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const long* dataValues) const;
    /*!  \overload
    */
    void putVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const float* dataValues) const;
    /*!  \overload
    */
    void putVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const double* dataValues) const;
    /*!  \overload
    */
    void putVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const unsigned short* dataValues) const;
    /*!  \overload
    */
    void putVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const unsigned int* dataValues) const;
    /*!  \overload
    */
    void putVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const unsigned long long* dataValues) const;
    /*! 
      Writes an array of values into the netCDF variable.
      The subsampled array section is specified by giving a corner, a vector of counts, and a stride vector. 

      \param startp  Vector specifying the index where the first data values will be written.  The indices are relative to 0, so for 
      example, the first data value of a variable would have index (0, 0, ... , 0). The elements of start correspond, in order, to the 
      variable's dimensions. Hence, if the variable is a record variable, the first index corresponds to the starting record number for writing the data values.

      \param countp  Vector specifying the number of indices selected along each dimension. 
      To write a single value, for example, specify count as (1, 1, ... , 1). The elements of 
      count correspond, in order, to the variable's dimensions. Hence, if the variable is a record 
      variable, the first element of count corresponds to a count of the number of records to write. Note: setting any element 
      of the count array to zero causes the function to exit without error, and without doing anything. 

      \param stridep  A vector of MPI_Offset integers that specifies the sampling interval along each dimension of the netCDF variable. 
      The elements of the stride vector correspond, in order, to the netCDF variable's dimensions (stride[0] gives the sampling interval 
      along the most slowly varying dimension of the netCDF variable). Sampling intervals are specified in type-independent units of 
      elements (a value of 1 selects consecutive elements of the netCDF variable along the corresponding dimension, a value of 2 selects 
      every other element, etc.). A NULL stride argument is treated as (1, 1, ... , 1).

      \param dataValues The data values. The order in which the data will be written to the netCDF variable is with the last 
      dimension of the specified variable varying fastest. If the type of data values differs from the netCDF variable type, type conversion will occur. 
      (However, no type conversion is  carried out for variables using the user-defined data types: ncmpi_Vlen, ncmpi_Opaque, ncmpi_Compound and ncmpi_Enum.
    */
    void putVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const long long* dataValues) const;

    void putVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const;

    ////////////////

    // Writes a mapped array section of values into the netCDF variable.
    /*!
      This is an overloaded member function, provided for convenience. 
      It differs from the above function in what argument(s) it accepts. 
      In addition, no data conversion is carried out. This means that 
      the type of the data in memory must match the type of the variable.
    */
    // void putVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const void* dataValues) const;
    /*! \overload
     */ 
    void putVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const char* dataValues) const;
    /*!  \overload
    */
    void putVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const unsigned char* dataValues) const;
    /*!  \overload
    */
    void putVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const signed char* dataValues) const;
    /*!  \overload
    */
    void putVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const short* dataValues) const;
    /*!  \overload
    */
    void putVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const int* dataValues) const;
    /*!  \overload
    */
    void putVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const long* dataValues) const;
    /*!  \overload
    */
    void putVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const float* dataValues) const;
    /*!  \overload
    */
    void putVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const double* dataValues) const;
    /*!  \overload
    */
    void putVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const unsigned short* dataValues) const;
    /*!  \overload
    */
    void putVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const unsigned int* dataValues) const;
    /*!  \overload
    */
    void putVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const unsigned long long* dataValues) const;
    /*! 
      Writes a mapped array section of values into the netCDF variable.
      The mapped array section is specified by giving a corner, a vector of counts, a stride vector, and an index mapping vector. 
      The index mapping vector is a vector of integers that specifies the mapping between the dimensions of a netCDF variable and the in-memory structure of the internal data array. 
      No assumptions are made about the ordering or length of the dimensions of the data array. 

      \param countp  Vector specifying the number of indices selected along each dimension. 
      To write a single value, for example, specify count as (1, 1, ... , 1). The elements of 
      count correspond, in order, to the variable's dimensions. Hence, if the variable is a record 
      variable, the first element of count corresponds to a count of the number of records to write. Note: setting any element 
      of the count array to zero causes the function to exit without error, and without doing anything. 

      \param stridep  A vector of MPI_Offset integers that specifies the sampling interval along each dimension of the netCDF variable. 
      The elements of the stride vector correspond, in order, to the netCDF variable's dimensions (stride[0] gives the sampling interval 
      along the most slowly varying dimension of the netCDF variable). Sampling intervals are specified in type-independent units of 
      elements (a value of 1 selects consecutive elements of the netCDF variable along the corresponding dimension, a value of 2 selects 
      every other element, etc.). A NULL stride argument is treated as (1, 1, ... , 1).

      \param imap Vector  specifies the mapping between the dimensions of a netCDF variable and the in-memory structure of the internal data array. 
      The elements of the index mapping vector correspond, in order, to the netCDF variable's dimensions (imap[0] gives the distance between elements 
      of the internal array corresponding to the most slowly varying dimension of the netCDF variable). Distances between elements are 
      specified in type-independent units of elements (the distance between internal elements that occupy adjacent memory locations is 
      1 and not the element's byte-length as in netCDF 2). A NULL argument means the memory-resident values have the same structure as 
      the associated netCDF variable.

      \param dataValues The data values. The order in which the data will be written to the netCDF variable is with the last 
      dimension of the specified variable varying fastest. If the type of data values differs from the netCDF variable type, type conversion will occur. 
     (However, no type conversion is carried out for variables using the user-defined data types:  ncmpi_Vlen, ncmpi_Opaque, ncmpi_Compound and ncmpi_Enum.)
*/
    void putVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const long long* dataValues) const;

    void putVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const;

    ////////////////////

    //  data writing (collective data mode)

    ////////////////////

    // Writes the entire data into the netCDF variable.
    /*!
      This is an overloaded member function, provided for convenience. 
      It differs from the above function in what argument(s) it accepts. 
      In addition, no data conversion is carried out. This means that 
      the type of the data in memory must match the type of the variable.
    */
    // void putVar_all(const void* dataValues) const;
    /*! \overload
     */ 
    void putVar_all(const char* dataValues) const;
    /*!  \overload
    */
    void putVar_all(const unsigned char* dataValues) const;
    /*!  \overload
    */
    void putVar_all(const signed char* dataValues) const;
    /*!  \overload
    */
    void putVar_all(const short* dataValues) const;
    /*!  \overload
    */
    void putVar_all(const int* dataValues) const;
    /*!  \overload
    */
    void putVar_all(const long* dataValues) const;
    /*!  \overload
    */
    void putVar_all(const float* dataValues) const;
    /*!  \overload
    */
    void putVar_all(const double* dataValues) const;
    /*!  \overload
    */
    void putVar_all(const unsigned short* dataValues) const;
    /*!  \overload
    */
    void putVar_all(const unsigned int* dataValues) const;
    /*!  \overload
    */
    void putVar_all(const unsigned long long* dataValues) const;
    /*! 
      Writes the entire data into the netCDF variable.
      This is the simplest interface to use for writing a value in a scalar variable 
      or whenever all the values of a multidimensional variable can all be 
      written at once. The values to be written are associated with the 
      netCDF variable by assuming that the last dimension of the netCDF 
      variable varies fastest in the C interface. 
      
      Take care when using the simplest forms of this interface with 
      record variables when you don't specify how many records are to be 
      written. If you try to write all the values of a record variable 
      into a netCDF file that has no record data yet (hence has 0 records), 
      nothing will be written. Similarly, if you try to write all of a record 
      variable but there are more records in the file than you assume, more data 
      may be written to the file than you supply, which may result in a 
      segmentation violation.
      
      \param dataValues The data values. The order in which the data will be written to the netCDF variable is with the last 
      dimension of the specified variable varying fastest. If the type of data values differs from the netCDF variable type, type conversion will occur. 
      (However, no type conversion is carried out for variables using the user-defined data types:
      ncmpi_Vlen, ncmpi_Opaque, ncmpi_Compound and ncmpi_Enum.)
    */
    void putVar_all(const long long* dataValues) const;

    void putVar_all(const void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const;

    /////////////////////////

    // Writes a single datum into the netCDF variable.
    /*!
      This is an overloaded member function, provided for convenience. 
      It differs from the above function in what argument(s) it accepts. 
      In addition, no data conversion is carried out. This means that 
      the type of the data in memory must match the type of the variable.
    */
    // void putVar_all(const std::vector<MPI_Offset>& index, const void* dataumValue) const;
    /*! \overload
     */ 
    void putVar_all(const std::vector<MPI_Offset>& index, const std::string& dataumValue) const;
    /*!  \overload
    */
    void putVar_all(const std::vector<MPI_Offset>& index, const unsigned char* dataumValue) const;
    /*!  \overload
    */
    void putVar_all(const std::vector<MPI_Offset>& index, const signed char* dataumValue) const;
    /*!  \overload
    */
    void putVar_all(const std::vector<MPI_Offset>& index, const short dataumValue) const;
    /*!  \overload
    */
    void putVar_all(const std::vector<MPI_Offset>& index, const int dataumValue) const;
    /*!  \overload
    */
    void putVar_all(const std::vector<MPI_Offset>& index, const long dataumValue) const;
    /*!  \overload
    */
    void putVar_all(const std::vector<MPI_Offset>& index, const float dataumValue) const;
    /*!  \overload
    */
    void putVar_all(const std::vector<MPI_Offset>& index, const double dataumValue) const;
    /*!  \overload
    */
    void putVar_all(const std::vector<MPI_Offset>& index, const unsigned short dataumValue) const;
    /*!  \overload
    */
    void putVar_all(const std::vector<MPI_Offset>& index, const unsigned int dataumValue) const;
    /*!  \overload
    */
    void putVar_all(const std::vector<MPI_Offset>& index, const unsigned long long dataumValue) const;
    /*! 
      Writes a single datum into the netCDF variable.

      \param index      Vector specifying the index where the data values will be written. The indices are relative to 0, so for example, 
      the first data value of a two-dimensional variable would have index (0,0). The elements of index must correspond to the variable's dimensions. 
      Hence, if the variable uses the unlimited dimension, the first index would correspond to the unlimited dimension.

      \param datumValue The data value. If the type of data values differs from the netCDF variable type, type conversion will occur.
      (However, no type conversion is carried out for variables using the user-defined data types:
      ncmpi_Vlen, ncmpi_Opaque, ncmpi_Compound and ncmpi_Enum.)
    */
    void putVar_all(const std::vector<MPI_Offset>& index, const long long dataumValue) const;

    void putVar_all(const std::vector<MPI_Offset>& index, const void* dataumValue, MPI_Offset bufcount, MPI_Datatype buftype) const;

    /////////////////////////

    // Writes an array of values into the netCDF variable.
    /*!
      This is an overloaded member function, provided for convenience. 
      It differs from the above function in what argument(s) it accepts. 
      In addition, no data conversion is carried out. This means that 
      the type of the data in memory must match the type of the variable.
    */
    // void putVar_all(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const void* dataValues) const;
    /*! \overload
     */ 
    void putVar_all(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const char* dataValues) const;
    /*!  \overload
    */
    void putVar_all(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const unsigned char* dataValues) const;
    /*!  \overload
    */
    void putVar_all(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const signed char* dataValues) const;
    /*!  \overload
    */
    void putVar_all(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const short* dataValues) const;
    /*!  \overload
    */
    void putVar_all(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const int* dataValues) const;
    /*!  \overload
    */
    void putVar_all(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const long* dataValues) const;
    /*!  \overload
    */
    void putVar_all(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const float* dataValues) const;
    /*!  \overload
    */
    void putVar_all(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const double* dataValues) const;
    /*!  \overload
    */
    void putVar_all(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const unsigned short* dataValues) const;
    /*!  \overload
    */
    void putVar_all(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const unsigned int* dataValues) const;
    /*!  \overload
    */
    void putVar_all(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const unsigned long long* dataValues) const;
    /*! 
      Writes an array of values into the netCDF variable.
      The portion of the netCDF variable to write is specified by giving a corner and a vector of edge lengths 
      that refer to an array section of the netCDF variable. The values to be written are associated with 
      the netCDF variable by assuming that the last dimension of the netCDF variable varies fastest. 

      \param startp  Vector specifying the index where the first data values will be written.  The indices are relative to 0, so for 
      example, the first data value of a variable would have index (0, 0, ... , 0). The elements of start correspond, in order, to the 
      variable's dimensions. Hence, if the variable is a record variable, the first index corresponds to the starting record number for writing the data values.

      \param countp  Vector specifying the number of indices selected along each dimension. 
      To write a single value, for example, specify count as (1, 1, ... , 1). The elements of 
      count correspond, in order, to the variable's dimensions. Hence, if the variable is a record 
      variable, the first element of count corresponds to a count of the number of records to write. Note: setting any element 
      of the count array to zero causes the function to exit without error, and without doing anything. 

      \param dataValues The data values. The order in which the data will be written to the netCDF variable is with the last 
      dimension of the specified variable varying fastest. If the type of data values differs from the netCDF variable 
      type, type conversion will occur. (However, no type conversion is 
      carried out for variables using the user-defined data types:
      ncmpi_Vlen, ncmpi_Opaque, ncmpi_Compound and ncmpi_Enum.)
    */
    void putVar_all(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const long long* dataValues) const;
    void putVar_all(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const;

    ////////////////

    // Writes a set of subsampled array values into the netCDF variable.
    /*!
      This is an overloaded member function, provided for convenience. 
      It differs from the above function in what argument(s) it accepts. 
      In addition, no data conversion is carried out. This means that 
      the type of the data in memory must match the type of the variable.
    */
    // void putVar_all(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const void* dataValues) const;
    /*! \overload
     */ 
    void putVar_all(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const char* dataValues) const;
    /*!  \overload
    */
    void putVar_all(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const unsigned char* dataValues) const;
    /*!  \overload
    */
    void putVar_all(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const signed char* dataValues) const;
    /*!  \overload
    */
    void putVar_all(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const short* dataValues) const;
    /*!  \overload
    */
    void putVar_all(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const int* dataValues) const;
    /*!  \overload
    */
    void putVar_all(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const long* dataValues) const;
    /*!  \overload
    */
    void putVar_all(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const float* dataValues) const;
    /*!  \overload
    */
    void putVar_all(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const double* dataValues) const;
    /*!  \overload
    */
    void putVar_all(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const unsigned short* dataValues) const;
    /*!  \overload
    */
    void putVar_all(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const unsigned int* dataValues) const;
    /*!  \overload
    */
    void putVar_all(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const unsigned long long* dataValues) const;
    /*! 
      Writes an array of values into the netCDF variable.
      The subsampled array section is specified by giving a corner, a vector of counts, and a stride vector. 

      \param startp  Vector specifying the index where the first data values will be written.  The indices are relative to 0, so for 
      example, the first data value of a variable would have index (0, 0, ... , 0). The elements of start correspond, in order, to the 
      variable's dimensions. Hence, if the variable is a record variable, the first index corresponds to the starting record number for writing the data values.

      \param countp  Vector specifying the number of indices selected along each dimension. 
      To write a single value, for example, specify count as (1, 1, ... , 1). The elements of 
      count correspond, in order, to the variable's dimensions. Hence, if the variable is a record 
      variable, the first element of count corresponds to a count of the number of records to write. Note: setting any element 
      of the count array to zero causes the function to exit without error, and without doing anything. 

      \param stridep  A vector of MPI_Offset integers that specifies the sampling interval along each dimension of the netCDF variable. 
      The elements of the stride vector correspond, in order, to the netCDF variable's dimensions (stride[0] gives the sampling interval 
      along the most slowly varying dimension of the netCDF variable). Sampling intervals are specified in type-independent units of 
      elements (a value of 1 selects consecutive elements of the netCDF variable along the corresponding dimension, a value of 2 selects 
      every other element, etc.). A NULL stride argument is treated as (1, 1, ... , 1).

      \param dataValues The data values. The order in which the data will be written to the netCDF variable is with the last 
      dimension of the specified variable varying fastest. If the type of data values differs from the netCDF variable type, type conversion will occur. 
      (However, no type conversion is  carried out for variables using the user-defined data types: ncmpi_Vlen, ncmpi_Opaque, ncmpi_Compound and ncmpi_Enum.
    */
    void putVar_all(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const long long* dataValues) const;

    void putVar_all(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const;

    ////////////////

    // Writes a mapped array section of values into the netCDF variable.
    /*!
      This is an overloaded member function, provided for convenience. 
      It differs from the above function in what argument(s) it accepts. 
      In addition, no data conversion is carried out. This means that 
      the type of the data in memory must match the type of the variable.
    */
    // void putVar_all(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const void* dataValues) const;
    /*! \overload
     */ 
    void putVar_all(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const char* dataValues) const;
    /*!  \overload
    */
    void putVar_all(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const unsigned char* dataValues) const;
    /*!  \overload
    */
    void putVar_all(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const signed char* dataValues) const;
    /*!  \overload
    */
    void putVar_all(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const short* dataValues) const;
    /*!  \overload
    */
    void putVar_all(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const int* dataValues) const;
    /*!  \overload
    */
    void putVar_all(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const long* dataValues) const;
    /*!  \overload
    */
    void putVar_all(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const float* dataValues) const;
    /*!  \overload
    */
    void putVar_all(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const double* dataValues) const;
    /*!  \overload
    */
    void putVar_all(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const unsigned short* dataValues) const;
    /*!  \overload
    */
    void putVar_all(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const unsigned int* dataValues) const;
    /*!  \overload
    */
    void putVar_all(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const unsigned long long* dataValues) const;
    /*! 
      Writes a mapped array section of values into the netCDF variable.
      The mapped array section is specified by giving a corner, a vector of counts, a stride vector, and an index mapping vector. 
      The index mapping vector is a vector of integers that specifies the mapping between the dimensions of a netCDF variable and the in-memory structure of the internal data array. 
      No assumptions are made about the ordering or length of the dimensions of the data array. 

      \param countp  Vector specifying the number of indices selected along each dimension. 
      To write a single value, for example, specify count as (1, 1, ... , 1). The elements of 
      count correspond, in order, to the variable's dimensions. Hence, if the variable is a record 
      variable, the first element of count corresponds to a count of the number of records to write. Note: setting any element 
      of the count array to zero causes the function to exit without error, and without doing anything. 

      \param stridep  A vector of MPI_Offset integers that specifies the sampling interval along each dimension of the netCDF variable. 
      The elements of the stride vector correspond, in order, to the netCDF variable's dimensions (stride[0] gives the sampling interval 
      along the most slowly varying dimension of the netCDF variable). Sampling intervals are specified in type-independent units of 
      elements (a value of 1 selects consecutive elements of the netCDF variable along the corresponding dimension, a value of 2 selects 
      every other element, etc.). A NULL stride argument is treated as (1, 1, ... , 1).

      \param imap Vector  specifies the mapping between the dimensions of a netCDF variable and the in-memory structure of the internal data array. 
      The elements of the index mapping vector correspond, in order, to the netCDF variable's dimensions (imap[0] gives the distance between elements 
      of the internal array corresponding to the most slowly varying dimension of the netCDF variable). Distances between elements are 
      specified in type-independent units of elements (the distance between internal elements that occupy adjacent memory locations is 
      1 and not the element's byte-length as in netCDF 2). A NULL argument means the memory-resident values have the same structure as 
      the associated netCDF variable.

      \param dataValues The data values. The order in which the data will be written to the netCDF variable is with the last 
      dimension of the specified variable varying fastest. If the type of data values differs from the netCDF variable type, type conversion will occur. 
     (However, no type conversion is carried out for variables using the user-defined data types:  ncmpi_Vlen, ncmpi_Opaque, ncmpi_Compound and ncmpi_Enum.)
*/
    void putVar_all(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const long long* dataValues) const;

    void putVar_all(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const;

    ////////////////////

    // Writes a list of subarrays into the netCDF variable. (independent I/O APIs)
    /*!
      This is an overloaded member function, provided for convenience. 
      It differs from the above function in what argument(s) it accepts. 
      In addition, no data conversion is carried out. This means that 
      the type of the data in memory must match the type of the variable.
    */
    // void putVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const void* dataValues) const;
    /*! \overload
     */ 
    void putVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const char* dataValues) const;
    /*!  \overload
    */
    void putVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const unsigned char* dataValues) const;
    /*!  \overload
    */
    void putVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const signed char* dataValues) const;
    /*!  \overload
    */
    void putVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const short* dataValues) const;
    /*!  \overload
    */
    void putVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const int* dataValues) const;
    /*!  \overload
    */
    void putVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const long* dataValues) const;
    /*!  \overload
    */
    void putVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const float* dataValues) const;
    /*!  \overload
    */
    void putVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const double* dataValues) const;
    /*!  \overload
    */
    void putVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const unsigned short* dataValues) const;
    /*!  \overload
    */
    void putVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const unsigned int* dataValues) const;
    /*!  \overload
    */
    void putVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const unsigned long long* dataValues) const;
    /*! 
      Writes a list of subarrays into the netCDF variable.
      Each subarray i is specified by giving a corner (starts[i][*]) and a vector of edge lengths (counts[i][*]). 
      The values to be written are associated with the netCDF variable by assuming that the last dimension of the netCDF variable varies fastest. 

      \param num
      Number of subarrays.

      \param starts
      2D array of size [num][ndims] where num is the number of subarrays to get and ndims if the number of dimensions of the specified variable.
      Each subarray i is specified by starts[i][*], the first of the data values will be written, and counts[i][*], edge lengths of the subarray.
      The indices are relative to 0, so for example, the first data value of a variable would have index (0, 0, ... , 0). 
      The length of the second dimension of starts and counts must be the same as the number of dimensions of the specified variable. 
      The elements of starts's second dimension correspond, in order, to the variable's dimensions. Hence, if the variable is a record variable, 
      the first element starts[*][0] would correspond to the starting record number for writing the data values.

      \param counts
      2D array of size [num][ndims] where num is the number of subarrays to get and ndims if the number of dimensions of the specified variable.
      To write a single value, for example, specify count as (1, 1, ... , 1). The length of each count[i] is the number of 
      dimensions of the specified variable. The elements of each count[i] correspond, in order, to the variable's dimensions. 
      Hence, if the variable is a record variable, the first element of count[i] corresponds to a count of the number of records to write.
      Note: setting any element of the count array to zero causes the function to exit without error, and without doing anything. 

      \param dataValues Pointer to the location into which the data value is written. If the type of 
      data value differs from the netCDF variable type, type conversion will occur.
    */
    void putVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const long long* dataValues) const;
    void putVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const;

    ////////////////

    // Writes an array of values into the netCDF variable. (collective I/O APIs)
    /*!
      This is an overloaded member function, provided for convenience. 
      It differs from the above function in what argument(s) it accepts. 
      In addition, no data conversion is carried out. This means that 
      the type of the data in memory must match the type of the variable.
    */
    // void putVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const void* dataValues) const;
    /*! \overload
     */ 
    void putVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const char* dataValues) const;
    /*!  \overload
    */
    void putVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const unsigned char* dataValues) const;
    /*!  \overload
    */
    void putVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const signed char* dataValues) const;
    /*!  \overload
    */
    void putVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const short* dataValues) const;
    /*!  \overload
    */
    void putVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const int* dataValues) const;
    /*!  \overload
    */
    void putVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const long* dataValues) const;
    /*!  \overload
    */
    void putVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const float* dataValues) const;
    /*!  \overload
    */
    void putVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const double* dataValues) const;
    /*!  \overload
    */
    void putVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const unsigned short* dataValues) const;
    /*!  \overload
    */
    void putVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const unsigned int* dataValues) const;
    /*!  \overload
    */
    void putVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const unsigned long long* dataValues) const;
    /*! 
      Writes an array of values into the netCDF variable.
      The portion of the netCDF variable to write is specified by giving a corner and a vector of edge lengths 
      that refer to an array section of the netCDF variable. The values to be written are associated with 
      the netCDF variable by assuming that the last dimension of the netCDF variable varies fastest. 

      \param startp  Vector specifying the index where the first data values will be written.  The indices are relative to 0, so for 
      example, the first data value of a variable would have index (0, 0, ... , 0). The elements of start correspond, in order, to the 
      variable's dimensions. Hence, if the variable is a record variable, the first index corresponds to the starting record number for writing the data values.

      \param countp  Vector specifying the number of indices selected along each dimension. 
      To write a single value, for example, specify count as (1, 1, ... , 1). The elements of 
      count correspond, in order, to the variable's dimensions. Hence, if the variable is a record 
      variable, the first element of count corresponds to a count of the number of records to write. Note: setting any element 
      of the count array to zero causes the function to exit without error, and without doing anything. 

      \param dataValues The data values. The order in which the data will be written to the netCDF variable is with the last 
      dimension of the specified variable varying fastest. If the type of data values differs from the netCDF variable 
      type, type conversion will occur. (However, no type conversion is 
      carried out for variables using the user-defined data types:
      ncmpi_Vlen, ncmpi_Opaque, ncmpi_Compound and ncmpi_Enum.)
    */
    void putVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const long long* dataValues) const;
    void putVarn_all(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const;

    /* vard APIs take filetype and buftype */
    void putVard    (MPI_Datatype filetype, const void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const;
    void putVard_all(MPI_Datatype filetype, const void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype) const;

    ////////////////

    //  Nonblocking data writing

    ////////////////////
   
   
    // Writes the entire data into the netCDF variable.
    /*!
      This is an overloaded member function, provided for convenience. 
      It differs from the above function in what argument(s) it accepts. 
      In addition, no data conversion is carried out. This means that 
      the type of the data in memory must match the type of the variable.
    */
    // void iputVar(const void* dataValues, int *req) const;
    /*! \overload
     */ 
    void iputVar(const char* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVar(const unsigned char* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVar(const signed char* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVar(const short* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVar(const int* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVar(const long* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVar(const float* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVar(const double* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVar(const unsigned short* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVar(const unsigned int* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVar(const unsigned long long* dataValues, int *req) const;
    /*! 
      Writes the entire data into the netCDF variable.
      This is the simplest interface to use for writing a value in a scalar variable 
      or whenever all the values of a multidimensional variable can all be 
      written at once. The values to be written are associated with the 
      netCDF variable by assuming that the last dimension of the netCDF 
      variable varies fastest in the C interface. 
      
      Take care when using the simplest forms of this interface with 
      record variables when you don't specify how many records are to be 
      written. If you try to write all the values of a record variable 
      into a netCDF file that has no record data yet (hence has 0 records), 
      nothing will be written. Similarly, if you try to write all of a record 
      variable but there are more records in the file than you assume, more data 
      may be written to the file than you supply, which may result in a 
      segmentation violation.
      
      \param dataValues The data values. The order in which the data will be written to the netCDF variable is with the last 
      dimension of the specified variable varying fastest. If the type of data values differs from the netCDF variable type, type conversion will occur. 
      (However, no type conversion is carried out for variables using the user-defined data types:
      ncmpi_Vlen, ncmpi_Opaque, ncmpi_Compound and ncmpi_Enum.)
    */
    void iputVar(const long long* dataValues, int *req) const;

    void iputVar(const void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype, int *req) const;

    /////////////////////////

    // Writes a single datum into the netCDF variable.
    /*!
      This is an overloaded member function, provided for convenience. 
      It differs from the above function in what argument(s) it accepts. 
      In addition, no data conversion is carried out. This means that 
      the type of the data in memory must match the type of the variable.
    */
    // void iputVar(const std::vector<MPI_Offset>& index, const void* dataumValue, int *req) const;
    /*! \overload
     */ 
    void iputVar(const std::vector<MPI_Offset>& index, const std::string& dataumValue, int *req) const;
    /*!  \overload
    */
    void iputVar(const std::vector<MPI_Offset>& index, const unsigned char* dataumValue, int *req) const;
    /*!  \overload
    */
    void iputVar(const std::vector<MPI_Offset>& index, const signed char* dataumValue, int *req) const;
    /*!  \overload
    */
    void iputVar(const std::vector<MPI_Offset>& index, const short dataumValue, int *req) const;
    /*!  \overload
    */
    void iputVar(const std::vector<MPI_Offset>& index, const int dataumValue, int *req) const;
    /*!  \overload
    */
    void iputVar(const std::vector<MPI_Offset>& index, const long dataumValue, int *req) const;
    /*!  \overload
    */
    void iputVar(const std::vector<MPI_Offset>& index, const float dataumValue, int *req) const;
    /*!  \overload
    */
    void iputVar(const std::vector<MPI_Offset>& index, const double dataumValue, int *req) const;
    /*!  \overload
    */
    void iputVar(const std::vector<MPI_Offset>& index, const unsigned short dataumValue, int *req) const;
    /*!  \overload
    */
    void iputVar(const std::vector<MPI_Offset>& index, const unsigned int dataumValue, int *req) const;
    /*!  \overload
    */
    void iputVar(const std::vector<MPI_Offset>& index, const unsigned long long dataumValue, int *req) const;
    /*! 
      Writes a single datum into the netCDF variable.

      \param index      Vector specifying the index where the data values will be written. The indices are relative to 0, so for example, 
      the first data value of a two-dimensional variable would have index (0,0). The elements of index must correspond to the variable's dimensions. 
      Hence, if the variable uses the unlimited dimension, the first index would correspond to the unlimited dimension.

      \param datumValue The data value. If the type of data values differs from the netCDF variable type, type conversion will occur.
      (However, no type conversion is carried out for variables using the user-defined data types:
      ncmpi_Vlen, ncmpi_Opaque, ncmpi_Compound and ncmpi_Enum.)
    */
    void iputVar(const std::vector<MPI_Offset>& index, const long long dataumValue, int *req) const;

    void iputVar(const std::vector<MPI_Offset>& index, const void* dataumValue, MPI_Offset bufcount, MPI_Datatype buftype, int *req) const;

    /////////////////////////

    // Writes an array of values into the netCDF variable.
    /*!
      This is an overloaded member function, provided for convenience. 
      It differs from the above function in what argument(s) it accepts. 
      In addition, no data conversion is carried out. This means that 
      the type of the data in memory must match the type of the variable.
    */
    // void iputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const void* dataValues, int *req) const;
    /*! \overload
     */ 
    void iputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const char* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const unsigned char* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const signed char* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const short* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const int* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const long* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const float* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const double* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const unsigned short* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const unsigned int* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const unsigned long long* dataValues, int *req) const;
    /*! 
      Writes an array of values into the netCDF variable.
      The portion of the netCDF variable to write is specified by giving a corner and a vector of edge lengths 
      that refer to an array section of the netCDF variable. The values to be written are associated with 
      the netCDF variable by assuming that the last dimension of the netCDF variable varies fastest. 

      \param startp  Vector specifying the index where the first data values will be written.  The indices are relative to 0, so for 
      example, the first data value of a variable would have index (0, 0, ... , 0). The elements of start correspond, in order, to the 
      variable's dimensions. Hence, if the variable is a record variable, the first index corresponds to the starting record number for writing the data values.

      \param countp  Vector specifying the number of indices selected along each dimension. 
      To write a single value, for example, specify count as (1, 1, ... , 1). The elements of 
      count correspond, in order, to the variable's dimensions. Hence, if the variable is a record 
      variable, the first element of count corresponds to a count of the number of records to write. Note: setting any element 
      of the count array to zero causes the function to exit without error, and without doing anything. 

      \param dataValues The data values. The order in which the data will be written to the netCDF variable is with the last 
      dimension of the specified variable varying fastest. If the type of data values differs from the netCDF variable 
      type, type conversion will occur. (However, no type conversion is 
      carried out for variables using the user-defined data types:
      ncmpi_Vlen, ncmpi_Opaque, ncmpi_Compound and ncmpi_Enum.)
    */
    void iputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const long long* dataValues, int *req) const;
    void iputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype, int *req) const;

    ////////////////

    // Writes a set of subsampled array values into the netCDF variable.
    /*!
      This is an overloaded member function, provided for convenience. 
      It differs from the above function in what argument(s) it accepts. 
      In addition, no data conversion is carried out. This means that 
      the type of the data in memory must match the type of the variable.
    */
    // void iputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const void* dataValues, int *req) const;
    /*! \overload
     */ 
    void iputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const char* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const unsigned char* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const signed char* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const short* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const int* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const long* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const float* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const double* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const unsigned short* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const unsigned int* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const unsigned long long* dataValues, int *req) const;
    /*! 
      Writes an array of values into the netCDF variable.
      The subsampled array section is specified by giving a corner, a vector of counts, and a stride vector. 

      \param startp  Vector specifying the index where the first data values will be written.  The indices are relative to 0, so for 
      example, the first data value of a variable would have index (0, 0, ... , 0). The elements of start correspond, in order, to the 
      variable's dimensions. Hence, if the variable is a record variable, the first index corresponds to the starting record number for writing the data values.

      \param countp  Vector specifying the number of indices selected along each dimension. 
      To write a single value, for example, specify count as (1, 1, ... , 1). The elements of 
      count correspond, in order, to the variable's dimensions. Hence, if the variable is a record 
      variable, the first element of count corresponds to a count of the number of records to write. Note: setting any element 
      of the count array to zero causes the function to exit without error, and without doing anything. 

      \param stridep  A vector of MPI_Offset integers that specifies the sampling interval along each dimension of the netCDF variable. 
      The elements of the stride vector correspond, in order, to the netCDF variable's dimensions (stride[0] gives the sampling interval 
      along the most slowly varying dimension of the netCDF variable). Sampling intervals are specified in type-independent units of 
      elements (a value of 1 selects consecutive elements of the netCDF variable along the corresponding dimension, a value of 2 selects 
      every other element, etc.). A NULL stride argument is treated as (1, 1, ... , 1).

      \param dataValues The data values. The order in which the data will be written to the netCDF variable is with the last 
      dimension of the specified variable varying fastest. If the type of data values differs from the netCDF variable type, type conversion will occur. 
      (However, no type conversion is  carried out for variables using the user-defined data types: ncmpi_Vlen, ncmpi_Opaque, ncmpi_Compound and ncmpi_Enum.
    */
    void iputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const long long* dataValues, int *req) const;

    void iputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype, int *req) const;

    ////////////////

    // Writes a mapped array section of values into the netCDF variable.
    /*!
      This is an overloaded member function, provided for convenience. 
      It differs from the above function in what argument(s) it accepts. 
      In addition, no data conversion is carried out. This means that 
      the type of the data in memory must match the type of the variable.
    */
    // void iputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const void* dataValues, int *req) const;
    /*! \overload
     */ 
    void iputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const char* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const unsigned char* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const signed char* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const short* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const int* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const long* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const float* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const double* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const unsigned short* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const unsigned int* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const unsigned long long* dataValues, int *req) const;
    /*! 
      Writes a mapped array section of values into the netCDF variable.
      The mapped array section is specified by giving a corner, a vector of counts, a stride vector, and an index mapping vector. 
      The index mapping vector is a vector of integers that specifies the mapping between the dimensions of a netCDF variable and the in-memory structure of the internal data array. 
      No assumptions are made about the ordering or length of the dimensions of the data array. 

      \param countp  Vector specifying the number of indices selected along each dimension. 
      To write a single value, for example, specify count as (1, 1, ... , 1). The elements of 
      count correspond, in order, to the variable's dimensions. Hence, if the variable is a record 
      variable, the first element of count corresponds to a count of the number of records to write. Note: setting any element 
      of the count array to zero causes the function to exit without error, and without doing anything. 

      \param stridep  A vector of MPI_Offset integers that specifies the sampling interval along each dimension of the netCDF variable. 
      The elements of the stride vector correspond, in order, to the netCDF variable's dimensions (stride[0] gives the sampling interval 
      along the most slowly varying dimension of the netCDF variable). Sampling intervals are specified in type-independent units of 
      elements (a value of 1 selects consecutive elements of the netCDF variable along the corresponding dimension, a value of 2 selects 
      every other element, etc.). A NULL stride argument is treated as (1, 1, ... , 1).

      \param imap Vector  specifies the mapping between the dimensions of a netCDF variable and the in-memory structure of the internal data array. 
      The elements of the index mapping vector correspond, in order, to the netCDF variable's dimensions (imap[0] gives the distance between elements 
      of the internal array corresponding to the most slowly varying dimension of the netCDF variable). Distances between elements are 
      specified in type-independent units of elements (the distance between internal elements that occupy adjacent memory locations is 
      1 and not the element's byte-length as in netCDF 2). A NULL argument means the memory-resident values have the same structure as 
      the associated netCDF variable.

      \param dataValues The data values. The order in which the data will be written to the netCDF variable is with the last 
      dimension of the specified variable varying fastest. If the type of data values differs from the netCDF variable type, type conversion will occur. 
     (However, no type conversion is carried out for variables using the user-defined data types:  ncmpi_Vlen, ncmpi_Opaque, ncmpi_Compound and ncmpi_Enum.)
*/
    void iputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const long long* dataValues, int *req) const;

    void iputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype, int *req) const;

    ////////////////////

    // Writes a list of subarrays into the netCDF variable. (independent I/O APIs)
    /*!
      This is an overloaded member function, provided for convenience. 
      It differs from the above function in what argument(s) it accepts. 
      In addition, no data conversion is carried out. This means that 
      the type of the data in memory must match the type of the variable.
    */
    // void putVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const void* dataValues) const;
    /*! \overload
     */ 
    void iputVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const char* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const unsigned char* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const signed char* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const short* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const int* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const long* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const float* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const double* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const unsigned short* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const unsigned int* dataValues, int *req) const;
    /*!  \overload
    */
    void iputVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const unsigned long long* dataValues, int *req) const;
    /*! 
      Writes a list of subarrays into the netCDF variable.
      Each subarray i is specified by giving a corner (starts[i][*]) and a vector of edge lengths (counts[i][*]). 
      The values to be written are associated with the netCDF variable by assuming that the last dimension of the netCDF variable varies fastest. 

      \param num
      Number of subarrays.

      \param starts
      2D array of size [num][ndims] where num is the number of subarrays to get and ndims if the number of dimensions of the specified variable.
      Each subarray i is specified by starts[i][*], the first of the data values will be written, and counts[i][*], edge lengths of the subarray.
      The indices are relative to 0, so for example, the first data value of a variable would have index (0, 0, ... , 0). 
      The length of the second dimension of starts and counts must be the same as the number of dimensions of the specified variable. 
      The elements of starts's second dimension correspond, in order, to the variable's dimensions. Hence, if the variable is a record variable, 
      the first element starts[*][0] would correspond to the starting record number for writing the data values.

      \param counts
      2D array of size [num][ndims] where num is the number of subarrays to get and ndims if the number of dimensions of the specified variable.
      To write a single value, for example, specify count as (1, 1, ... , 1). The length of each count[i] is the number of 
      dimensions of the specified variable. The elements of each count[i] correspond, in order, to the variable's dimensions. 
      Hence, if the variable is a record variable, the first element of count[i] corresponds to a count of the number of records to write.
      Note: setting any element of the count array to zero causes the function to exit without error, and without doing anything. 

      \param dataValues Pointer to the location into which the data value is written. If the type of 
      data value differs from the netCDF variable type, type conversion will occur.
    */
    void iputVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const long long* dataValues, int *req) const;
    void iputVarn(const int num, MPI_Offset* const starts[], MPI_Offset* const counts[], const void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype, int *req) const;

    ////////////////////

    //  Buffered nonblocking data writing

    ////////////////////
   
    // Writes the entire data into the netCDF variable.
    /*!
      This is an overloaded member function, provided for convenience. 
      It differs from the above function in what argument(s) it accepts. 
      In addition, no data conversion is carried out. This means that 
      the type of the data in memory must match the type of the variable.
    */
    // void bputVar(const void* dataValues, int *req) const;
    /*! \overload
     */ 
    void bputVar(const char* dataValues, int *req) const;
    /*!  \overload
    */
    void bputVar(const unsigned char* dataValues, int *req) const;
    /*!  \overload
    */
    void bputVar(const signed char* dataValues, int *req) const;
    /*!  \overload
    */
    void bputVar(const short* dataValues, int *req) const;
    /*!  \overload
    */
    void bputVar(const int* dataValues, int *req) const;
    /*!  \overload
    */
    void bputVar(const long* dataValues, int *req) const;
    /*!  \overload
    */
    void bputVar(const float* dataValues, int *req) const;
    /*!  \overload
    */
    void bputVar(const double* dataValues, int *req) const;
    /*!  \overload
    */
    void bputVar(const unsigned short* dataValues, int *req) const;
    /*!  \overload
    */
    void bputVar(const unsigned int* dataValues, int *req) const;
    /*!  \overload
    */
    void bputVar(const unsigned long long* dataValues, int *req) const;
    /*! 
      Writes the entire data into the netCDF variable.
      This is the simplest interface to use for writing a value in a scalar variable 
      or whenever all the values of a multidimensional variable can all be 
      written at once. The values to be written are associated with the 
      netCDF variable by assuming that the last dimension of the netCDF 
      variable varies fastest in the C interface. 
      
      Take care when using the simplest forms of this interface with 
      record variables when you don't specify how many records are to be 
      written. If you try to write all the values of a record variable 
      into a netCDF file that has no record data yet (hence has 0 records), 
      nothing will be written. Similarly, if you try to write all of a record 
      variable but there are more records in the file than you assume, more data 
      may be written to the file than you supply, which may result in a 
      segmentation violation.
      
      \param dataValues The data values. The order in which the data will be written to the netCDF variable is with the last 
      dimension of the specified variable varying fastest. If the type of data values differs from the netCDF variable type, type conversion will occur. 
      (However, no type conversion is carried out for variables using the user-defined data types:
      ncmpi_Vlen, ncmpi_Opaque, ncmpi_Compound and ncmpi_Enum.)
    */
    void bputVar(const long long* dataValues, int *req) const;

    void bputVar(const void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype, int *req) const;

    /////////////////////////

    // Writes a single datum into the netCDF variable.
    /*!
      This is an overloaded member function, provided for convenience. 
      It differs from the above function in what argument(s) it accepts. 
      In addition, no data conversion is carried out. This means that 
      the type of the data in memory must match the type of the variable.
    */
    // void bputVar(const std::vector<MPI_Offset>& index, const void* dataumValue, int *req) const;
    /*! \overload
     */ 
    void bputVar(const std::vector<MPI_Offset>& index, const std::string& dataumValue, int *req) const;
    /*!  \overload
    */
    void bputVar(const std::vector<MPI_Offset>& index, const unsigned char* dataumValue, int *req) const;
    /*!  \overload
    */
    void bputVar(const std::vector<MPI_Offset>& index, const signed char* dataumValue, int *req) const;
    /*!  \overload
    */
    void bputVar(const std::vector<MPI_Offset>& index, const short dataumValue, int *req) const;
    /*!  \overload
    */
    void bputVar(const std::vector<MPI_Offset>& index, const int dataumValue, int *req) const;
    /*!  \overload
    */
    void bputVar(const std::vector<MPI_Offset>& index, const long dataumValue, int *req) const;
    /*!  \overload
    */
    void bputVar(const std::vector<MPI_Offset>& index, const float dataumValue, int *req) const;
    /*!  \overload
    */
    void bputVar(const std::vector<MPI_Offset>& index, const double dataumValue, int *req) const;
    /*!  \overload
    */
    void bputVar(const std::vector<MPI_Offset>& index, const unsigned short dataumValue, int *req) const;
    /*!  \overload
    */
    void bputVar(const std::vector<MPI_Offset>& index, const unsigned int dataumValue, int *req) const;
    /*!  \overload
    */
    void bputVar(const std::vector<MPI_Offset>& index, const unsigned long long dataumValue, int *req) const;
    /*! 
      Writes a single datum into the netCDF variable.

      \param index      Vector specifying the index where the data values will be written. The indices are relative to 0, so for example, 
      the first data value of a two-dimensional variable would have index (0,0). The elements of index must correspond to the variable's dimensions. 
      Hence, if the variable uses the unlimited dimension, the first index would correspond to the unlimited dimension.

      \param datumValue The data value. If the type of data values differs from the netCDF variable type, type conversion will occur.
      (However, no type conversion is carried out for variables using the user-defined data types:
      ncmpi_Vlen, ncmpi_Opaque, ncmpi_Compound and ncmpi_Enum.)
    */
    void bputVar(const std::vector<MPI_Offset>& index, const long long dataumValue, int *req) const;

    void bputVar(const std::vector<MPI_Offset>& index, const void* dataumValue, MPI_Offset bufcount, MPI_Datatype buftype, int *req) const;

    /////////////////////////

    // Writes an array of values into the netCDF variable.
    /*!
      This is an overloaded member function, provided for convenience. 
      It differs from the above function in what argument(s) it accepts. 
      In addition, no data conversion is carried out. This means that 
      the type of the data in memory must match the type of the variable.
    */
    // void bputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const void* dataValues, int *req) const;
    /*! \overload
     */ 
    void bputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const char* dataValues, int *req) const;
    /*!  \overload
    */
    void bputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const unsigned char* dataValues, int *req) const;
    /*!  \overload
    */
    void bputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const signed char* dataValues, int *req) const;
    /*!  \overload
    */
    void bputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const short* dataValues, int *req) const;
    /*!  \overload
    */
    void bputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const int* dataValues, int *req) const;
    /*!  \overload
    */
    void bputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const long* dataValues, int *req) const;
    /*!  \overload
    */
    void bputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const float* dataValues, int *req) const;
    /*!  \overload
    */
    void bputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const double* dataValues, int *req) const;
    /*!  \overload
    */
    void bputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const unsigned short* dataValues, int *req) const;
    /*!  \overload
    */
    void bputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const unsigned int* dataValues, int *req) const;
    /*!  \overload
    */
    void bputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const unsigned long long* dataValues, int *req) const;
    /*! 
      Writes an array of values into the netCDF variable.
      The portion of the netCDF variable to write is specified by giving a corner and a vector of edge lengths 
      that refer to an array section of the netCDF variable. The values to be written are associated with 
      the netCDF variable by assuming that the last dimension of the netCDF variable varies fastest. 

      \param startp  Vector specifying the index where the first data values will be written.  The indices are relative to 0, so for 
      example, the first data value of a variable would have index (0, 0, ... , 0). The elements of start correspond, in order, to the 
      variable's dimensions. Hence, if the variable is a record variable, the first index corresponds to the starting record number for writing the data values.

      \param countp  Vector specifying the number of indices selected along each dimension. 
      To write a single value, for example, specify count as (1, 1, ... , 1). The elements of 
      count correspond, in order, to the variable's dimensions. Hence, if the variable is a record 
      variable, the first element of count corresponds to a count of the number of records to write. Note: setting any element 
      of the count array to zero causes the function to exit without error, and without doing anything. 

      \param dataValues The data values. The order in which the data will be written to the netCDF variable is with the last 
      dimension of the specified variable varying fastest. If the type of data values differs from the netCDF variable 
      type, type conversion will occur. (However, no type conversion is 
      carried out for variables using the user-defined data types:
      ncmpi_Vlen, ncmpi_Opaque, ncmpi_Compound and ncmpi_Enum.)
    */
    void bputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const long long* dataValues, int *req) const;
    void bputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype, int *req) const;

    ////////////////

    // Writes a set of subsampled array values into the netCDF variable.
    /*!
      This is an overloaded member function, provided for convenience. 
      It differs from the above function in what argument(s) it accepts. 
      In addition, no data conversion is carried out. This means that 
      the type of the data in memory must match the type of the variable.
    */
    // void bputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const void* dataValues, int *req) const;
    /*! \overload
     */ 
    void bputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const char* dataValues, int *req) const;
    /*!  \overload
    */
    void bputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const unsigned char* dataValues, int *req) const;
    /*!  \overload
    */
    void bputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const signed char* dataValues, int *req) const;
    /*!  \overload
    */
    void bputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const short* dataValues, int *req) const;
    /*!  \overload
    */
    void bputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const int* dataValues, int *req) const;
    /*!  \overload
    */
    void bputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const long* dataValues, int *req) const;
    /*!  \overload
    */
    void bputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const float* dataValues, int *req) const;
    /*!  \overload
    */
    void bputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const double* dataValues, int *req) const;
    /*!  \overload
    */
    void bputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const unsigned short* dataValues, int *req) const;
    /*!  \overload
    */
    void bputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const unsigned int* dataValues, int *req) const;
    /*!  \overload
    */
    void bputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const unsigned long long* dataValues, int *req) const;
    /*! 
      Writes an array of values into the netCDF variable.
      The subsampled array section is specified by giving a corner, a vector of counts, and a stride vector. 

      \param startp  Vector specifying the index where the first data values will be written.  The indices are relative to 0, so for 
      example, the first data value of a variable would have index (0, 0, ... , 0). The elements of start correspond, in order, to the 
      variable's dimensions. Hence, if the variable is a record variable, the first index corresponds to the starting record number for writing the data values.

      \param countp  Vector specifying the number of indices selected along each dimension. 
      To write a single value, for example, specify count as (1, 1, ... , 1). The elements of 
      count correspond, in order, to the variable's dimensions. Hence, if the variable is a record 
      variable, the first element of count corresponds to a count of the number of records to write. Note: setting any element 
      of the count array to zero causes the function to exit without error, and without doing anything. 

      \param stridep  A vector of MPI_Offset integers that specifies the sampling interval along each dimension of the netCDF variable. 
      The elements of the stride vector correspond, in order, to the netCDF variable's dimensions (stride[0] gives the sampling interval 
      along the most slowly varying dimension of the netCDF variable). Sampling intervals are specified in type-independent units of 
      elements (a value of 1 selects consecutive elements of the netCDF variable along the corresponding dimension, a value of 2 selects 
      every other element, etc.). A NULL stride argument is treated as (1, 1, ... , 1).

      \param dataValues The data values. The order in which the data will be written to the netCDF variable is with the last 
      dimension of the specified variable varying fastest. If the type of data values differs from the netCDF variable type, type conversion will occur. 
      (However, no type conversion is  carried out for variables using the user-defined data types: ncmpi_Vlen, ncmpi_Opaque, ncmpi_Compound and ncmpi_Enum.
    */
    void bputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const long long* dataValues, int *req) const;

    void bputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype, int *req) const;

    ////////////////

    // Writes a mapped array section of values into the netCDF variable.
    /*!
      This is an overloaded member function, provided for convenience. 
      It differs from the above function in what argument(s) it accepts. 
      In addition, no data conversion is carried out. This means that 
      the type of the data in memory must match the type of the variable.
    */
    // void bputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const void* dataValues, int *req) const;
    /*! \overload
     */ 
    void bputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const char* dataValues, int *req) const;
    /*!  \overload
    */
    void bputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const unsigned char* dataValues, int *req) const;
    /*!  \overload
    */
    void bputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const signed char* dataValues, int *req) const;
    /*!  \overload
    */
    void bputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const short* dataValues, int *req) const;
    /*!  \overload
    */
    void bputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const int* dataValues, int *req) const;
    /*!  \overload
    */
    void bputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const long* dataValues, int *req) const;
    /*!  \overload
    */
    void bputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const float* dataValues, int *req) const;
    /*!  \overload
    */
    void bputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const double* dataValues, int *req) const;
    /*!  \overload
    */
    void bputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const unsigned short* dataValues, int *req) const;
    /*!  \overload
    */
    void bputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const unsigned int* dataValues, int *req) const;
    /*!  \overload
    */
    void bputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const unsigned long long* dataValues, int *req) const;
    /*! 
      Writes a mapped array section of values into the netCDF variable.
      The mapped array section is specified by giving a corner, a vector of counts, a stride vector, and an index mapping vector. 
      The index mapping vector is a vector of integers that specifies the mapping between the dimensions of a netCDF variable and the in-memory structure of the internal data array. 
      No assumptions are made about the ordering or length of the dimensions of the data array. 

      \param countp  Vector specifying the number of indices selected along each dimension. 
      To write a single value, for example, specify count as (1, 1, ... , 1). The elements of 
      count correspond, in order, to the variable's dimensions. Hence, if the variable is a record 
      variable, the first element of count corresponds to a count of the number of records to write. Note: setting any element 
      of the count array to zero causes the function to exit without error, and without doing anything. 

      \param stridep  A vector of MPI_Offset integers that specifies the sampling interval along each dimension of the netCDF variable. 
      The elements of the stride vector correspond, in order, to the netCDF variable's dimensions (stride[0] gives the sampling interval 
      along the most slowly varying dimension of the netCDF variable). Sampling intervals are specified in type-independent units of 
      elements (a value of 1 selects consecutive elements of the netCDF variable along the corresponding dimension, a value of 2 selects 
      every other element, etc.). A NULL stride argument is treated as (1, 1, ... , 1).

      \param imap Vector  specifies the mapping between the dimensions of a netCDF variable and the in-memory structure of the internal data array. 
      The elements of the index mapping vector correspond, in order, to the netCDF variable's dimensions (imap[0] gives the distance between elements 
      of the internal array corresponding to the most slowly varying dimension of the netCDF variable). Distances between elements are 
      specified in type-independent units of elements (the distance between internal elements that occupy adjacent memory locations is 
      1 and not the element's byte-length as in netCDF 2). A NULL argument means the memory-resident values have the same structure as 
      the associated netCDF variable.

      \param dataValues The data values. The order in which the data will be written to the netCDF variable is with the last 
      dimension of the specified variable varying fastest. If the type of data values differs from the netCDF variable type, type conversion will occur. 
     (However, no type conversion is carried out for variables using the user-defined data types:  ncmpi_Vlen, ncmpi_Opaque, ncmpi_Compound and ncmpi_Enum.)
*/
    void bputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const long long* dataValues, int *req) const;

    void bputVar(const std::vector<MPI_Offset>& startp, const std::vector<MPI_Offset>& countp, const std::vector<MPI_Offset>& stridep, const std::vector<MPI_Offset>& imapp, const void* dataValues, MPI_Offset bufcount, MPI_Datatype buftype, int *req) const;

    void Inq_file_offset(MPI_Offset *offset);

  private:
      
    bool nullObject;

    int myId;	

    int groupId;
       
  };


}



#endif


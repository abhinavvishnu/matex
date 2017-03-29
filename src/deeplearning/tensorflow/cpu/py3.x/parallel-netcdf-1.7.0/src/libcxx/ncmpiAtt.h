#include "ncmpiType.h"
#include "ncmpiException.h"
#include <string>
#include <typeinfo>

#ifndef NcmpiAttClass
#define NcmpiAttClass

namespace PnetCDF
{

  /*! Abstract base class represents inherited by ncmpiVarAtt and ncmpiGroupAtt. */
  class NcmpiAtt
  {
  public:
    
    /*! destructor */
    virtual ~NcmpiAtt()=0;

    /*! Constructor generates a \ref isNull "null object". */
    NcmpiAtt ();
    
    /*! Constructor for non-null instances. */
    NcmpiAtt(bool nullObject); 

    /*! The copy constructor. */
    NcmpiAtt(const NcmpiAtt& rhs);

    /*! Get the attribute name. */
    std::string getName() const {return myName;}

    /*! Gets attribute length. */
    MPI_Offset  getAttLength() const;

    /*! Returns the attribute type. */
    NcmpiType  getType() const;

    /*! Gets parent group. */
    NcmpiGroup  getParentGroup() const;
      
    /*! equivalence operator */
    bool operator== (const NcmpiAtt& rhs) const;
      
    /*!  != operator */
    bool operator!=(const NcmpiAtt& rhs) const;     

    /*! \overload
     */ 
    void getValues(char* dataValues) const;
    /*! \overload
     */ 
    void getValues(unsigned char* dataValues) const;
    /*! \overload
     */ 
    void getValues(signed char* dataValues) const;
    /*! \overload
     */ 
    void getValues(short* dataValues) const;
    /*! \overload
     */ 
    void getValues(int* dataValues) const;
    /*! \overload
     */ 
    void getValues(long* dataValues) const;
    /*! \overload
     */ 
    void getValues(float* dataValues) const;
    /*! \overload
     */ 
    void getValues(double* dataValues) const;
    /*! \overload
     */ 
    void getValues(unsigned short* dataValues) const;
    /*! \overload
     */ 
    void getValues(unsigned int* dataValues) const;
    /*! \overload
     */ 
    void getValues(long long* dataValues) const;
    /*! \overload
     */ 
    void getValues(unsigned long long* dataValues) const;
    /*! \overload
     */ 
    void getValues(char** dataValues) const;

    /*! \overload
      (The string variable does not need preallocating.)
     */ 
    void getValues(std::string& dataValues) const;

    /*! 
      Gets a netCDF attribute.
      The user must ensure that the variable "dataValues" has sufficient space to hold the attribute.
      \param  dataValues On return contains the value of the attribute. 
      If the type of data values differs from the netCDF variable type, type conversion will occur. 
      (However, no type conversion is carried out for variables using the user-defined data types:
      ncmpi_Vlen, ncmpi_Opaque, ncmpi_Compound and ncmpi_Enum.)
    */
    void getValues(void* dataValues) const;

    /*! Returns true if this object is null (i.e. it has no contents); otherwise returns false. */
    bool isNull() const {return nullObject;}

  protected:
    /*! assignment operator */
    NcmpiAtt& operator= (const NcmpiAtt& rhs);
      
    bool nullObject;

    std::string myName;
    
    int groupId;
      
    int varId;
    
  };
  
}

#endif

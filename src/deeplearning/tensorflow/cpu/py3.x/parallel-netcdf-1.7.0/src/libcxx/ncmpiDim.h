#include <string>
#include <mpi.h>

#ifndef NcmpiDimClass
#define NcmpiDimClass


namespace PnetCDF
{
  class NcmpiGroup;  // forward declaration.
     
  /*! Class represents a netCDF dimension */
  class NcmpiDim   {
    
  public:
     
    /*! destructor*/
    ~NcmpiDim(){};

    /*! Constructor generates a \ref isNull "null object". */
    NcmpiDim ();

    /*! 
      Constructor for a dimension .
      The dimension must already exist in the netCDF file. New netCDF variables can be added using NcmpiGroup::addNcmpiDim();
      \param grp    Parent NcmpiGroup object.
      \param dimId  Id of the NcmpiDim object.
    */
    NcmpiDim(const NcmpiGroup& grp, int dimId);

    /*! assignment operator  */
    NcmpiDim& operator =(const NcmpiDim &);

    /*! equivalence operator */
    bool operator==(const NcmpiDim& rhs) const;     

    /*!  != operator */
    bool operator!=(const NcmpiDim& rhs) const;     

    /*! The copy constructor. */
    NcmpiDim(const NcmpiDim& ncmpiDim);

    /*! The name of this dimension.*/
    const std::string getName() const;

    /*! The netCDF Id of this dimension. */
    int getId() const {return myId;};
    
    /*! Gets a  NcmpiGroup object of the parent group. */
    NcmpiGroup getParentGroup() const;
      
    /*! Returns true if this is an unlimited dimension */
    bool isUnlimited() const;

    /*! The size of the dimension; for unlimited, this is the number of records written so far. */
    MPI_Offset  getSize() const;

    /*!renames the dimension */
    void rename( const std::string& newName);

    /*! Returns true if this object is null (i.e. it has no contents); otherwise returns false. */
    bool isNull() const  {return nullObject;}

    /*! comparator operator  */
    friend bool operator<(const NcmpiDim& lhs,const NcmpiDim& rhs);
    
    /*! comparator operator  */
    friend bool operator>(const NcmpiDim& lhs,const NcmpiDim& rhs);
    
  private:

    bool nullObject;

    int myId;	

    int groupId;

  };
  
}

 
#endif


#include "ncmpiAtt.h"

#ifndef NcmpiVarAttClass
#define NcmpiVarAttClass

namespace PnetCDF
{
  class NcmpiGroup;  // forward declaration.
  class NcmpiVar;    // forward declaration.

  /*! Class represents a netCDF attribute local to a netCDF variable. */
  class NcmpiVarAtt : public NcmpiAtt
  {
  public:
    
    /*! assignment operator */
    NcmpiVarAtt& operator= (const NcmpiVarAtt& rhs);
      
    /*! Constructor generates a \ref isNull "null object". */
    NcmpiVarAtt ();

    /*! The copy constructor. */
    NcmpiVarAtt(const NcmpiVarAtt& rhs) ;
      
    /*! 
      Constructor for an existing local attribute.
      \param  grp        Parent Group object.
      \param  NcmpiVar      Parent NcmpiVar object.
      \param  index      The index (id) of the attribute.
    */
    NcmpiVarAtt(const NcmpiGroup& grp, const NcmpiVar& ncmpiVar, const int index);
    
    /*! Returns the NcmpiVar parent object. */
    NcmpiVar getParentVar() const;

    /*! comparator operator */
    friend bool operator<(const NcmpiVarAtt& lhs,const NcmpiVarAtt& rhs);
    
    /*! comparator operator  */
    friend bool operator>(const NcmpiVarAtt& lhs,const NcmpiVarAtt& rhs);
    
  };
  
}

#endif

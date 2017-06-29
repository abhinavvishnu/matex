#include "ncmpiAtt.h"

#ifndef NcmpiGroupAttClass
#define NcmpiGroupAttClass

namespace PnetCDF
{
  class NcmpiGroup;  // forward declaration.

  /*! Class represents a netCDF group attribute */
  class NcmpiGroupAtt : public NcmpiAtt
  {
  public:
    
    /*! assignment operator */
    NcmpiGroupAtt& operator= (const NcmpiGroupAtt& rhs);
   
    /*! Constructor generates a \ref isNull "null object". */
    NcmpiGroupAtt ();
    
    /*! The copy constructor. */
    NcmpiGroupAtt(const NcmpiGroupAtt& rhs) ;
      
    /*! 
      Constructor for an existing global attribute.
      \param  grp        Parent Group object.
      \param  index      The index (id) of the attribute.
    */
    NcmpiGroupAtt(const NcmpiGroup& grp, const int index);
    
    /*! equivalence operator */
    bool operator== (const NcmpiGroupAtt& rhs);
      
    /*! comparator operator */
    friend bool operator<(const NcmpiGroupAtt& lhs,const NcmpiGroupAtt& rhs);
    
    /*! comparator operator */
    friend bool operator>(const NcmpiGroupAtt& lhs,const NcmpiGroupAtt& rhs);
    
  };
  
}

#endif

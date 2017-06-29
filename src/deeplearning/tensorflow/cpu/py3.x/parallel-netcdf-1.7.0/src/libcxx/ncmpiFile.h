#include <string>
#include "ncmpiGroup.h"

#ifndef NcmpiFileClass
#define NcmpiFileClass

//!  C++ API for PnetCDF.
namespace PnetCDF
{

  /*! 
    Class represents a netCDF root group.
    The Ncfile class is the same as the NcmpiGroup class with the additional
    functionality for opening and closing files.
   */
  class NcmpiFile : public NcmpiGroup
   {
   public:
    
      enum FileMode 
	 {
	    read,	//!< File exists, open read-only.
	    write,      //!< File exists, open for writing.
	    replace,	//!< Create new file, even if already exists.
	    newFile	//!< Create new file, fail if already exists.
	 };

      enum FileFormat
         {
	    classic,    //!< Classic format, classic data model
	    classic2,   //!< 64-bit offset format, classic data model
	    nc4,        //!< (default) netCDF-4/HDF5 format, enhanced data model
	    nc4classic, //!< netCDF-4/HDF5 format, classic data model
	    classic5,   //!< CDF-5 format, classic data model
            BadFormat
         };
    
         
      /*! Constructor generates a \ref isNull "null object". */
      NcmpiFile();

      /*! 
	Creates/opens a netCDF file.
	\param comm        MPI intra-communicator
	\param filePath    Name of netCDF optional path.
	\param fMode       The file mode:
	                    - 'read'    File exists, open for read-only.
	                    - 'write'   File exists, open for writing.
	                    - 'replace' Create new file, even it already exists.
	                    - 'newFile' Create new file, fail it exists already.
        \param info        MPI info object containing MPI and PnetCDF IO hints
      */
      NcmpiFile(const MPI_Comm    &comm,
                const std::string &filePath,
                FileMode           fMode,
                const MPI_Info    &info = MPI_INFO_NULL);

      /*! 
	Creates a netCDF file of a specified format.
	\param comm        MPI intra-communicator
	\param filePath    Name of netCDF optional path.
	\param fMode       The file mode:
	                    - 'replace' Create new file, even it already exists.
	                    - 'newFile' Create new file, fail it exists already.
        \param info        MPI info object containing MPI and PnetCDF IO hints
      */
      NcmpiFile(const MPI_Comm    &comm,
                const std::string &filePath,
                FileMode           fMode,
                FileFormat         fFormat,
                const MPI_Info    &info = MPI_INFO_NULL);
    
      /*! destructor */
      virtual ~NcmpiFile(); //closes file and releases all resources
     
      //! Synchronize an open netcdf dataset to disk
      void sync();

      //! Leave define mode, used for classic model
      void enddef();

      FileFormat getFormat( void ) const;

      void Wait_all(int num, int *array_of_requests, int *array_of_statuses);

      void Wait(int num, int *array_of_requests, int *array_of_statuses);

      void Cancel(int num, int *array_of_requests, int *array_of_statuses);

      void Buffer_attach(MPI_Offset bufsize);

      void Buffer_detach(void);

      void Inq_nreqs(int *nreqs);

      void Inq_buffer_usage(MPI_Offset *usage);

      void Inq_buffer_size(MPI_Offset *buf_size);

      void Inq_file_info(MPI_Info *info);

      void Inq_put_size(MPI_Offset *put_size);

      void Inq_get_size(MPI_Offset *get_size);

      void Inq_header_size(MPI_Offset *header_size);

      void Inq_header_extent(MPI_Offset *header_extent);

   private:
	/* Do not allow definition of NcmpiFile involving copying any NcmpiFile
           or NcmpiGroup.  Because the destructor closes the file and releases
           al resources such an action could leave NcmpiFile objects in an
           invalid state
         */
	NcmpiFile& operator =(const NcmpiGroup & rhs);
	NcmpiFile& operator =(const NcmpiFile & rhs);
	NcmpiFile(const NcmpiGroup& rhs);
	NcmpiFile(const NcmpiFile& rhs);
   };
  
}

#endif


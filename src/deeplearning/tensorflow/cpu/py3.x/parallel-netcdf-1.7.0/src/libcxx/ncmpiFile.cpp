#include "ncmpiFile.h"
#include "ncmpiCheck.h"
#include "ncmpiException.h"
#include "ncmpiByte.h"
#include<iostream>
#include<string>
#include<sstream>
using namespace std;
using namespace PnetCDF;
using namespace PnetCDF::exceptions;

// destructor
NcmpiFile::~NcmpiFile()
{
  // destructor may be called due to an exception being thrown
  // hence throwing an exception from within a destructor 
  // causes undefined behaviour! so just printing a warning message
  try
  {
    if (!nullObject)
      ncmpiCheck(ncmpi_close(myId),__FILE__,__LINE__);
  }
  catch (NcmpiException &e) 
  {
    cerr << e.what() << endl;
  }
}

// Constructor generates a null object.
NcmpiFile::NcmpiFile() : 
    NcmpiGroup()  // invoke base class constructor
{}

// constructor
NcmpiFile::NcmpiFile(const MPI_Comm  &comm,
                     const string    &filePath,
                     const FileMode   fMode,
                     const MPI_Info  &info /* = MPI_INFO_NULL */ )
{
  switch (fMode) 
    {
    case NcmpiFile::write:
      ncmpiCheck(ncmpi_open(comm, filePath.c_str(), NC_WRITE, info, &myId),__FILE__,__LINE__);
      break;
    case NcmpiFile::read:
      ncmpiCheck(ncmpi_open(comm, filePath.c_str(), NC_NOWRITE, info, &myId),__FILE__,__LINE__);
      break;
    case NcmpiFile::newFile:
      ncmpiCheck(ncmpi_create(comm, filePath.c_str(), NC_NOCLOBBER, info, &myId),__FILE__,__LINE__);
      break;
    case NcmpiFile::replace:
      ncmpiCheck(ncmpi_create(comm, filePath.c_str(), NC_CLOBBER, info, &myId),__FILE__,__LINE__);
      break;
    }
  nullObject=false;
}

// constructor with file type specified
NcmpiFile::NcmpiFile(const MPI_Comm   &comm,
                     const string     &filePath,
                     const FileMode    fMode,
                     const FileFormat  fFormat,
                     const MPI_Info   &info /* = MPI_INFO_NULL */)
{
  int format=0;
  switch (fFormat)
    {
    case NcmpiFile::classic:
	format = 0;
	break;
    case NcmpiFile::classic2:
	format = NC_64BIT_OFFSET;
	break;
    case NcmpiFile::nc4:
	format = NC_NETCDF4;
	break;
    case NcmpiFile::nc4classic:
	format = NC_NETCDF4 | NC_CLASSIC_MODEL;
	break;
    case NcmpiFile::classic5:
	format = NC_64BIT_DATA;
	break;
    case NcmpiFile::BadFormat:
        throw NcNotNCF("NetCDF: Unknown file format",__FILE__,__LINE__);
    }
  switch (fMode) 
    {
    case NcmpiFile::write:
      ncmpiCheck(ncmpi_open(comm, filePath.c_str(), format | NC_WRITE, info, &myId),__FILE__,__LINE__);
      break;
    case NcmpiFile::read:
      ncmpiCheck(ncmpi_open(comm, filePath.c_str(), format | NC_NOWRITE, info, &myId),__FILE__,__LINE__);
      break;
    case NcmpiFile::newFile:
      ncmpiCheck(ncmpi_create(comm, filePath.c_str(), format | NC_NOCLOBBER, info, &myId),__FILE__,__LINE__);
      break;
    case NcmpiFile::replace:
      ncmpiCheck(ncmpi_create(comm, filePath.c_str(), format | NC_CLOBBER, info, &myId),__FILE__,__LINE__);
      break;
    }
  nullObject=false;
}

// Synchronize an open netcdf dataset to disk
void NcmpiFile::sync(){
  ncmpiCheck(ncmpi_sync(myId),__FILE__,__LINE__);
}

// Leave define mode, used for classic model
void NcmpiFile::enddef() {
    ncmpiCheck(ncmpi_enddef(myId),__FILE__,__LINE__);
}

NcmpiFile::FileFormat NcmpiFile::getFormat( void ) const
{
    int the_format;
    ncmpiCheck(ncmpi_inq_format(myId, &the_format),__FILE__,__LINE__);

    switch (the_format) {
        case NC_FORMAT_CLASSIC:
            return classic;
        case NC_FORMAT_CDF2:
            return classic2;
        case NC_FORMAT_NETCDF4:
            return nc4;
        case NC_FORMAT_NETCDF4_CLASSIC:
            return nc4classic;
        case NC_FORMAT_CDF5:
            return classic5;
        default:
            return BadFormat;
    }
}

void NcmpiFile::Wait_all(int  num,
                         int *array_of_requests,
                         int *array_of_statuses)
{
    ncmpiCheck(ncmpi_wait_all(myId, num, array_of_requests, array_of_statuses),__FILE__,__LINE__);
}

void NcmpiFile::Wait(int  num,
                     int *array_of_requests,
                     int *array_of_statuses)
{
    ncmpiCheck(ncmpi_wait(myId, num, array_of_requests, array_of_statuses),__FILE__,__LINE__);
}

void NcmpiFile::Cancel(int  num,
                       int *array_of_requests,
                       int *array_of_statuses)
{
    ncmpiCheck(ncmpi_cancel(myId, num, array_of_requests, array_of_statuses),__FILE__,__LINE__);
}

void NcmpiFile::Buffer_attach(MPI_Offset bufsize)
{
    ncmpiCheck(ncmpi_buffer_attach(myId, bufsize),__FILE__,__LINE__);
}

void NcmpiFile::Buffer_detach(void)
{
    ncmpiCheck(ncmpi_buffer_detach(myId),__FILE__,__LINE__);
}

void NcmpiFile::Inq_buffer_usage(MPI_Offset *usage)
{
    ncmpiCheck(ncmpi_inq_buffer_usage(myId, usage),__FILE__,__LINE__);
}

void NcmpiFile::Inq_buffer_size(MPI_Offset *buf_size)
{
    ncmpiCheck(ncmpi_inq_buffer_size(myId, buf_size),__FILE__,__LINE__);
}

void NcmpiFile::Inq_nreqs(int *nreqs)
{
    ncmpiCheck(ncmpi_inq_nreqs(myId, nreqs),__FILE__,__LINE__);
}

void NcmpiFile::Inq_file_info(MPI_Info *info)
{
    ncmpiCheck(ncmpi_inq_file_info(myId, info),__FILE__,__LINE__);
}

void NcmpiFile::Inq_put_size(MPI_Offset *put_size)
{
    ncmpiCheck(ncmpi_inq_put_size(myId, put_size),__FILE__,__LINE__);
}

void NcmpiFile::Inq_get_size(MPI_Offset *get_size)
{
    ncmpiCheck(ncmpi_inq_get_size(myId, get_size),__FILE__,__LINE__);
}

void NcmpiFile::Inq_header_size(MPI_Offset *header_size)
{
    ncmpiCheck(ncmpi_inq_header_size(myId, header_size),__FILE__,__LINE__);
}

void NcmpiFile::Inq_header_extent(MPI_Offset *header_extent)
{
    ncmpiCheck(ncmpi_inq_header_extent(myId, header_extent),__FILE__,__LINE__);
}


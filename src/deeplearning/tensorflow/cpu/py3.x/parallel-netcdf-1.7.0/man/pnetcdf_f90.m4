.nr yr \n(yr+1900
.af mo 01
.af dy 01
.TH PNETCDF 3f90 "RELEASE_DATE" "Printed: \n(yr.\n(mo.\n(dy" "LIBRARY FUNCTIONS"
.SH NAME
PnetCDF \- Parallel library for accessing files in Network Common Data Form (CDF, CDF-2 and CDF-5 formats)
.SH SYNOPSIS
.ft B
.na
.nh
use pnetcdf
.sp
.SS Most Systems:
mpif90 ... -lpnetcdf
.sp
.SS CRAY PVP Systems:
f90 -dp -i64 ... -lpnetcdf

.ad
.hy
.SH "LIBRARY VERSION"
.LP
This document describes Parallel netCDF APIs
for the Fortran-90 programming language.
.HP
\fBcharacter*80 nf90mpi_inq_libvers(\|)
.RS
character(len=80) :: nf90mpi_inq_libvers\fR
.RE
.sp
Returns a string identifying the version of the PnetCDF library, and
when it was built, like: "RELEASE_STR".
.LP
The RCS \fBident(1)\fP command will find a string like
"$\|Id: @\|(#) PnetCDF library version
RELEASE_STR $"
in the library. The SCCS \fBwhat(1)\fP command will find a string like
"PnetCDF library version RELEASE_STR".
.SH "ROUTINE DESCRIPTIONS"
.LP
All PnetCDF functions (except
\fBnf90mpi_inq_libvers(\|)\fR and \fBnf90mpi_strerror(\|)\fR) return an integer 
status.
This behavior replaces the \fBrcode\fR argument
used in previous versions of the library.
If this returned status value is not equal to
\fBnf90_noerr\fR (zero), it
indicates that an error occurred. The possible status values are defined in 
the module \fBpnetcdf\fP.
.HP
\fBfunction nf90mpi_strerror(\fIncerr\fP)\fR
.RS
.nf
integer, intent(in) :: ncerr
character(len=80) :: nf90mpi_strerror
.fi
.sp
Returns a string textual translation of the \fIncerr\fP
value, like "Attribute or variable name contains illegal characters"
or "No such file or directory".
.RE
.HP
\fBfunction nf90mpi_create(\fIcomm\fP, \fIpath\fP, \fIcmode\fP, \fIinfo\fP, \fIncid\fP)\fR
.RS
.nf
integer, intent(in) :: comm
character(len=*), intent(in) :: path
integer, intent(in) :: cmode
integer, intent(in) :: info
integer, intent(out) :: ncid
integer :: nf90mpi_create
.fi
.sp
Creates a new netCDF dataset at \fIpath\fP collectively by a group of MPI
processes specified by \fIcomm\fP, returning a netCDF ID in \fIncid\fP.  The
argument \fIcmode\fP may include the bitwise-or of the following flags:
\fBnf90_noclobber\fR to protect existing datasets (default is \fBnf90_clobber\fR,
silently blows them away), \fBnf90_share\fR for stronger metadata data consistency
control, \fBnf90_64bit_offset\fR to create a file in the 64-bit offset format
(CDF-2), as opposed to classic format, the default, or \fBnf90_64bit_data\fR to
create a file in the 64-bit data format (CDF-5).
Use either \fBnf90_64bit_offset\fR or \fBnf90_64bit_data\fR.
The 64-bit offset format allows the creation of very large files with far fewer
restrictions than netCDF classic format, but can only be read by the netCDF
library version 3.6 or greater. Users are cautioned that files that use the
64-bit offset format will not be recognized by netCDF applications linked to an
earlier version of the netCDF library than 3.6.  Applications linked to version
3.6 or later will be able to transparently access either the classic format or
64-bit offset format.
The 64-bit data format allows the creation of very large array variables.
CDF-5 files currently will not be recognized by netCDF 3 or 4 library.
.

The argument \fIcmode\fP must be consistent among all MPI processes that
collectively create the file.  The argument \fIinfo\fP is an MPI info object.
Users can use it to supply the file access hints further performance
improvement.  The hints include existing MPI-IO hints as well as hints defined
and used in PnetCDF.
.sp
When a netCDF dataset is created, it is opened in \fbnf90_write\fR mode.
When this function returns, the new netCDF dataset is in define mode.
.RE
.HP
\fBfunction nf90mpi_open(\fIcomm\fP, \fIpath\fP, \fImode\fP, \fIinfo\fP, \fIncid\fP)\fR
.RS
.nf
integer, intent(in) :: comm
character(len=*), intent(in) :: path
integer, intent(in) :: mode
integer, intent(in) :: info
integer, intent(out) :: ncid
integer :: nf90mpi_open
.fi
.sp
Opens an existing netCDF dataset at \fIpath\fP collectively by a group of MPI
processes specified by \fIcomm\fP, returning a netCDF ID in \fIncid\fP.  The type
of access is described by the \fImode\fP parameter, which may include the
bitwise-or of the following flags: \fBnf90_write\fR for read-write access (default
read-only), \fBnf90_share\fR for stronger metadata data consistency control.
.sp

The argument \fImode\fP must be consistent among all MPI processes that
collectively open the file.  The argument \fIinfo\fP is an MPI info object.
Users can use it to supply the file access hints further performance
improvement.  The hints include existing MPI-IO hints as well as hints defined
and used in PnetCDF.
.RE
.HP
\fBfunction nf90mpi_redef(\fIncid\fP)\fR
.RS
.nf
integer, intent(in) :: ncid
integer :: nf90mpi_redef
.fi
.sp
Puts an open netCDF dataset into define mode, 
so dimensions, variables, and attributes can be added or renamed and 
attributes can be deleted.
.RE
.HP
\fBfunction nf90mpi_enddef(\fIncid\fP)\fR
.RS
.nf
integer, intent(in) :: ncid
integer :: nf90mpi_enddef
.fi
.sp
Takes an open netCDF dataset out of define mode.
The changes made to the netCDF dataset
while it was in define mode are checked and committed to disk if no
problems occurred.
After a successful call, variable data can be read or written to the dataset.
.RE
.HP
\fBfunction nf90mpi_sync(\fIncid\fP)\fR
.RS
.nf
integer, intent(in) :: ncid
integer :: nf90mpi_sync
.fi
.sp
Unless the
\fBnf90_share\fR
bit is set in
\fBnf90mpi_open(\|)\fR or \fBnf90mpi_create(\|)\fR,
data written by PnetCDF APIs may be cached by local file system on each compute
node.  This API flushes cached data by calling MPI_File_sync.
.RE
.HP
\fBfunction nf90mpi_abort(\fIncid\fP)\fR
.RS
.nf
integer, intent(in) :: ncid
integer :: nf90mpi_abort
.fi
.sp
You don't need to call this function.
This function is called automatically by
\fBnf90mpi_close(\|)\fR
if the netCDF dataset was in define mode and something 
goes wrong with the commit.
If the netCDF dataset isn't in define mode, then this function is equivalent to
\fBnf90mpi_close(\|)\fR.
If it is called after
\fBnf90mpi_redef(\|)\fR,
but before
\fBnf90mpi_enddef(\|)\fR,
the new definitions are not committed and the dataset is closed.
If it is called after
\fBnf90mpi_create(\|)\fR
but before
\fBnf90mpi_enddef(\|)\fR,
the dataset disappears.
.RE
.HP
\fBfunction nf90mpi_close(\fIncid\fP)\fR
.RS
.nf
integer, intent(in) :: ncid
integer :: nf90mpi_close
.fi
.sp
.sp
Closes an open netCDF dataset.
If the dataset is in define mode,
\fBnf90mpi_enddef(\|)\fR
will be called before closing.
After a dataset is closed, its ID may be reassigned to another dataset.
.RE
.HP
\fBfunction nf90mpi_inquire(\fIncid\fP, \fIndims\fP, \fInvars\fP,
\fInatts\fP, \fIunlimdimid\fP, \fInformat\fP)\fR
.RS
.nf
integer, intent(in) :: ncid
integer, optional, intent(out) :: ndims, nvars
integer, optional, intent(out) :: natts, unlimdimid
integer, optional, intent(out) :: nformat
integer :: nf90mpi_inquire
.fi
.sp
Inquire about an open netCDF dataset.
\fIncid\fP is the netCDF ID of the open dataset.
Upon successful return,
\fIndims\fP will contain  the
number of dimensions defined for this netCDF dataset,
\fInvars\fP will contain the number of variables,
\fInatts\fP will contain the number of attributes, and
\fIunlimdimid\fP will contain the
dimension ID of the unlimited dimension if one exists, or
0 otherwise.
\fInformat\fP will contain the format version number, rarely needed
because the library detects the format version and behaves
appropriately.
.RE
.HP
\fBfunction nf90mpi_def_dim(\fIncid\fP, \fIname\fP, \fIlen\fP, \fIdimid\fP)\fR
.RS
.nf
integer, intent(in) :: ncid
character(len=*), intent(in) :: name
integer, intent(in) :: len
integer, intent(out) :: dimid
integer :: nf90mpi_def_dim
.fi
.sp
Adds a new dimension to an open netCDF dataset, which must be 
in define mode.
\fIname\fP is the dimension name.
\fIlen\fP is the size of the new dimension or \fBnf90mpi_unlimited\fP to define
the unlimited dimension.
On return, \fIdimid\fP will contain the dimension ID of the newly created 
dimension.
.RE
.HP
\fBfunction nf90mpi_inq_dimid(\fIncid\fP, \fIname\fP, \fIdimid\fP)\fR
.RS
.nf
integer, intent(in) :: ncid
character(len=*), intent(in) :: name
integer, intent(out) :: dimid
integer :: nf90mpi_inq_dimid
.fi
.sp
Given an open netCDF dataset and dimension name, returns the dimension ID of the
netCDF dimension in \fIdimid\fP.
.RE
.HP
\fBfunction nf90mpi_inquire_dimension(\fIncid\fP, \fIdimid\fP, \fIname\fP, \fIlen\fP)\fR
.RS
.nf
integer, intent(in) :: ncid, dimid
character(len=*), optional, intent(out) :: name
integer, optional, intent(out) :: len
integer :: nf90mpi_inquire_dimension
.fi
.sp
Inquire about a dimension.
\fIname\fP should be  big enough (\fBnf90_max_name\fR)
to hold the dimension name as the name will be copied into your storage.
The length return parameter, \fIlen\fP
will contain the size of the dimension.
For the unlimited dimension, the returned length is the current
maximum value used for writing into any of the variables which use
the dimension.
.RE
.HP
\fBfunction nf90mpi_rename_dim(\fIncid\fP, \fIdimid\fP, \fIname\fP)\fR
.RS
.nf
integer, intent(in) :: ncid
character(len=*), intent(in) :: name
integer, intent(in) :: dimid
integer :: nf90mpi_rename_dim
.fi
.sp
Renames an existing dimension in an open netCDF dataset.
If the new name is longer than the old name, the netCDF dataset must be in 
define mode.
You cannot rename a dimension to have the same name as another dimension.
.RE
.HP
\fBfunction nf90mpi_def_var(\fIncid\fP, \fIname\fP, \fIxtype\fP, \fIdimids\fP, \fIvarid\fP)\fR
.RS
.nf
integer, intent(in) :: ncid
character(len=*), intent(in) :: name
integer, intent(in) :: xtype
integer, optional, dimension(:), intent(in) :: dimids
integer :: nf90mpi_def_var
.fi
.sp
Adds a new variable to a netCDF dataset. The netCDF must be in define mode.
\fIname\fP will be the name of the netCDF variable.
\fIxtype\fP is the external, netCDF type of the variable and should be one of
\fBnf90_byte\fP,
\fBnf90_char\fP,
\fBnf90_short\fP,
\fBnf90_int\fP,
\fBnf90_float\fP, or
\fBnf90_double\fP
for CDF-1 and CDF-2 file formats.
CDF-5 defines additional external types:
\fBnf90_ubyte\fP,
\fBnf90_ushort\fP,
\fBnf90_uint\fP,
\fBnf90_int64\fP, and
\fBnf90_uint64\fP.
The optional \fIdimids\fP argument contains the dimension ID-s of the domain
of the netCDF variable and, consequently, determines the rank of the
created variable:
if \fIdimids\fP is omitted, then the netCDF variable will be a scalar;
if \fIdimids\fP is a scalar, then the netCDF variable will be 1 dimensional;
and if \fIdimids\fP is a vector, then the netCDF variable will
have rank equal to the number of elements in \fIdimids\fP.
\fIvarid\fP will be set to the netCDF variable ID.
.RE
.HP
\fBfunction nf90mpi_inq_varid(\fIncid\fP, \fIname\fP, \fIvarid\fP)\fR
.RS
.nf
integer, intent(in) :: ncid
character(len=*), intent(in) :: name
integer, intent(out) :: varid
integer :: nf90mpi_inq_varid
.fi
.sp
Returns the ID of a netCDF variable in \fIvarid\fP given an open netCDF dataset
and the name of the variable.
.RE
.HP
\fBfunction nf90mpi_inquire_variable(\fIncid\fP, \fIvarid\fP, \fIname\fP, 
\fIxtype\fP, \fIndims\fP, \fIdimids\fP, \fInatts\fP)\fR
.RS
.nf
integer, intent(in) :: ncid, varid
character(len=*), optional, intent(out) :: name
integer, optional, intent(out) :: xtype, ndims
integer, dimension(*), optional, intent(out) :: dimids
integer, optional, intent(out) :: natts
integer :: nf90mpi_inquire_variable
.fi
.sp
Inquire about a netCDF variable in an open netCDF dataset, given its 
variable ID.
On return, \fIname\fP will contain the name of the variable and should 
be capacious enough (\fBnf90_max_name\fP).
\fIxtype\fP will contain the external, netCDF type of the variable.
\fIndims\fP will contain the dimensionality of the netCDF variable: if the
variable is a scalar, then size(\fIndims\fP) will be zero; otherwise,
size(\fIndims\fP) will be the rank of the variable and \fIndims\fP will contain
the dimension ID-s of the netCDF dimensions that constitute the domain of the
variable.
\fInatts\fP will contain the number of attributes associated with the netCDF
variable.
.RE
.HP
\fBfunction nf90mpi_rename_var(\fIncid\fP, \fIvarid\fP, \fIname\fP)\fR
.RS
.nf
integer, intent9in) :: ncid, varid
character(len=*), intent(in) :: newname
integer :: nf90mpi_rename_var
.fi
.sp
Changes the name of a netCDF variable.
If the new name is longer than the old name, the netCDF must be in define mode.
You cannot rename a variable to have the name of any existing variable.
.RE
.HP
\fBfunction nf90mpi_put_var(\fIncid\fP, \fIvarid\fP, \fIvalues\fP, 
\fIstart\fP, \fIstride\fP, \fIimap\fP)\fR
.RS
.nf
integer, intent(in) :: ncid, varid
<<whatever>>, intent(in) :: values
integer, dimension(:), optional, intent(in) :: start
integer, dimension(:), optional, intent(in) ::  stride
integer, dimension(:), optional, intent(in) ::  imap
integer :: nf90mpi_put_var
.fi
.sp
Writes a value or values to a netCDF variable.
The netCDF dataset must be open and in data mode.
\fIvalues\fP contains the value(s) what will be written to the netCDF variable
identified by \fIncid\fP and \fIvarid\fP; it may be a scalar or an array and
must be of type
\fBcharacter\fP,
\fBinteger(kind=OneByteInt)\fP,
\fBinteger(kind=TwoByteInt)\fP,
\fBinteger(kind=FourByteInt)\fP,
\fBinteger(kind=EightByteInt)\fP,
\fBreal(kind=FourByteReal)\fP, or
\fBreal(kind=EightByteReal)\fP.
All values are converted to the external type
of the netCDF variable, if possible; otherwise, an
\fBnf90_erange\fR error is returned.
The optional argument \fIstart\fP specifies
the starting index in the netCDF variable for writing for each
dimension of the netCDF variable.
The optional argument \fIstride\fP specifies the sampling stride
(the interval between accessed values in the netCDF variable)
for each dimension of the netCDF variable (see COMMON ARGUMENT DESCRIPTIONS
below).
The optional argument \fIimap\fP specifies the in-memory arrangement of the data
values (see COMMON ARGUMENT DESCRIPTIONS below).
.RE
.HP
\fBfunction nf90mpi_get_var(\fIncid\fP, \fIvarid\fP, \fIvalues\fP, 
\fIstart\fP, \fIstride\fP, \fIimap\fP)\fR
.RS
.nf
integer, intent(in) :: ncid, varid
<<whatever>>, intent(out) :: values
integer, dimension(:), optional, intent(in) :: start
integer, dimension(:), optional, intent(in) ::  stride
integer, dimension(:), optional, intent(in) ::  imap
integer :: nf90mpi_get_var
.fi
.sp
Reads a value or values from a netCDF variable.
The netCDF dataset must be open and in data mode.
\fIvalues\fP will receive the value(s) what will be read from the netCDF
 variable
identified by \fIncid\fP and \fIvarid\fP; it may be a scalar or an array and
must be of type
\fBcharacter\fP,
\fBinteger(kind=OneByteInt)\fP,
\fBinteger(kind=TwoByteInt)\fP,
\fBinteger(kind=FourByteInt)\fP,
\fBinteger(kind=EightByteInt)\fP,
\fBreal(kind=FourByteReal)\fP, or
\fBreal(kind=EightByteReal)\fP.
All values are converted from the external type
of the netCDF variable, if possible; otherwise, an
\fBnf90_erange\fR error is returned.
The optional argument \fIstart\fP specifies
the starting index in the netCDF variable for reading for each
dimension of the netCDF variable.
The optional argument \fIstride\fP specifies the sampling stride
(the interval between accessed values in the netCDF variable)
for each dimension of the netCDF variable (see COMMON ARGUMENT DESCRIPTIONS
below).
The optional argument \fIimap\fP specifies the in-memory arrangement of the data
values (see COMMON ARGUMENT DESCRIPTIONS below).
.RE
.HP
\fBfunction nf90mpi_inquire_attribute(\fIncid\fP, \fIvarid\fP, \fIname\fP,
\fIxtype\fP, \fIlen\fP, \fIattnum\fP)\fR
.RS
.nf
integer, intent(in) :: ncid, varid
character(len=*), intent(in) :: name
integer, optional, intent(out) :: xtype, len, attnum
integer :: nf90mpi_inquire_attribute
.fi
.sp
Inquires about the netCDF attribute named \fIname\fP, of variable \fIvarid\fP,
in the open netCDF dataset \fIncid\fP.
\fIxtype\fP will contain the external, netCDF type of the variable.
\fIlen\fP will contain the number of elements in the attribute.
\fIattnum\fP will contain the attribute number.
.RE
.HP
\fBfunction nf90mpi_inq_attname(\fIncid\fP, \fIvarid\fP, \fIattnum\fP, 
\fIname\fP)\fR
.RS
.nf
integer, intent(in) :: ncid, varid, attnum
character(len=*), intent(out) :: name
integer :: nf90mpi_inq_attname
.fi
.sp
Gets the
name of an attribute, given its variable ID and attribute number.
This function is useful in generic applications that
need to get the names of all the attributes associated with a variable
because attributes are accessed by name rather than number in all other
attribute functions (the number of an attribute is more volatile than
the name because it can change when other attributes of the same variable
are deleted).  The attributes for each variable are numbered
from 1 (the first attribute) to
\fInatts\fP, where \fInatts\fP is
the number of attributes for the variable, as returned from a call to
\fBnf90mpi_inquire_variable(\|)\fR.
.RE
.HP
\fBfunction nf90mpi_put_att(\fIncid\fP, \fIvarid\fP, \fIname\fP,
\fIvalues\fP)\fR
.RS
.nf
integer, intent(in) :: ncid, varid
character(len=*), intent(in) :: name
<<whatever>>, intent(in) :: values
integer :: nf90mpi_put_att
.fi
.sp
Unlike variables, attributes do not have 
separate functions for defining and writing values.
This function defines a new attribute with a value or changes
the value of an existing attribute.
If the attribute is new, or if the space required to
store the attribute value is greater than before,
the netCDF dataset must be in define mode.
\fIvalues\fP contains the attribute values to be written; it may be a scalar
or a vector and must be of type
\fBcharacter\fP,
\fBinteger(kind=OneByteInt)\fP,
\fBinteger(kind=TwoByteInt)\fP,
\fBinteger(kind=FourByteInt)\fP,
\fBinteger(kind=EightByteInt)\fP,
\fBreal(kind=FourByteReal)\fP, or
\fBreal(kind=EightByteReal)\fP.
.RE
.HP
\fBfunction nf90mpi_get_att(\fIncid\fP, \fIvarid\fP, \fIname\fP, \
fIvalues\fP)\fR
.RS
.nf
integer, intent(in) :: ncid, varid
character(len=*), intent(in) :: name
<<whatever>>, intent(out) :: values
integer :: nf90mpi_get_att
.fi
.sp
Gets the value(s) of a netCDF attribute, given its
variable ID and name.
The values are returned in \fIvalues\fP, which must be of type
\fBcharacter\fP,
\fBinteger(kind=OneByteInt)\fP,
\fBinteger(kind=TwoByteInt)\fP,
\fBinteger(kind=FourByteInt)\fP,
\fBinteger(kind=EightByteInt)\fP,
\fBreal(kind=FourByteReal)\fP, or
\fBreal(kind=EightByteReal)\fP.
Converts from the external type to the type
of the receiving variable, if possible; otherwise returns an \fBnf90_erange\fR
error.
All values of the attribute
are returned, so you must allocate enough space to hold
them.  If you don't know how much space to reserve, call
\fBnf90mpi_inquire_attribute(\|)\fR
first to find out the length of the attribute.
.RE
.HP
\fBfunction nf90mpi_copy_att(\fIncid_in\fP, \fIvarid_in\fP, \fIname\fP, 
\fIncid_out\fP, \fIvarid_out\fP)\fR
.RS
.nf
integer, intent(in) :: ncid_in, varid_in
character(len=*), intent(in) :: name
integer, intent(in) :: ncid_out, varid_out
integer :: nf90mpi_copy_att
.fi
.sp
Copies an
attribute from one netCDF dataset to another.  It can also be used to
copy an attribute from one variable to another within the same netCDF
dataset.
\fIncid_in\fP is the netCDF ID of an input netCDF dataset from which the
attribute will be copied.
\fIvarid_in\fP
is the ID of the variable in the input netCDF dataset from which the
attribute will be copied, or \fBnf90_global\fR
for a global attribute.
\fIname\fP
is the name of the attribute in the input netCDF dataset to be copied.
\fIncid_out\fP
is the netCDF ID of the output netCDF dataset to which the attribute will be 
copied.
It is permissible for the input and output netCDF ID's to be the same.  The
output netCDF dataset should be in define mode if the attribute to be
copied does not already exist for the target variable, or if it would
cause an existing target attribute to grow.
\fIvarid_out\fP
is the ID of the variable in the output netCDF dataset to which the 
attribute will
be copied, or \fBnf90_global\fR to copy to a global attribute.
.RE
.HP
\fBfunction nf90mpi_rename_att(\fIncid\fP, \fIvarid\fP, \fIname\fP, 
\fInewname\fP)\fR
.RS
.nf
integer, intent(in) :: ncid, varid
character(len=*), intent(in) :: name, newname
integer :: nf90mpi_rename_att
.fi
.sp
Changes the
name of an attribute.  If the new name is longer than the original name,
the netCDF must be in define mode.  You cannot rename an attribute to
have the same name as another attribute of the same variable.
\fIname\fP is the original attribute name.
\fInewname\fP
is the new name to be assigned to the specified attribute.  If the new name
is longer than the old name, the netCDF dataset must be in define mode.
.RE
.HP
\fBfunction nf90mpi_del_att(\fIncid\fP, \fIvarid\fP, \fIname\fP)\fR
.RS
.nf
integer, intent(in) :: ncid, varid
character(len=*), intent(in) :: name
integer :: nf90mpi_del_att
.fi
.sp
Deletes an attribute from a netCDF dataset.  The dataset must be in
define mode.
.RE
.SH "COMMON ARGUMENT DESCRIPTIONS"
.LP
In this section we define some common arguments which are used in the 
"FUNCTION DESCRIPTIONS" section.
.TP
integer \fIncid\fP
is the netCDF ID returned from a previous, successful call to
\fBnf90mpi_open(\|)\fR or \fBnf90mpi_create(\|)\fR
.TP
character(len=*) \fIname\fP
is the name of a dimension, variable, or attribute.
It shall begin with an alphabetic character, followed by
zero or more alphanumeric characters including the underscore
(`_') or hyphen (`-').  Case is significant.
The maximum allowable number of characters 
is \fBnf90_max_name\fR.
Names that begin with an underscore (`_') are reserved for use
by the netCDF and PnetCDF interfaces.
.TP
integer \fIxtype\fP
specifies the external data type of a netCDF variable or attribute and
is one of the following:
\fBnf90_byte\fR, \fBnf90_char\fR, \fBnf90_short\fR, \fBnf90_int\fR, 
\fBnf90_float\fR, or \fBnf90_double\fR.
These are used to specify 8-bit integers,
characters, 16-bit integers, 32-bit integers, 32-bit IEEE floating point
numbers, and 64-bit IEEE floating-point numbers, respectively.

.TP
integer \fIdimids\fP
is a vector of dimension ID's and defines the shape of a netCDF variable.
The size of the vector shall be greater than or equal to the
rank (i.e. the number of dimensions) of the variable (\fIndims\fP).
The vector shall be ordered by the speed with which a dimension varies:
\fIdimids\fP(\|1) shall be the dimension ID of the most rapidly
varying dimension and
\fIdimids\fP(\fIndims\fP)
shall be the dimension ID of the most slowly
varying dimension.
The maximum possible number of
dimensions for a variable is given by the symbolic constant
\fBnf90_max_var_dims\fR.
.TP
integer \fIdimid\fP
is the ID of a netCDF dimension.
netCDF dimension ID's are allocated sequentially from the 
positive
integers beginning with 1.
.TP
integer \fIndims\fP
is either the total number of dimensions in a netCDF dataset or the rank
(i.e. the number of dimensions) of a netCDF variable.
The value shall not be negative or greater than the symbolic constant 
\fBnf90_max_var_dims\fR.
.TP
integer \fIvarid\fP
is the ID of a netCDF variable or (for the attribute-access functions) 
the symbolic constant
\fBnf90_global\fR,
which is used to reference global attributes.
netCDF variable ID's are allocated sequentially from the 
positive
integers beginning with 1.
.TP
integer \fInatts\fP
is the number of global attributes in a netCDF dataset  for the
\fBnf90mpi_inquire(\|)\fR
function or the number
of attributes associated with a netCDF variable for the
\fBnf90mpi_varinq(\|)\fR
function.
.TP
integer(kind=MPI_OFFSET) \fIstart\fP
specifies the starting point
for accessing a netCDF variable's data values
in terms of the indicial coordinates of 
the corner of the array section.
The indices start at 1;
thus, the first data
value of a variable is (1, 1, ..., 1).
The size of the vector shall be at least the rank of the associated
netCDF variable and its elements shall correspond, in order, to the
variable's dimensions.
.TP
integer(kind=MPI_OFFSET) \fIstride\fP
specifies the sampling interval along each dimension of the netCDF
variable.   The elements of the stride vector correspond, in order,
to the netCDF variable's dimensions (\fIstride\fP(1))
gives the sampling interval along the most rapidly 
varying dimension of the netCDF variable).  Sampling intervals are
specified in type-independent units of elements (a value of 1 selects
consecutive elements of the netCDF variable along the corresponding
dimension, a value of 2 selects every other element, etc.).

.TP
integer(kind=MPI_OFFSET) \fIimap\fP
specifies the mapping between the dimensions of a netCDF variable and
the in-memory structure of the internal data array.  The elements of
the index mapping vector correspond, in order, to the netCDF variable's
dimensions (\fIimap\fP gives the distance
between elements of the internal array corresponding to the most
rapidly varying dimension of the netCDF variable).
Distances between elements are specified in type-independent units of
elements (the distance between internal elements that occupy adjacent
memory locations is 1 and not the element's byte-length as in netCDF 2).

.SH "VARIABLE PREFILLING"
.LP
PnetCDF does not support data filling.
.SH "ENVIRONMENT VARIABLES"
.TP 4
.B PNETCDF_SAFE_MODE
Set to 1 to enable metadata consistency check. Warning messages will
be printed to stdout if any inconsistency is detected.
.SH "MAILING-LISTS"
.LP
A mailing list is available for
discussion of the PnetCDF interface and announcements about PnetCDF bugs,
fixes, and enhancements.
To subscribe or unsubscribe to the PnetCDF mailing list,
visit https://lists.mcs.anl.gov/mailman/listinfo/parallel-netcdf
.RE
.SH "SEE ALSO"
.LP
.BR ncmpidump (1),
.BR ncmpigen (1),
.BR ncmpidiff (1),
.BR ncmpivalid (1),
.BR pnetcdf (3f90).
.LP
\fIPnetCDF User's Guide\fP, published
by Northwestern University and Argonne National Laboratory.
This document is adopted from the
\fInetCDF User's Guide\fP, developed at
the Unidata Program Center, University Corporation for Atmospheric
Research, located in Boulder, Colorado.

PnetCDF home page at http://cucis.ece.northwestern.edu/projects/PnetCDF/.

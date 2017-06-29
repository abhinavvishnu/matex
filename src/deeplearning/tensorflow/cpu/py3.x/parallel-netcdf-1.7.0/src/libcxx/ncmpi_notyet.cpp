#include <pnetcdf.h>
#include <iostream>
#include "ncmpi_notyet.h"

#ifndef __func__
#define __func__ __FUNCTION__
#endif

 int
ncmpi_insert_compound(int ncid, nc_type xtype, const char *name,
                   MPI_Offset offset, nc_type field_typeid){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

/* Insert a named array into a compound type. */
 int
ncmpi_insert_array_compound(int ncid, nc_type xtype, const char *name,
                         MPI_Offset offset, nc_type field_typeid,
                         int ndims, const int *dim_sizes){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}
 int
ncmpi_insert_enum(int ncid, nc_type xtype, const char *name,
               const void *value){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}
 int
ncmpi_def_compound(int ncid, MPI_Offset size, const char *name, nc_type *typeidp){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_inq_enum(int ncid, nc_type xtype, char *name, nc_type *base_nc_typep,
            MPI_Offset *base_sizep, MPI_Offset *num_membersp){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

/* Get information about an enum member: a name and value. Name size
 *  * will be <= NC_MAX_NAME. */
 int
ncmpi_inq_enum_member(int ncid, nc_type xtype, int idx, char *name,
                   void *value){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_inq_enum_ident(int ncid, nc_type xtype, long long value, char *identifier){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_inq_compound_nfields(int ncid, nc_type xtype, MPI_Offset *nfieldsp){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_inq_compound_name(int ncid, nc_type xtype, char *name){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

/* Get the size of a compound type. */
 int
ncmpi_inq_compound_size(int ncid, nc_type xtype, MPI_Offset *sizep){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

/* Given the xtype and the fieldid, get all info about it. */
 int
ncmpi_inq_compound_field(int ncid, nc_type xtype, int fieldid, char *name,
                      MPI_Offset *offsetp, nc_type *field_typeidp, int *ndimsp,
                      int *dim_sizesp){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

/* Given the typeid and the fieldid, get the name. */
 int
ncmpi_inq_compound_fieldname(int ncid, nc_type xtype, int fieldid,
                          char *name){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}
 int
ncmpi_inq_compound(int ncid, nc_type xtype, char *name, MPI_Offset *sizep,
                MPI_Offset *nfieldsp){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}
/* Given the xtype and the name, get the fieldid. */
 int
ncmpi_inq_compound_fieldindex(int ncid, nc_type xtype, const char *name,
                           int *fieldidp){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

/* Given the xtype and fieldid, get the offset. */
 int
ncmpi_inq_compound_fieldoffset(int ncid, nc_type xtype, int fieldid,
                            MPI_Offset *offsetp){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

/* Given the xtype and the fieldid, get the type of that field. */
 int
ncmpi_inq_compound_fieldtype(int ncid, nc_type xtype, int fieldid,
                          nc_type *field_typeidp){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

/* Given the xtype and the fieldid, get the number of dimensions for
 *  * that field (scalars are 0). */
 int
ncmpi_inq_compound_fieldndims(int ncid, nc_type xtype, int fieldid,
                           int *ndimsp){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}
 int
ncmpi_inq_compound_fielddim_sizes(int ncid, nc_type xtype, int fieldid,
                               int *dim_sizes){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_inq_unlimdims(int ncid, int *nunlimdimsp, int *unlimdimidsp){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_inq_grpname_len(int ncid, MPI_Offset *lenp){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

/* Given an ncid, find the ncid of its parent group. */
 int
ncmpi_inq_grp_parent(int ncid, int *parent_ncid){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

/* Given a name and parent ncid, find group ncid. */
 int
ncmpi_inq_grp_ncid(int ncid, const char *grp_name, int *grp_ncid){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}
/* Given a full name and ncid, find group ncid. */
 int
ncmpi_inq_grp_full_ncid(int ncid, const char *full_name, int *grp_ncid){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

/* Get a list of ids for all the variables in a group. */
 int
ncmpi_inq_varids(int ncid, int *nvars, int *varids){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

/* Find all dimids for a location. This finds all dimensions in a
 *  * group, or any of its parents. */
 int
ncmpi_inq_dimids(int ncid, int *ndims, int *dimids, int include_parents){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_inq_type(int ncid, nc_type xtype, char *name, MPI_Offset *size){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

/* Find all user-defined types for a location. This finds all
 *  * user-defined types in a group. */
 int
ncmpi_inq_typeids(int ncid, int *ntypes, int *typeids){*ntypes=0; std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

/* Are two types equal? */
 int
ncmpi_inq_type_equal(int ncid1, nc_type typeid1, int ncid2,
                  nc_type typeid2, int *equal){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}
/* Create a group. its ncid is returned in the new_ncid pointer. */
 int
ncmpi_def_grp(int parent_ncid, const char *name, int *new_ncid){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

/* Rename a group */
 int
ncmpi_rename_grp(int grpid, const char *name){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_inq_grps(int ncid, int *numgrps, int *ncids){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

/* Given locid, find name of group. (Root group is named "/".) */
 int
ncmpi_inq_grpname(int ncid, char *name){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

/* Given ncid, find full name and len of full name. (Root group is
 *  * named "/", with length 1.) */
 int
ncmpi_inq_grpname_full(int ncid, MPI_Offset *lenp, char *full_name){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_def_enum(int ncid, nc_type base_typeid, const char *name,
            nc_type *typeidp){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_def_vlen(int ncid, const char *name, nc_type base_typeid, nc_type *xtypep){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

/* Find out about a vlen. */
 int
ncmpi_inq_vlen(int ncid, nc_type xtype, char *name, MPI_Offset *datum_sizep,
            nc_type *base_nc_typep){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_free_vlen(nc_vlen_t *vl){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_free_vlens(MPI_Offset len, nc_vlen_t vlens[]){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

/* Put or get one element in a vlen array. */
 int
ncmpi_put_vlen_element(int ncid, int typeid1, void *vlen_element,
                    MPI_Offset len, const void *data){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_get_vlen_element(int ncid, int typeid1, const void *vlen_element,
                    MPI_Offset *len, void *data){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}
 int
ncmpi_def_opaque(int ncid, MPI_Offset size, const char *name, nc_type *xtypep){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

/* Get information about an opaque type. */
 int
ncmpi_inq_opaque(int ncid, nc_type xtype, char *name, MPI_Offset *sizep){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_inq_user_type(int ncid, nc_type xtype, char *name, MPI_Offset *size,
                 nc_type *base_nc_typep, MPI_Offset *nfieldsp, int *classp){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_def_var_chunking(int ncid, int varid, int storage, const MPI_Offset *chunksizesp){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}
 int
ncmpi_inq_var_chunking(int ncid, int varid, int *storagep, MPI_Offset *chunksizesp){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_def_var_endian(int ncid, int varid, int endian){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

/* Learn about the endianness of a variable. */
 int
ncmpi_inq_var_endian(int ncid, int varid, int *endianp){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}
 int
ncmpi_set_chunk_cache(MPI_Offset size, MPI_Offset nelems, float preemption){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_def_var_deflate(int ncid, int varid, int shuffle, int deflate,
                   int deflate_level){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}
 int
ncmpi_inq_var_deflate(int ncid, int varid, int *shufflep,
                   int *deflatep, int *deflate_levelp){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}
 int
ncmpi_inq_var_szip(int ncid, int varid, int *options_maskp, int *pixels_per_blockp){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

/* Set fletcher32 checksum for a var. This must be done after nc_def_var
 *    and before nc_enddef. */
 int
ncmpi_def_var_fletcher32(int ncid, int varid, int fletcher32){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

/* Inquire about fletcher32 checksum for a var. */
 int
ncmpi_inq_var_fletcher32(int ncid, int varid, int *fletcher32p){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_put_var1_string(int ncid, int varid, const MPI_Offset *indexp,
                   const char **op){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_get_var1_string(int ncid, int varid, const MPI_Offset *indexp,
                   char **ip){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_put_var_string(int ncid, int varid,
                   const char **op){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_get_var_string(int ncid, int varid,
                   char **ip){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_put_vara_string(int ncid, int varid, const MPI_Offset *startp,
                   const MPI_Offset *countp,
                   const char **op){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_get_vara_string(int ncid, int varid, const MPI_Offset *startp,
                   const MPI_Offset *countp,
                   char **ip){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_put_vars_string(int ncid, int varid, const MPI_Offset *startp,
                   const MPI_Offset *countp, const MPI_Offset *stridep,
                   const char **op){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_get_vars_string(int ncid, int varid, const MPI_Offset *startp,
                   const MPI_Offset *countp, const MPI_Offset *stridep,
                   char **ip){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_put_varm_string(int ncid, int varid, const MPI_Offset *startp,
                   const MPI_Offset *countp, const MPI_Offset *stridep,
                   const MPI_Offset * imapp, const char **op){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_get_varm_string(int ncid, int varid, const MPI_Offset *startp,
                   const MPI_Offset *countp, const MPI_Offset *stridep,
                   const MPI_Offset * imapp, char **ip){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_put_varn_string(int ncid, int varid, int num, MPI_Offset* const starts[],
                   MPI_Offset* const count[], const char **op){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_get_varn_string(int ncid, int varid, int num, MPI_Offset* const starts[],
                   MPI_Offset* const count[], char **ip){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_put_varn_string_all(int ncid, int varid, int num, MPI_Offset* const starts[],
                   MPI_Offset* const count[], const char **op){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_get_varn_string_all(int ncid, int varid, int num, MPI_Offset* const starts[],
                   MPI_Offset* const count[], char **ip){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_iput_var1_string(int ncid, int varid, const MPI_Offset *indexp, const char **op, int *req){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_bput_var1_string(int ncid, int varid, const MPI_Offset *indexp, const char **op, int *req){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_iget_var1_string(int ncid, int varid, const MPI_Offset *indexp, char **ip, int *req){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_iput_var_string(int ncid, int varid, const char **op, int *req){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_bput_var_string(int ncid, int varid, const char **op, int *req){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_iget_var_string(int ncid, int varid, char **ip, int *req){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_iput_vara_string(int ncid, int varid, const MPI_Offset *startp, const MPI_Offset *countp, const char **op, int *req){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_bput_vara_string(int ncid, int varid, const MPI_Offset *startp, const MPI_Offset *countp, const char **op, int *req){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_iget_vara_string(int ncid, int varid, const MPI_Offset *startp, const MPI_Offset *countp, char **ip, int *req){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_iput_vars_string(int ncid, int varid, const MPI_Offset *startp, const MPI_Offset *countp, const MPI_Offset *stridep, const char **op, int *req){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_bput_vars_string(int ncid, int varid, const MPI_Offset *startp, const MPI_Offset *countp, const MPI_Offset *stridep, const char **op, int *req){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_iget_vars_string(int ncid, int varid, const MPI_Offset *startp, const MPI_Offset *countp, const MPI_Offset *stridep, char **ip, int *req){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_iput_varm_string(int ncid, int varid, const MPI_Offset *startp, const MPI_Offset *countp, const MPI_Offset *stridep, const MPI_Offset * imapp, const char **op, int *req){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_bput_varm_string(int ncid, int varid, const MPI_Offset *startp, const MPI_Offset *countp, const MPI_Offset *stridep, const MPI_Offset * imapp, const char **op, int *req){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}

 int
ncmpi_iget_varm_string(int ncid, int varid, const MPI_Offset *startp, const MPI_Offset *countp, const MPI_Offset *stridep, const MPI_Offset * imapp, char **ip, int *req){std::cout << __func__ << " not implemented" << std::endl; return NC_EINVAL;}



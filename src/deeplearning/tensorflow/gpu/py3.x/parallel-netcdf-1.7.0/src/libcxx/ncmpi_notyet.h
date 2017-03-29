#ifndef H_NC
#define H_NC
extern int
ncmpi_insert_compound(int ncid, nc_type xtype, const char *name,
                   MPI_Offset offset, nc_type field_typeid);

/* Insert a named array into a compound type. */
extern int
ncmpi_insert_array_compound(int ncid, nc_type xtype, const char *name,
                         MPI_Offset offset, nc_type field_typeid,
                         int ndims, const int *dim_sizes);
extern int
ncmpi_insert_enum(int ncid, nc_type xtype, const char *name,
               const void *value);
extern int
ncmpi_def_compound(int ncid, MPI_Offset size, const char *name, nc_type *typeidp);

extern int
ncmpi_inq_enum(int ncid, nc_type xtype, char *name, nc_type *base_nc_typep,
            MPI_Offset *base_sizep, MPI_Offset *num_membersp);

/* Get information about an enum member: a name and value. Name size
 *  * will be <= NC_MAX_NAME. */
extern int
ncmpi_inq_enum_member(int ncid, nc_type xtype, int idx, char *name,
                   void *value);

extern int
ncmpi_inq_enum_ident(int ncid, nc_type xtype, long long value, char *identifier);

extern int
ncmpi_inq_compound_nfields(int ncid, nc_type xtype, MPI_Offset *nfieldsp);

extern int
ncmpi_inq_compound_name(int ncid, nc_type xtype, char *name);

/* Get the size of a compound type. */
extern int
ncmpi_inq_compound_size(int ncid, nc_type xtype, MPI_Offset *sizep);

/* Get the number of fields in this compound type. */
extern int
ncmpi_inq_compound_nfields(int ncid, nc_type xtype, MPI_Offset *nfieldsp);

/* Given the xtype and the fieldid, get all info about it. */
extern int
ncmpi_inq_compound_field(int ncid, nc_type xtype, int fieldid, char *name,
                      MPI_Offset *offsetp, nc_type *field_typeidp, int *ndimsp,
                      int *dim_sizesp);

/* Given the typeid and the fieldid, get the name. */
extern int
ncmpi_inq_compound_fieldname(int ncid, nc_type xtype, int fieldid,
                          char *name);
extern int
ncmpi_inq_compound(int ncid, nc_type xtype, char *name, MPI_Offset *sizep,
                MPI_Offset *nfieldsp);
/* Given the xtype and the name, get the fieldid. */
extern int
ncmpi_inq_compound_fieldindex(int ncid, nc_type xtype, const char *name,
                           int *fieldidp);

/* Given the xtype and fieldid, get the offset. */
extern int
ncmpi_inq_compound_fieldoffset(int ncid, nc_type xtype, int fieldid,
                            MPI_Offset *offsetp);

/* Given the xtype and the fieldid, get the type of that field. */
extern int
ncmpi_inq_compound_fieldtype(int ncid, nc_type xtype, int fieldid,
                          nc_type *field_typeidp);

/* Given the xtype and the fieldid, get the number of dimensions for
 *  * that field (scalars are 0). */
extern int
ncmpi_inq_compound_fieldndims(int ncid, nc_type xtype, int fieldid,
                           int *ndimsp);
extern int
ncmpi_inq_compound_fielddim_sizes(int ncid, nc_type xtype, int fieldid,
                               int *dim_sizes);

extern int
ncmpi_inq_unlimdims(int ncid, int *nunlimdimsp, int *unlimdimidsp);

extern int
ncmpi_inq_grpname_len(int ncid, MPI_Offset *lenp);

/* Given an ncid, find the ncid of its parent group. */
extern int
ncmpi_inq_grp_parent(int ncid, int *parent_ncid);

/* Given a name and parent ncid, find group ncid. */
extern int
ncmpi_inq_grp_ncid(int ncid, const char *grp_name, int *grp_ncid);
/* Given a full name and ncid, find group ncid. */
extern int
ncmpi_inq_grp_full_ncid(int ncid, const char *full_name, int *grp_ncid);

/* Get a list of ids for all the variables in a group. */
extern int
ncmpi_inq_varids(int ncid, int *nvars, int *varids);

/* Find all dimids for a location. This finds all dimensions in a
 *  * group, or any of its parents. */
extern int
ncmpi_inq_dimids(int ncid, int *ndims, int *dimids, int include_parents);

extern int
ncmpi_inq_type(int ncid, nc_type xtype, char *name, MPI_Offset *size);

/* Find all user-defined types for a location. This finds all
 *  * user-defined types in a group. */
extern int
ncmpi_inq_typeids(int ncid, int *ntypes, int *typeids);

/* Are two types equal? */
extern int
ncmpi_inq_type_equal(int ncid1, nc_type typeid1, int ncid2,
                  nc_type typeid2, int *equal);
/* Create a group. its ncid is returned in the new_ncid pointer. */
extern int
ncmpi_def_grp(int parent_ncid, const char *name, int *new_ncid);

/* Rename a group */
extern int
ncmpi_rename_grp(int grpid, const char *name);

extern int
ncmpi_inq_grps(int ncid, int *numgrps, int *ncids);

/* Given locid, find name of group. (Root group is named "/".) */
extern int
ncmpi_inq_grpname(int ncid, char *name);

/* Given ncid, find full name and len of full name. (Root group is
 *  * named "/", with length 1.) */
extern int
ncmpi_inq_grpname_full(int ncid, MPI_Offset *lenp, char *full_name);

extern int
ncmpi_def_enum(int ncid, nc_type base_typeid, const char *name,
            nc_type *typeidp);

extern int
ncmpi_def_vlen(int ncid, const char *name, nc_type base_typeid, nc_type *xtypep);

/* Find out about a vlen. */
extern int
ncmpi_inq_vlen(int ncid, nc_type xtype, char *name, MPI_Offset *datum_sizep,
            nc_type *base_nc_typep);

typedef struct {
    MPI_Offset len; /**< Length of VL data (in base type units) */
    void *p;    /**< Pointer to VL data */
} nc_vlen_t;

extern int
ncmpi_free_vlen(nc_vlen_t *vl);

extern int
ncmpi_free_vlens(MPI_Offset len, nc_vlen_t vlens[]);

/* Put or get one element in a vlen array. */
extern int
ncmpi_put_vlen_element(int ncid, int typeid1, void *vlen_element,
                    MPI_Offset len, const void *data);

extern int
ncmpi_get_vlen_element(int ncid, int typeid1, const void *vlen_element,
                    MPI_Offset *len, void *data);
extern int
ncmpi_def_opaque(int ncid, MPI_Offset size, const char *name, nc_type *xtypep);

/* Get information about an opaque type. */
extern int
ncmpi_inq_opaque(int ncid, nc_type xtype, char *name, MPI_Offset *sizep);

extern int
ncmpi_inq_user_type(int ncid, nc_type xtype, char *name, MPI_Offset *size,
                 nc_type *base_nc_typep, MPI_Offset *nfieldsp, int *classp);

extern int
ncmpi_def_var_chunking(int ncid, int varid, int storage, const MPI_Offset *chunksizesp);
extern int
ncmpi_inq_var_chunking(int ncid, int varid, int *storagep, MPI_Offset *chunksizesp);

extern int
ncmpi_def_var_endian(int ncid, int varid, int endian);

/* Learn about the endianness of a variable. */
extern int
ncmpi_inq_var_endian(int ncid, int varid, int *endianp);
extern int
ncmpi_set_chunk_cache(MPI_Offset size, MPI_Offset nelems, float preemption);

extern int
ncmpi_def_var_deflate(int ncid, int varid, int shuffle, int deflate,
                   int deflate_level);
extern int
ncmpi_inq_var_deflate(int ncid, int varid, int *shufflep,
                   int *deflatep, int *deflate_levelp);
extern int
ncmpi_inq_var_szip(int ncid, int varid, int *options_maskp, int *pixels_per_blockp);

/* Set fletcher32 checksum for a var. This must be done after nc_def_var
 *    and before nc_enddef. */
extern int
ncmpi_def_var_fletcher32(int ncid, int varid, int fletcher32);

/* Inquire about fletcher32 checksum for a var. */
extern int
ncmpi_inq_var_fletcher32(int ncid, int varid, int *fletcher32p);

#endif

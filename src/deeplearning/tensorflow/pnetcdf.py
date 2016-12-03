import sys
import numpy as np
from mpi4py import MPI
import ctypes
from ctypes import c_byte
from ctypes import c_int
from ctypes import c_char_p
from ctypes import c_void_p
from ctypes import byref

NC_NOWRITE = 0
NC_NOERR = 0

NC_NAT   =  0   
NC_BYTE  =  1   
NC_CHAR  =  2   
NC_SHORT =  3   
NC_INT   =  4   
NC_LONG  =  NC_INT  
NC_FLOAT =  5   
NC_DOUBLE=  6   
NC_UBYTE =  7   
NC_USHORT=  8   
NC_UINT  =  9   
NC_INT64 =  10  
NC_UINT64=  11  
NC_STRING=  12  


if MPI._sizeof(MPI.Comm) == ctypes.sizeof(ctypes.c_int):
    MPI_Comm = ctypes.c_int
    MPI_Info = ctypes.c_int
else:
    MPI_Comm = ctypes.c_void_p
    MPI_Info = ctypes.c_void_p

# some MPI implementations don't have MPI_OFFSET as a datatype
try:
    if MPI.OFFSET.size == ctypes.sizeof(ctypes.c_int):
        MPI_Offset = ctypes.c_int
    elif MPI.OFFSET.size == ctypes.sizeof(ctypes.c_long):
        MPI_Offset = ctypes.c_long
    elif MPI.OFFSET.size == ctypes.sizeof(ctypes.c_longlong):
        MPI_Offset = ctypes.c_long
    else:
        assert False
except:
    MPI_Offset = ctypes.c_longlong

c_byte_p = ctypes.POINTER(c_byte)
c_int_p = ctypes.POINTER(c_int)
MPI_Offset_p = ctypes.POINTER(MPI_Offset)

_lib = ctypes.CDLL('libpnetcdf.so')

_ncmpi_strerror = _lib.ncmpi_strerror
_ncmpi_strerror.argtypes = [c_int]
_ncmpi_strerror.restype = c_char_p

def ncmpi_strerror(err):
    return _ncmpi_strerror(err)

def errcheck(retval):
    if retval != NC_NOERR:
        print "ERROR:",retval
        print ncmpi_strerror(retval)
    assert retval == NC_NOERR

_ncmpi_open = _lib.ncmpi_open
_ncmpi_open.argtypes = [MPI_Comm, c_char_p, c_int, MPI_Info, c_int_p]
_ncmpi_open.restype = c_int

def ncmpi_open(name):
    comm_ptr = MPI._addressof(MPI.COMM_WORLD)
    comm_val = MPI_Comm.from_address(comm_ptr)
    info_ptr = MPI._addressof(MPI.INFO_NULL)
    info_val = MPI_Comm.from_address(info_ptr)
    ncid = c_int()
    retval = _ncmpi_open(comm_val, name, NC_NOWRITE, info_val, byref(ncid))
    errcheck(retval)
    return ncid.value

_ncmpi_close = _lib.ncmpi_close
_ncmpi_close.argtypes = [c_int]
_ncmpi_close.restype = c_int

def ncmpi_close(ncid):
    retval = _ncmpi_close(ncid)
    errcheck(retval)

_ncmpi_inq = _lib.ncmpi_inq
_ncmpi_inq.argtypes = [c_int, c_int_p, c_int_p, c_int_p, c_int_p]
_ncmpi_inq.restype = c_int

def ncmpi_inq(ncid):
    ndims = c_int()
    nvars = c_int()
    ngatts = c_int()
    unlimdimi = c_int()
    retval = _ncmpi_inq(ncid, byref(ndims), byref(nvars), byref(ngatts), byref(unlimdimi))
    errcheck(retval)
    return ndims.value,nvars.value,ngatts.value,unlimdimi.value

_ncmpi_inq_dimlen = _lib.ncmpi_inq_dimlen
_ncmpi_inq_dimlen.argtypes = [c_int, c_int, MPI_Offset_p]
_ncmpi_inq_dimlen.restype = c_int

def ncmpi_inq_dimlen(ncid, i):
    size = MPI_Offset()
    retval = _ncmpi_inq_dimlen(ncid, i, byref(size))
    errcheck(retval)
    return size.value

_ncmpi_inq_vartype = _lib.ncmpi_inq_vartype
_ncmpi_inq_vartype.argtypes = [c_int, c_int, c_int_p]
_ncmpi_inq_vartype.restype = c_int

def ncmpi_inq_vartype(ncid, i):
    vartype = c_int()
    retval = _ncmpi_inq_vartype(ncid, i, byref(vartype))
    errcheck(retval)
    return vartype.value

_ncmpi_inq_varndims = _lib.ncmpi_inq_varndims
_ncmpi_inq_varndims.argtypes = [c_int, c_int, c_int_p]
_ncmpi_inq_varndims.restype = c_int

def ncmpi_inq_varndims(ncid, i):
    varndims = c_int()
    retval = _ncmpi_inq_varndims(ncid, i, byref(varndims))
    errcheck(retval)
    return varndims.value

_ncmpi_inq_vardimid = _lib.ncmpi_inq_vardimid
_ncmpi_inq_vardimid.argtypes = [c_int, c_int, c_int_p]
_ncmpi_inq_vardimid.restype = c_int

def ncmpi_inq_vardimid(ncid, i):
    varndims = ncmpi_inq_varndims(ncid, i)
    vardimids = (varndims*c_int)()
    retval = _ncmpi_inq_vardimid(ncid, i, vardimids)
    errcheck(retval)
    return tuple(vardimids)

_ncmpi_get_vara_schar_all = _lib.ncmpi_get_vara_schar_all
_ncmpi_get_vara_schar_all.argtypes = [c_int, c_int, MPI_Offset_p, MPI_Offset_p, c_byte_p]
_ncmpi_get_vara_schar_all.restype = c_int

def ncmpi_get_vara_schar_all(ncid, varid, start, count):
    data = np.empty(count, dtype=np.int8)
    assert len(start) == len(count)
    start_c = (len(start)*MPI_Offset)()
    count_c = (len(start)*MPI_Offset)()
    for i in range(len(start)):
        start_c[i] = start[i]
        count_c[i] = count[i]
    retval = _ncmpi_get_vara_schar_all(ncid, varid, start_c, count_c,
            data.ctypes.data_as(c_byte_p))
    errcheck(retval)
    return data

_ncmpi_get_vara_int_all = _lib.ncmpi_get_vara_int_all
_ncmpi_get_vara_int_all.argtypes = [c_int, c_int, MPI_Offset_p, MPI_Offset_p, c_int_p]
_ncmpi_get_vara_int_all.restype = c_int

def ncmpi_get_vara_int_all(ncid, varid, start, count):
    data = np.empty(count, dtype=np.int32)
    assert len(start) == len(count)
    start_c = (len(start)*MPI_Offset)()
    count_c = (len(start)*MPI_Offset)()
    for i in range(len(start)):
        start_c[i] = start[i]
        count_c[i] = count[i]
    retval = _ncmpi_get_vara_int_all(ncid, varid, start_c, count_c,
            data.ctypes.data_as(c_int_p))
    errcheck(retval)
    return data


def read_pnetcdf(filename):
    import sys
    from mpi4py import MPI

    DEBUG = False

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    ncid = ncmpi_open(filename)
    ndims,nvars,ngatts,unlimdim = ncmpi_inq(ncid)

    if DEBUG and 0 == rank:
        print "ncid",ncid
        print "ndims",ndims
        print "nvars",nvars
        print "ngatts",ngatts
        print "unlimdim",unlimdim

    total = ncmpi_inq_dimlen(ncid, unlimdim)
    count = total / size
    remain = total % size

    if 0 == rank:
        print "total images", total
        print "image subset (%d,%d)=%d" % (total,size,count)
        print "image subset remainder", remain

    start = int(rank * count)
    stop = int(rank * count + count)
    if rank < remain:
        start += rank
        stop += rank + 1
    else :
        start += remain
        stop += remain

    # read the two vars, either labels or data
    data = None
    labels = None
    for varid in range(nvars):
        vartype = ncmpi_inq_vartype(ncid, varid)
        vardimids = ncmpi_inq_vardimid(ncid, varid)
        count = [ncmpi_inq_dimlen(ncid,dimid) for dimid in vardimids]
        offset = [0 for c in count]
        # MPI-IO, and transitively pnetcdf, can only read 2GB chunks due
        # to the "int" interface for indices
        chunksize = 2147483647L
        count[0] = stop-start
        offset[0] = start
        prodcount = np.prod(count)
        if DEBUG:
            for worker in range(size):
                if worker == rank:
                    print "worker",worker
                    print "vartype",vartype
                    print "vardimids",vardimids
                    print "count",count
                    print "offset",offset
                    print "prodcount",prodcount
                    sys.stdout.flush()
                comm.Barrier()
        if vartype == NC_BYTE:
            if prodcount < chunksize:
                if 0 == rank:
                    print "reading data whole",count[0]
                    sys.stdout.flush()
                data = ncmpi_get_vara_schar_all(ncid, varid, offset, count)
            else:
                data = np.empty(count, dtype=np.int8)
                newoffset = offset[:]
                newcount = count[:]
                newcount[0] = 1
                newprodcount = np.prod(newcount)
                newcount[0] = chunksize/newprodcount
                newprodcount = np.prod(newcount)
                assert newprodcount < chunksize
                cur = 0
                while cur < count[0]:
                    if cur+newcount[0] > count[0]:
                        newcount[0] = count[0]-cur
                    if 0 == rank:
                        print "reading data chunk",cur,"...",cur+newcount[0]
                        sys.stdout.flush()
                    data[cur:cur+newcount[0]] = ncmpi_get_vara_schar_all(
                        ncid, varid, newoffset, newcount)
                    cur += newcount[0]
                    newoffset[0] += newcount[0]
            data = data.view(np.uint8)
            if DEBUG:
                for worker in range(size):
                    if worker == rank:
                        print data
                        sys.stdout.flush()
                    comm.Barrier()
        elif vartype == NC_INT:
            if prodcount < chunksize:
                if 0 == rank:
                    print "reading label whole",count[0]
                    sys.stdout.flush()
                labels = ncmpi_get_vara_int_all(ncid, varid, offset, count)
            else:
                labels = np.empty(count, dtype=np.int8)
                newoffset = offset[:]
                newcount = count[:]
                newcount[0] = 1
                newprodcount = np.prod(newcount)
                newcount[0] = chunksize/newprodcount
                newprodcount = np.prod(newcount)
                assert newprodcount < chunksize
                cur = 0
                while cur < count[0]:
                    if cur+newcount[0] > count[0]:
                        newcount[0] = count[0]-cur
                    if 0 == rank:
                        print "reading label chunk",cur,"...",cur+newcount[0]
                        sys.stdout.flush()
                    labels[cur:cur+newcount[0]] = ncmpi_get_vara_int_all(
                        ncid, varid, newoffset, newcount)
                    cur += newcount[0]
                    newoffset[0] += newcount[0]
            if DEBUG:
                for worker in range(size):
                    if worker == rank:
                        print labels
                        sys.stdout.flush()
                    comm.Barrier()
        else:
            assert False
    ncmpi_close(ncid)
    return data,labels

if __name__ == '__main__':
    read_pnetcdf(sys.argv[1])

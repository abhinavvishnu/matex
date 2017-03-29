#!/bin/bash
# a test to ensure we can work with files created by serial netcdf.  does
# require both ncgen and ncdump to be in your path.

# Step 0: see if we have the utilities in our path:

which ncgen >/dev/null 

if [ $? -ne 0 ] ; then
	echo "could not find 'ncgen' (from serial netcdf) in path. exiting"
	exit 1
fi

which ncmpidump >/dev/null

if [ $? -ne 0 ] ; then
	echo "could not find 'ncmpidump' (from parallel-netcdf) in path. exiting"
	exit 1
fi

OUTPUT=geo-${RANDOM}-${RANDOM}.nc
rm -f $OUTPUT

# Step 1: create the file:
ncgen -b -v 2 -o $OUTPUT geopotential.ncdump

# step 2: ensure we can at least parse the header
ncmpidump -h $OUTPUT >/dev/null

if [ $? -ne 0 ] ; then
	echo "error parsing generated netcdf file!"
	exit 1
else
	echo " No Errors"
	rm $OUTPUT
	exit 0
fi

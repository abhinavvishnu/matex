#!/bin/bash
# quick test of re-entering define mode: the file created by ncmpigen and the
# file created by test program should be bit-for-bit identical, else there is
# an error

if [ $# -ne 2 ] ; then
	echo "usage: $0 <executable> <reference>"
	echo "example: $0 redef1 redef-good.ncdump"
	exit -1
fi

REFERENCE=redef1-a-${RANDOM}-${RANDOM}-${RANDOM}.nc

# dataset via ncmpigen:
../../src/utils/ncmpigen/ncmpigen -v 2 -o $REFERENCE $2

# dataset via test
$1

# now compare

diff $REFERENCE redef1.nc

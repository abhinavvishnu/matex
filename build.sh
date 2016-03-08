#!/bin/sh

# where this script is currently running
BUILDPATH=`pwd`

# where this script is located
cd `dirname $0` || exit 1
SCRIPTPATH=`pwd`
cd $BUILDPATH || exit 1

# user may have provided an install path
if test -z "$1"
then
    INSTALLPATH=$BUILDPATH
else
    if test -d "$1"
    then
        cd $1 || exit 1
        INSTALLPATH=`pwd`
        cd $BUILDPATH || exit 1
    else
        echo "invalid install path '$1'"
        exit 1
    fi
fi

# how many CPUs available for the build?
NPROC=`grep -c processor /proc/cpuinfo`
if test $NPROC -eq 0
then
    NPROC=1
fi

echo ""
echo " Script directory: $SCRIPTPATH"
echo "  Build directory: $BUILDPATH"
echo "Install directory: $INSTALLPATH"
echo "   parallel build: $NPROC"
echo ""

MPICH="mpich-3.1"
MPICH_BUILD="$MPICH-build"
MATEX=matex
MATEX_BUILD="$MATEX-build"

echo "==========================================================="
echo "checking for MPI"
echo "==========================================================="
matex_mpicc="mpicc mpixlc_r mpixlc hcc mpxlc_r mpxlc sxmpicc mpifcc mpgcc mpcc cmpicc"
#matex_mpicc=""
MPICC=""
for mpicc in $matex_mpicc
do
    save_IFS="$IFS"
    IFS=:
    for dir in $PATH
    do
        IFS="$save_IFS"
        test -z "$dir" && dir=.
        prog="$dir/$mpicc"
        if test -f "$prog" && test -x "$prog"
        then
            MPICC="$prog"
            break
        fi
    done
    IFS="$save_IFS"
    test -z "$MPICC" || break
done
if test -z "$MPICC"
then
    echo "not found"
else
    echo "$MPICC"
fi
echo ""

if test -z "$MPICC"
then
    echo "==========================================================="
    echo "configuring $MPICH"
    echo "==========================================================="
    echo ""
    if test -f "$MPICH_BUILD/config.status"
    then
        echo "$MPICH already configured"
    else
        rm -rf $MPICH_BUILD
        mkdir $MPICH_BUILD || exit 1
        cd $MPICH_BUILD || exit 1
        echo "$SCRIPTPATH/contrib/$MPICH/configure --prefix=\"$INSTALLPATH\""
        $SCRIPTPATH/contrib/$MPICH/configure --prefix="$INSTALLPATH" || exit 1
        cd $BUILDPATH || exit 1
        if ! test -f "$MPICH_BUILD/config.status"
        then
            echo "$MPICH configure failed, see $MPICH_BUILD/config.log for details"
            exit 1
        fi
    fi
    echo ""

    echo "==========================================================="
    echo "making $MPICH"
    echo "==========================================================="
    echo ""
    if test -f "$INSTALLPATH/bin/mpicc" && test -x "$INSTALLPATH/bin/mpicc"
    then
        echo "$MPICH already installed"
    else
        cd $MPICH_BUILD || exit 1
        echo "make -j $NPROC"
        make -j $NPROC || exit 1
        cd $BUILDPATH || exit 1
    fi
    echo ""

    echo "==========================================================="
    echo "installing $MPICH"
    echo "==========================================================="
    echo ""
    if test -f "$INSTALLPATH/bin/mpicc" && test -x "$INSTALLPATH/bin/mpicc"
    then
        echo "$MPICH already installed"
    else
        cd $MPICH_BUILD || exit 1
        echo "make install"
        make install || exit 1
        cd $BUILDPATH || exit 1
    fi
    echo ""

    # add our 'internal' MPI to the PATH
    echo "adding \"$INSTALLPATH/bin\" to PATH"
    echo ""
    export PATH="$INSTALLPATH/bin:$PATH"
fi

echo "==========================================================="
echo "configuring $MATEX"
echo "==========================================================="
echo ""
if test -f "$MATEX_BUILD/config.status"
then
    echo "$MATEX already configured"
else
    rm -rf $MATEX_BUILD
    mkdir $MATEX_BUILD || exit 1
    cd $MATEX_BUILD || exit 1
    echo "$SCRIPTPATH/configure --prefix=\"$INSTALLPATH\""
    $SCRIPTPATH/configure --prefix="$INSTALLPATH" --without-blas || exit 1
    cd $BUILDPATH || exit 1
    if ! test -f "$MATEX_BUILD/config.status"
    then
        echo "$MATEX configure failed, see $MATEX_BUILD/config.log for details"
        exit 1
    fi
fi
echo ""

echo "==========================================================="
echo "making $MATEX"
echo "==========================================================="
echo ""
if test -f "$INSTALLPATH/bin/matex_svm" && test -x "$INSTALLPATH/bin/matex_svm"
then
    echo "$MATEX already installed"
else
    cd $MATEX_BUILD || exit 1
    echo "make V=0 -j $NPROC"
    make V=0 -j $NPROC || exit 1
    cd $BUILDPATH || exit 1
fi
echo ""

echo "==========================================================="
echo "installing $MATEX"
echo "==========================================================="
echo ""
if test -f "$INSTALLPATH/bin/matex_svm" && test -x "$INSTALLPATH/bin/matex_svm"
then
    echo "$MATEX already installed"
else
    cd $MATEX_BUILD || exit 1
    echo "make V=0 install"
    make V=0 install || exit 1
    cd $BUILDPATH || exit 1
fi
echo ""

# add our 'internal' MPI to the PATH
echo "make sure to add \"$INSTALLPATH/bin\" to your PATH"

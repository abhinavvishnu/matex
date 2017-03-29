# $Id: rules.make 2285 2015-12-30 20:48:25Z wkliao $

# The purpose of this file is to contain common make(1) rules.
# It should be processed by every execution of the that utility.

.SUFFIXES:
.SUFFIXES:	.a .o .i .f .c .cpp .F .y .l .m4 .f90 .F90


################################################################################
# Compilation (including preprocessing):

.c.o:
	$(COMPILE.c) $<

.c.i:
	$(CPP) $(CPPFLAGS) $< >$@

.cpp.o:
	$(COMPILE.cxx) $<

.f.o:
	$(COMPILE.f) $<

.f90.o:
	$(COMPILE.f90) $<

.F90.o:
	$(COMPILE.F90) $<

.F.o:
	$(COMPILE.F) $<

# Not all FORTRAN compilers support C-preprocessing of *.F files; ergo, a 
# relatively complicated rule ensues.
# .F.o:
# 	@case "$(COMPILE.F)" in	\
# 	    '')	\
# 		set -x;	\
# 		$(FPP) $(FPPFLAGS) -C $*.F | grep -v '^#' >$*-tmp.f || 	\
# 		    ($(RM) -f $*-tmp.f ; exit 1);	\
# 		$(COMPILE.f) -o $@ $*-tmp.f || ($(RM) -f $*-tmp.f; exit 1);	\
# 		$(RM) -f $*-tmp.f;	\
# 		;;	\
# 	    *)	\
# 		$(COMPILE.F) $<;	\
# 		;;	\
# 	esac

#.F.f:
#	$(FPP) $(FPPFLAGS) $*.F | grep -v '^#' >$*.f || ($(RM) -f $*.f; exit 1)

.m4.c:
	$(M4) $(M4FLAGS) $< >$@

.m4.F:
	$(M4) $(M4FLAGS) $< >$@

.m4.f90:
	$(M4) $(M4FLAGS) $< >$@

.m4.F90:
	$(M4) $(M4FLAGS) $< >$@


################################################################################
# Libraries:

# lib:		$(LIBRARY)

#-------------------------------------------------------------------------------
# Shared Libraries:
#
# Here only as a place holder and notebook.  Don't try to use this stuff
# unless you really, really know what you're doing!  (And even then we
# guarantee nothing!)
#
shared_library:
	@case `uname -sr` in \
	HP-UX*) \
	    $(MAKE) $(MFLAGS) hpux_shared_library;; \
	IRIX*) \
	    $(MAKE) $(MFLAGS) irix_shared_library;; \
	Linux*) \
	    $(MAKE) $(MFLAGS) linux_shared_library;; \
	OSF1*) \
	    $(MAKE) $(MFLAGS) osf1_shared_library;; \
	'SunOS 4'*) \
	    $(MAKE) $(MFLAGS) sunos4_shared_library;; \
	'SunOS 5'*) \
	    $(MAKE) $(MFLAGS) sunos5_shared_library;; \
	*) \
	    echo 1>&2 "Don't know how to make a shared library" \
		"on this system"; \
	    exit 1;; \
	esac

hpux_shared_library:
	nm libpnetcdf.a | grep extern | grep entry | \
	    awk '-F|' '{print $$1}' | sed 's/^/-u /' >symbols.log
	ld -o $(LIBRARY:.a=.sl) -b -c symbols.log $(LIBRARY)
	$(RM) -f symbols.log
irix_shared_library:
	ld -o $(LIBRARY:.a=.so) -shared -no_archive \
	    -all $(LIBRARY) -none -lc -lC $(LIBS)
linux_shared_library:
	mpicc -o $(LIBRARY:.a=.so) -shared -Wl,--whole-archive $(LIBRARY) -Wl,--no-whole-archive
	#ld -o $(LIBRARY:.a=.so) -shared --whole-archive $(LIBRARY)
osf1_shared_library:
	ld -o $(LIBRARY:.a=.so) -shared -all $(LIBRARY)
sunos4_shared_library:
	undefopts=`/bin/nm $(LIBRARY) | awk '$$2~/^T$$/{print $$3}' | \
		   sed 's/^/-u /'` && \
	    ld -o $(LIBRARY:.a=.so) $$undefopts $(LIBRARY)
sunos5_shared_library:
	undefopts=`/usr/ccs/bin/nm $(LIBRARY) | grep GLOB | grep FUNC | \
		   awk '-F|' '{print $$8}' | sed 's/^/-u /'` && \
	    ld -o $(LIBRARY:.a=.so) -G $$undefopts $(LIBRARY)


################################################################################
# Linking:


################################################################################
# $(INSTALL)ation:

$(INCDIR)/$(HEADER):	$(INCDIR) $(HEADER)
	$(INSTALL) $(srcdir)/$(HEADER) $@
$(INCDIR)/$(HEADER1):	$(INCDIR) $(HEADER1)
	$(INSTALL) $(srcdir)/$(HEADER1) $@
$(INCDIR)/$(HEADER2):	$(INCDIR) $(HEADER2)
	$(INSTALL) $(srcdir)/$(HEADER2) $@
$(INCDIR)/$(HEADER3):	$(INCDIR) $(HEADER3)
	$(INSTALL) $(srcdir)/$(HEADER3) $@

$(LIBDIR)/$(LIBRARY):	$(LIBDIR) $(LIBRARY)
	$(INSTALL) -d -m 755 $(LIBDIR)
	$(INSTALL) -m 644  $(LIBRARY) $@

$(BINDIR)/$(PROGRAM):	$(BINDIR) $(PROGRAM)
	$(INSTALL) -d -m 755 $(BINDIR)
	$(INSTALL) -m 755  $(PROGRAM) $@

#$(BINDIR) \
#$(INCDIR) \
#$(LIBDIR) \
#$(MANDIR) :
#	-test -d $@ || mkdir $@

#$(MANDIR)/man1 \
#$(MANDIR)/man3 \
#$(MANDIR)/man3f \
#$(MANDIR)/man3f90 :		$(MANDIR)
#	-test -d $@ || mkdir $@

# $(MANDIR)/man1/$(MANUAL):	$(MANDIR)/man1 $(MANUAL)
# 	$(INSTALL) $(srcdir)/$(MANUAL) $@
# $(MANDIR)/man3/$(MANUAL):	$(MANDIR)/man3 $(MANUAL)
# 	$(INSTALL) $(srcdir)/$(MANUAL) $@
# $(MANDIR)/man3f/$(MANUAL):	$(MANDIR)/man3 $(MANDIR)/man3/$(MANUAL) \
# 				$(MANDIR)/man3f
# 	$(RM) -f $@
# 	$(LN_S) $(MANDIR)/man3/$(MANUAL) $@
# $(MANDIR)/man3f90/$(MANUAL):	$(MANDIR)/man3 $(MANDIR)/man3/$(MANUAL) \
# 				$(MANDIR)/man3f90
# 	$(RM) -f $@
# 	$(LN_S) $(MANDIR)/man3/$(MANUAL) $@

################################################################################
# Cleanup:

clean:		FORCE
	@if [ -n "$(SUBDIRS)" ]; then \
	    subdirs="$(SUBDIRS)"; \
	    for subdir in $$subdirs; do \
		(cd $$subdir && $(MAKE) $(MFLAGS) clean) ; \
	    done; \
	fi
	@$(RM) -f *.o *.a *.so *.sl *.i *.Z core core.* $(GARBAGE) \
		  *.gcda *.gcno gmon.out

distclean:	FORCE
	@if [ -n "$(PACKING_SUBDIRS)" ]; then \
	    subdirs="$(PACKING_SUBDIRS)"; \
	    for subdir in $$subdirs; do \
		(cd $$subdir && $(MAKE) $(MFLAGS) distclean) ; \
	    done; \
	fi
	@if [ -n "$(PACKING_SUBDIRS)" ]; then \
	    subdirs="$(PACKING_SUBDIRS)"; \
	    for subdir in $$subdirs; do \
		if ! [ $(srcdir) -ef `pwd` ] ; then rmdir $$subdir ; fi \
	    done; \
	fi
	@$(RM) -rf SunWS_cache
	@$(RM) -f *.o *.a *.so *.sl *.i *.Z core core.* $(GARBAGE) \
		  *.gcda *.gcno gmon.out \
	          MANIFEST *.log $(DIST_GARBAGE) cscope.out cscope.files
	@$(RM) -f Makefile

################################################################################
# Dependencies:

# This target should only need to be made at the UPC.
# NOTES:
#   *  The target file might be a symbolic link.
#   *  The name of the target doesn't match the name of the created file to
#      prevent constant updating of the included file `depend' by make(1).
#
deps:		FORCE
	$(CC_MAKEDEPEND) $(CPPFLAGS) *.c | grep -v '/usr/include' >>depend
	sort -u -o depend depend


################################################################################
# Distribution:

# The following rule echoes the contents of $(PACKING_LIST) in the
# current directory and in all subdirectories.  All pathnames are made
# relative to the current directory.
#
MANIFEST.echo:	FORCE
	echo $(PACKING_LIST) | fmt -1
	if [ -n "$(PACKING_SUBDIRS)" ]; then \
	    subdirs="$(PACKING_SUBDIRS)"; \
	    for subdir in $$subdirs; do \
		(cd $$subdir && \
		echo 1>&2 Creating $@ in `pwd` && \
		$(MAKE) $(MFLAGS) MANIFEST.echo | sed "s|^|$$subdir/|") || exit 1; \
	    done; \
	else \
	   :; \
	fi

# The following rule ensures that all files in $(PACKING_LIST) exist in
# the current directory and in all subdirectories.
#
ensure_manifest:	$(PACKING_LIST) FORCE
	if [ -n "$(SUBDIRS)" ]; then \
	    subdirs="$(SUBDIRS)"; \
	    for subdir in $$subdirs; do \
		(cd $$subdir && \
		echo 1>&2 Creating $@ in `pwd` && \
		$(MAKE) $(MFLAGS) ensure_manifest) || exit 1; \
	    done; \
	else \
	   :; \
	fi


################################################################################
# Misc:

FORCE:

.PHONY: FORCE all library clean distclean TAGS clean_macros rmdir_src_test b-test c-test f-test
.PHONY: subdirs $(SUBDIRS) install $(INSTALLDIRS) uninstall $(UNINSTALLDIRS)
.PHONY: tests check testing $(CHECK_DIRS) $(PTEST_DIRS) verbose_check verbose_testing $(VCHECK_DIRS)
.PHONY: ptest ptests ptest2 ptest4 ptest6 ptest8 ptest10


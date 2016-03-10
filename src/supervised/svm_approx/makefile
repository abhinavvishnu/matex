#CC=/home/vishnu/Cascade/src/projects/matex_research/mpi/mvapich_install/bin/mpicxx
CC=mpicxx
#CC=/home/vishnu/Cascade/src/projects/matex_research/mpi/mpich_install/bin/mpicxx
CFLAGS=-c -Wall -O3 -std=gnu++0x 
LDFLAGS=
SOURCES=driver.cpp distance.cpp dataset.cpp comm.cpp smo.cpp shrinking.cpp testing.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=smo

all: $(SOURCES) $(EXECUTABLE)
		
$(EXECUTABLE): $(OBJECTS) 
		$(CC) $(OBJECTS) $(LDFLAGS) -o $@

.cpp.o:
		$(CC) $(CFLAGS) $< -o $@

clean:
		rm -rf *.o  $(EXECUTABLE)


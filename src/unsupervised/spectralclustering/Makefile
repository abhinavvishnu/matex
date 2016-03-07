CC=mpicxx
C=gcc
CFLAGS=-O3 -Wall -Wno-sign-compare

all: compute_distance distance_to_similarity evd kmeans

clean:
	rm -f *.o parpack/*.o parpack/*.a f2c/*.o f2c/*.a
	rm -f compute_distance distance_to_similarity evd kmeans


common.o: common.cc common.h
	$(CC) -c $(CFLAGS) common.cc -o common.o

compute_distance.o: compute_distance.cc compute_distance.h
	$(CC) -c $(CFLAGS) compute_distance.cc -o compute_distance.o

compute_distance: compute_distance.o common.o
	$(CC) $(CFLAGS) common.o compute_distance.o -o compute_distance

distance_to_similarity.o: distance_to_similarity.cc distance_to_similarity.h
	$(CC) -c $(CFLAGS) distance_to_similarity.cc -o distance_to_similarity.o

distance_to_similarity: distance_to_similarity.o common.o
	$(CC) $(CFLAGS) common.o distance_to_similarity.o -o distance_to_similarity

parpacklib:
	( cd parpack; $(MAKE))

f2clib:
	( cd f2c; $(MAKE))

evd.o: evd.cc evd.h
	$(CC) -c $(CFLAGS) evd.cc -o evd.o

evd: evd.o common.o parpacklib f2clib
	$(CC) $(CFLAGS) common.o evd.o -Lparpack/ -Lf2c/ -lparpack -lf2c -o evd

kmeans.o: kmeans.cc kmeans.h
	$(CC) -c $(CFLAGS) kmeans.cc -o kmeans.o

kmeans: kmeans.o common.o
	$(CC) $(CFLAGS) common.o kmeans.o -o kmeans

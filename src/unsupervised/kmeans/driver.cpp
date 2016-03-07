#include "kmeans.hpp"
#include <cassert>
#include <cstdlib>

int main(int argc, char **argv) {
    assert(argc == 3);

    kmeans *k = new kmeans(&argc, &argv);

    // seed the initial centroids
    
    k->seed();

    // call the iterative part
    k->iterative();



    return 0;        
}

#include "knn.hpp"
#include <cassert>
#include <cstdlib>

int main(int argc, char **argv) {
    KNN *knn = new KNN(&argc, &argv);

    knn->usage();
    knn->train();
    knn->test();

    return 0;        
}

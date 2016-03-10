#include "smo.hpp"
#include <cassert>
#include <cstdlib>

int main(int argc, char **argv) {
    SMO *smo = new SMO(&argc, &argv);

    smo->run();
//    delete smo;

    return 0;        
}

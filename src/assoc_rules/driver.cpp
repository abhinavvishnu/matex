#include "fpgrowth.hpp"
#include <cassert>
#include <cstdlib>

int main(int argc, char **argv) {
    FPG *fpg = new FPG(&argc, &argv);

    fpg->usage();

    fpg->run();

    delete fpg;
    return 0;        
}

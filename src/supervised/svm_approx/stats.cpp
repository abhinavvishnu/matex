#include "stats.hpp"
#include <iostream>

using namespace std;

Stats::Stats() {
    allreduce_time = 0;
    bcast_time = 0;
}

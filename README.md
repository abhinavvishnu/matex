MaTEx: Machine Learning Toolkit for Extreme Scale
=================================================

MaTEx is a collection of high performance parallel machine learning and
data mining (MLDM) algorithms, targeted for desktops, supercomputers
and cloud computing systems. MaTEx provides a handful of widely used
algorithms in Clustering, Classification and Association Rule Mining
(ARM). As of current release it supports K-means, Spectral Clustering
algorithms for Clustering, Support Vector Machines, KNN algorithms for
Classification, and FP-Growth for Association Rule Mining.

MaTEx uses state-of-the-art programming models such as Message Passing
Interface (MPI) and Global Arrays (GA) for targeting massively parallel
systems readily available on modern desktops, supercomputers and cloud
computing systems.

MaTEx bundles required software for parallel computing such as
mpich-3.1 and Global Arrays-5.3. These package are automatically built,
if they are not found on your system. Please refer to the support page
for more details. 

Building MaTEx
--------------
The INSTALL file contains the generic instructions for building any
configure-based software.

We have provided a build.sh file in the current directory which will
automate some of the installation of dependencies (the bundled MPI and
GA packages) as well as the installation of MaTEx. If the build.sh is
successful, you will have a ./bin directory containing the binaries for
the various tools MaTEx provides.

Help
----
Email matex-users@googlegroups.com for all questions/bugs/requests.

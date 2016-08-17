MaTEx: Machine Learning Toolkit for Extreme Scale
=================================================

MaTEx is a collection of high performance parallel machine learning and
data mining (MLDM) algorithms, targeted for desktops, supercomputers
and cloud computing systems. 

Supported Algorithms:
--------------------
1) Deep Neural Networks (extending Google TensorFlow with MPI)
2) k-means, Spectral Clustering
3) KNN, Support Vector Machines

MaTEx uses Message Passing Interface (MPI), which can be used on
Desktops, Cloud Computing Systems and Supercomputers.

System Software Requirements:
-----------------------------
MaTEx bundles required software for parallel computing such as
mpich-3.1. They are automatically built, if they are not found on your system. 

Building MaTEx
--------------
Please refer to the INSTALL file for details.

We have provided a build.sh file in the current directory which will
automate some of the installation of dependencies (the bundled MPI and
GA packages) as well as the installation of MaTEx. 

Help
----
Email matex-users@googlegroups.com for all questions/bugs/requests.

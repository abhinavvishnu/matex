MaTEx: Machine Learning Toolkit for Extreme Scale
=================================================

MaTEx is a collection of high performance parallel machine learning and
data mining (MLDM) algorithms, targeted for desktops, supercomputers
and cloud computing systems. 

Supported Algorithms
--------------------
1) **Deep Neural Networks (extending Google TensorFlow with MPI)**

2) k-means, Spectral Clustering

3) KNN, Support Vector Machines

MaTEx uses Message Passing Interface (MPI), which can be used on
Desktops, Cloud Computing Systems and Supercomputers.

System Software Requirements
-----------------------------
MaTEx bundles required software for parallel computing such as
mpich-3.1. They are automatically built, if they are not found on your system. 

Building MaTEx
--------------
Please refer to the INSTALL file for details.

We have provided a build.sh file in the current directory which will
automate some of the installation of dependencies (the bundled MPI and
GA packages) as well as the installation of MaTEx. 

Publications
------------
1) Fault Tolerant Support Vector Machines. Sameh Shohdy, Abhinav Vishnu, and
Gagan Agrawal. International Conference on Parallel Processing (ICPP'16)

2) Accelerating Deep Learning with Shrinkage and Recall. Shuai Zheng,
Abhinav Vishnu, and Chrish Ding. Arxiv Report.

3) Distributed TensorFlow with MPI. Abhinav Vishnu, Charles Siegel and Jeff
Daily. ArXiv Report.

4) Fault Modeling of Extreme Scale Applications using Machine Learning.
Abhinav Vishnu, Hubertus van Dam, Nathan Tallent, Darren Kerbyson and
Adolfy Hoisie. IEEE International Parallel and Distributed Processing
Symposium (IPDPS), May, 2016 (pdf)

5) Predicting the top and bottom ranks of billboard songs using Machine
Learning. Vivek Datla and Abhinav Vishnu. ArXiv report

6) Fast and Accurate Support Vector Machines on Large Scale Systems.
Abhinav Vishnu, Jeyanthi Narasimhan, and Lawrence Holder, Darren
Kerbyson and Adolfy Hoisie. IEEE Cluster 2015, September, 2015 (pdf).

7) Large Scale Frequent Pattern Mining using MPI One-Sided Model. Abhinav
Vishnu, and Khushbu Agarwal. IEEE Cluster 2015, September, 2015 (pdf).

8) Acclerating k-NN with Hybrid MPI and OpenSHMEM. Jian Lin, Khaled
Hamidouche, Jie Zhang, Xiaoyi Lu, Abhinav Vishnu and Dhabaleswar Panda.
OpenSHMEM Workshop, August, 2015. pdf .

Acknowledgement
---------------

MaTEx is supported by PNNL Analysis in Motion (AIM) initiative and US
Government.

Support
-------
Email matex-users@googlegroups.com for all questions/bugs/requests.

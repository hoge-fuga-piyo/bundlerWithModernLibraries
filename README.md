# README #

### What is this repository for? ###
This repository is the implementation of Structure from Motion reffering to following paper. (Not exactly same.)  
[Modeling the World from Internet Photo Collections](http://phototour.cs.washington.edu/ModelingTheWorld_ijcv07.pdf)

* Required
	* OpenCV 3.4 with following modules
		* extra modules
		* vtk
	* Ceres Solver
	* yaml-cpp 0.6.2
	* Google Test
	* Exiv2 (For using C++17, we use latest version from master branch on github. Exiv2 v0.27.2-RC3 or under cannot use on C++17.)
* Option
	* OpenMP

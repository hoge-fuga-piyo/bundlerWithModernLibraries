# README #

This repository is the implementation of Structure from Motion reffering to following paper. (Not exactly same.)  
[Snavely, Noah, Steven M. Seitz, and Richard Szeliski. "Modeling the world from internet photo collections." International journal of computer vision 80.2 (2008): 189-210.](http://phototour.cs.washington.edu/ModelingTheWorld_ijcv07.pdf)

* Required
	* OpenCV 3.4 with following modules
		* extra modules
		* vtk
	* Ceres Solver
	* yaml-cpp 0.6.2
	* Exiv2 (For using C++17, we use latest version from master branch on github. Exiv2 v0.27.2-RC3 or under cannot work on C++17.)
	* Google Test
* Option
	* OpenMP

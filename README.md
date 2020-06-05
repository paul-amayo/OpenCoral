# OpenCoral


**OpenCoral** (COnvex Relaxation ALgorithm) 

The `Coral` repository contains the source code for the core algorithm. It
should be model agnostic, so feel free to write an appropriate model for your purposes.


### Related Publications:
* [**Geometric Multi-Model Fitting with a Convex Relaxation Algorithm**]
*P Amayo,P. Piniés, L. M. Paz, and P. Newman*, CVPR 2018.


## Author
- Paul Amayo (paul.amayo@uct.ac.za)

## Dependencies
- Ubuntu 16.04
- Boost 1.58
- OpenCV 3.2
- Eigen 3.3.6
- Cuda 10.2
- OpenMP
- GLogs

## Installation
**NOTE:** These instructions assume you are running Ubuntu 16.04 or higher.

1. Install `apt` dependencies:
```bash
sudo apt-get install libboost-all-dev
```

2. Install OpenCV 3.2:
Please consult the
[OpenCV docs](http://docs.opencv.org/3.0-beta/doc/tutorials/introduction/linux_install/linux_install.html) for
instructions.

3. Install Eigen 3.3.6

4. Install `OpenCoral`:
```bash
cd OpenCoral
mkdir build
cd build
cmake -D CMAKE_INSTALL_PREFIX=path/to/install/directory ..
make install
```

## Usage
See `example` for example usage.


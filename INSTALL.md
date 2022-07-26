## Installation

### Requirements:
(We have tested this project with PyTorch 1.7.1 and torchvision 0.8.2.)
- PyTorch >= 1.0. 
- torchvision
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- OpenCV
- CUDA >= 9.0

### Step-by-step installation

```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name PDNet
conda activate PDNet

# this installs the right pip and dependencies for the fresh python
conda install ipython

# PDNet and coco api dependencies
pip install ninja yacs cython matplotlib tqdm opencv-python future

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for PyTorch 1.1.0 and CUDA 10.0
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch

export INSTALL_DIR=$PWD

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext

# install PyTorch Detection
cd $INSTALL_DIR
git clone https://github.com/yangli18/PDNet.git
cd PDNet

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop

# install the prediction collection module
cd pdnet_core/pred_collect_ops
python setup.py build develop


unset INSTALL_DIR

# or if you are on macOS
# MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py build develop
```

## Installation
### Basic dependencies
We recommend creating a new conda environment for a clean installation of the dependencies. All following commands are assumed to be executed within this conda environment. The code has been tested on CentOS 7.8, python 3.7 and CUDA 10.2. We used RTX 2080 Ti for training and testing our models.

```
conda create --name nba python=3.7
conda activate nba
```
Make sure CUDA 10.2 is your default cuda. If your CUDA 10.2 is installed in `/usr/local/cuda-10.2`, add the following lines to your `~/.bashrc` and run `source ~/.bashrc`:
```
export PATH=$PATH:/usr/local/cuda-10.2/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.2/lib64
export CPATH=$CPATH:/usr/local/cuda-10.2/include
``` 

Install dependencies, we use ${REPO_ROOT_DIR} to represent the working directory of this repo.
```
cd ${REPO_ROOT_DIR}
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

### Mesh penetration optimization dependencies
Please install our modified torch-mesh-isect CUDA extension for pytorch 1.5.1. Before installing the extension, make sure to set the environment variable *$CUDA_SAMPLES_INC* to the path that contains the header `helper_math.h`, which can be found in the repo [CUDA Samples repository](https://github.com/NVIDIA/cuda-samples). If your `helper_math.h` is located at `/usr/local/cuda-10.2/samples/common/inc`, add the following lines to your `~/.bashrc` and run `source ~/.bashrc`: 
```
export CUDA_SAMPLES_INC=/usr/local/cuda-10.2/samples/common/inc
```
After that, run the following commands:
```
cd img_to_mesh/extensions/torch-mesh-isect
python setup.py install
```

### Mesh evaluation dependencies
Please install [open3d 0.9.0](http://www.open3d.org/) for ICP. You also need to install the Chamfer Distance and Earth-Mover Distance CUDA extension:
```
pip install https://github.com/intel-isl/Open3D/releases/download/v0.9.0/open3d-0.9.0.0-cp37-cp37m-manylinux1_x86_64.whl
cd img_to_mesh/extensions/chamfer
python setup.py install
cd img_to_mesh/extensions/emd
python setup.py install
```
If your gcc version is 9, you may encounter errors when you build the extensions for cuda 10.2 and pytorch 1.5.1. To solve this problem, you can install gcc 8 on your machine using following commands:
```
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8
```
then select gcc version using the following command:
```
sudo update-alternatives --config gcc
```

### Demo dependencies
Please make sure [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) is installed on your system. For our case, we use the [Windows binary](https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases/tag/v1.5.1) and run the preprocessing code on a Windows machine. If you have difficulties installing OpenPose, you can also try [Alphapose](https://github.com/MVIG-SJTU/AlphaPose) and make sure the output json file is in [openpose format](https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/output.md). You may need to modify our preprocessing code a little bit if you are using Alphapose (We do not try with Alphapose though).

### Customized training dependencies
If you want to customize the hyperparameters of the spiral convolution training, please make sure the following packages are installed in python 3. 
- chumpy: This [link](https://github.com/mattloper/chumpy/issues/40) provides some solution for installing chumpy when pip>=20.1.
- opendr: This [link](https://github.com/mattloper/opendr/issues/19) provides some solution for installing opendr in python 3.
- [PSBody Mesh package](https://github.com/MPI-IS/mesh)

## Download pretrained checkpoints
We provide pretrained checkpoints at the [Google Drive](https://drive.google.com/drive/folders/1p7aJqogONpCVvR4gCEpgz_kOzEqnuym7?usp=sharing). After you download the checkpoints. please copy the folder to the target paths:
```
cd ${REPO_ROOT_DIR}
mkdir -p img_to_mesh/log
cp -r pretrained_ckpt/pose img_to_mesh/log
cp -r pretrained_ckpt/mesh img_to_mesh/log
mkdir -p global_position/field_lines/log
cp -r pretrained_ckpt/court_lines global_position/field_lines/log
```
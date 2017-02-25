# EECS-542_Assignment2

## Dataset
Download **PASCAL VOC 2011** and **VOC devkit** to root directory of this project, naming the folder as `TrainVal` and `VOCdevkit`, respectively. For the download links, please refer to the project instruction. Note that we are workign on **PASCAL VOC 2011 Segmentation Task**, please pay attention to the version and task.

### Storage on Flux

You could save the data and devkit in login node of Flux. However, you are encouraged to store them under your home directory under`/scratch/YOUR_UNIQUENAME` as the storage space is limited in login node, i.e.`/home/YOUR_UNIQUENAME`. 

If you save the data and devkit under`/scratch/YOUR_UNIQUENAME`, please create a symlink for each of the folder under where you save the work directory (code), as we have the convention that the dataset is saved under project root directory.

**If you are not sure how to create symlinks**, just simply clone the work to login node on flux and save the dataset and devkit under project root directory. It should be fine as long as the code could find the data.

## Installing Torch on Flux
After logging into Flux, under your home directory, do

**Load modules**

1. module load cuda
2. module load cudnn
3. module load openblas

**Clone and install**

1. git clone https://github.com/torch/distro.git ~/torch --recursive 
2. cd ~/torch; ./install.sh; source ~/.bashrc

**Install cutorch** 

-- not sure if the following works on flux
1. luarocks install torch
2. luarocks install cutorch

**If you need hdf5**

1. module load hdf5/1.8.16/gcc
2. luarocks install hdf5

**Checking whether the installation is successful**

Under Bash prompt, type `th`, and then `enter` (just like how you use python console). You should be seeing a logo of Torch and you are good to go.

## Useful References
### Github Repos

* [Caffe implementation of the FCN paper.](https://github.com/yunfan0621/fcn.berkeleyvision.org) This is the foundation of our project. Essentailly we are migrating this project to Torch.
* [An intuitive and clear tutorial of using Torch in deep learning.](https://github.com/soumith/cvpr2015/blob/master/Deep%20Learning%20with%20Torch.ipynb) Very good starter on using Torch to do deep learning, including but not restricted to how to send data to network and how to build a network.
* [A released project written in Torch from Jia Deng's lab.](https://github.com/yunfan0621/pose-hg-train) This is a little bit tedious, but we could follow the idea and organization of the project.
* [Home repo of Torch, where you could refer all the packages and helper document.](https://github.com/torch/torch7) If you are looking for the explanation of a certain function or you want to do something but you are not sure how to, please google it, and you are likely ending up in this repo.

### Key files in Berkeley FCN repo
In case you are not familiar with caffe, here is a brief instruction introducing some key files in Berkeley FCN repo.

#### Helper functions
* `./voc_layers.py`: shows how images are sent to the network. It seems that they are selecting a single image at a time, and do the training. Since we are focusing on a Fully Convolutional Network, we do not have to crop/resize the images to a fixed size any more. However, we still need to preprocess the data (e.g. subtract the mean, and you could also find instructions on how to do this using torch online).
* `./surgery.py`: manipulate on data and parameters of layers.
* `./score.py`: compute the metrics scores defined in the paper.
* `./infer.py`: create a semantic segmentation result using a trained model on a  given test image.

#### Training Code
* `./voc-fcn8s/`, `./voc-fcn16s/`, and `./voc-fcn32s/`: experiments correspond to different settings mentioned in the paper.
* `./voc-fcn8s/net.py`: how a network is created using pycaffe, there is a counterpart using Torch.
* `./voc-fcn8s/solve.py`: high-level training code, you could see how differnt function are put into use here.
* `./voc-fcn8s/solver.prototxt`: specifies hyper-parameters here
* `./voc-fcn8s/train.prototxt`: specifies the network structure, you could verify this by drawing out the network and compare to the one in the paper. 

#### Others
* `/data/`: discusses the dataset used in this experiment
* `/ilsvrc-nets/`: pre-trained networks, we will just fine-tune the parameters based on these models, instead of training from scratch.




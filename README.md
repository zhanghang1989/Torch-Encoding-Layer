# Deep Encoding
Created by [Hang Zhang](http://hangzh.com/)

### Table of Contents
0. [Introduction](#introduction)
0. [Installation](#installation)
0. [Experiments](#experiments)
0. [Benchmarks](#benchmarks)
0. [Acknowldgements](#acknowldgements)

## Introduction
We also provide PyTorch [implementation](https://github.com/zhanghang1989/PyTorch-Encoding)(recommanded, memory efficient). This repo is a Torch implementation of Encoding Layer as described in the paper:

**Deep TEN: Texture Encoding Network** [[arXiv]](https://arxiv.org/pdf/1612.02844.pdf)  
  [Hang Zhang](http://hangzh.com/), [Jia Xue](http://jiaxueweb.com/), [Kristin Dana](http://eceweb1.rutgers.edu/vision/dana.html)
```
@article{zhang2016deep,
  title={Deep TEN: Texture Encoding Network},
  author={Zhang, Hang and Xue, Jia and Dana, Kristin},
  journal={arXiv preprint arXiv:1612.02844},
  year={2016}
}
```

<div style="text-align:center"><img src ="https://raw.githubusercontent.com/zhanghang1989/Deep-Encoding/master/images/compare3.png" width="500" /></div>	

 Traditional methods such as bag-of-words BoW (left) have a structural similarity to more recent FV-CNN methods (center). Each component is optimized in separate steps. In our approach (right) the entire pipeline is learned in an integrated manner, tuning each component for the task at hand (end-to-end texture/material/pattern recognition).


## Installation
On Linux
```bash
luarocks install https://raw.githubusercontent.com/zhanghang1989/Deep-Encoding/master/deep-encoding-scm-1.rockspec
```
On OSX
```bash
CC=clang CXX=clang++ luarocks install https://raw.githubusercontent.com/zhanghang1989/Deep-Encoding/master/deep-encoding-scm-1.rockspec
```
## Experiments
- The Joint Encoding experiment in Sec4.2 will execute by default (tested using 1 Titan X GPU). This achieves *12.89%* percentage error on STL-10 dataset, which is ***49.8%*** relative improvement comparing to pervious state-of-the art *25.67%* of Zhao *et. al. 2015*.:
  ```bash
  git clone https://github.com/zhanghang1989/Deep-Encoding
  cd Deep-Encoding/experiments
  th main.lua
  ```
- Training Deep-TEN on MINC-2500 in Sec4.1 using 4 GPUs. 
	
	0. Please download the pre-trained
[ResNet-50](https://d2j0dndfm35trm.cloudfront.net/resnet-50.t7) Torch model 
and the [MINC-2500](http://opensurfaces.cs.cornell.edu/static/minc/minc-2500.tar.gz) dataset to ``minc`` folder before executing the program (tested using 4 Titan X GPUs).
	```bash
	th main.lua -retrain resnet-50.t7 -ft true \
	-netType encoding -nCodes 32 -dataset minc \
	-data minc/ -nClasses 23 -batchSize 64 \
	-nGPU 4 -multisize true
	```
	
	0. To get comparable results using 2 GPUs, you should change the batch size and the corresponding learning rate:
  ```bash
	th main.lua -retrain resnet-50.t7 -ft true \
	-netType encoding -nCodes 32 -dataset minc \
	-data minc/ -nClasses 23 -batchSize 32 \
	-nGPU 2 -multisize true -LR 0.05\
	```
		
### Benchmarks
Dataset                      |MINC-2500| FMD | GTOS | KTH |4D-Light
:----------------------------|:-------:|:---:|:----:|:---:|:------:
FV-SIFT                      |46.0     |47.0 |65.5  |66.3 |58.4
FV-CNN(VD)                   |61.8     |75.0 |77.1  |71.0 |70.4
FV-CNN(VD) <sub>multi<sub>   |63.1     |74.0 |79.2  |77.8 |76.5 
FV-CNN(ResNet)<sub>multi<sub>|69.3     |78.2 |77.1  |78.3 |77.6
Deep-TEN\*(**ours**) |**81.3**|80.2<sub>±0.9<sub>|**84.5<sub>±2.9<sub>**|**84.5<sub>±3.5<sub>**|**81.7<sub>±1.0<sub>**
State-of-the-Art             |76.0<sub>±0.2<sub>|**82.4<sub>±1.4<sub>**| 81.4|81.1<sub>±1.5<sub>|77.0<sub>±1.1<sub>

### Acknowldgements
We thank Wenhan Zhang from Physics department, Rutgers University for discussions of mathematic models. 
This work was supported by National Science Foundation award IIS-1421134. 
A GPU used for this research was donated by the NVIDIA Corporation.

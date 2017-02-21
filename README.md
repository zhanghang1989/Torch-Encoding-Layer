# Deep Encoding
Created by [Hang Zhang](http://hangzh.com/)

### Table of Contents
0. [Introduction](#introduction)
0. [Install](#install)
0. [Experiments](#experiments)

## Introduction
This repo is a Torch implementation of Deep Encoding (Encoding Layer) as described in the [paper](https://arxiv.org/pdf/1612.02844.pdf). If you use Encoding Layer in your research, please cite our paper:

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


## Install
```bash
luarocks install https://raw.githubusercontent.com/zhanghang1989/Deep-Encoding/master/deep-encoding-scm-1.rockspec
```

## Experiments
0. The Joint Encoding experiment in Sec4.2 will execute by default (tested using 1 Titan X GPU). This achieves *12.89%* percentage error on STL-10 dataset, which is ***49.8%*** relative improvement comparing to pervious state-of-the art *25.67%* of Zhao *et. al. 2015*.:

  ```bash
  git clone https://github.com/zhanghang1989/Deep-Encoding
  cd Deep-Encoding/experiments
  th main.lua
  ```
0. Training Deep-TEN on MINC-2500 in Sec4.1 using 4 GPUs. 
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

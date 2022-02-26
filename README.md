# Black-Box-Defense

This repository contains the code and models necessary to replicate the results of our recent paper:

**How to Robustify Black-Box ML Models? A Zeroth-Order Optimization Perspective** <br>
*[Yimeng Zhang](https://damon-demon.github.io), Yuguang Yao, Jinghan Jia, [Jinfeng Yi](https://jinfengyi.net), [Mingyi Hong](https://people.ece.umn.edu/~mhong/mingyi.html), [Shiyu Chang](https://engineering.ucsb.edu/people/shiyu-chang), [Sijia Liu](https://lsjxjtu.github.io)* <br>

**ICLR'22 (Spotlight)** <br>
Paper: https://openreview.net/forum?id=W9G_ImpHlQd <br>

We formulate the problem of black-box defense (as shown in Fig. 1) and investigate it through the lens of zeroth-order (ZO) optimization. Different from existing work, our paper aims to design the restriction-least black-box defense and our formulation is built upon a query-based black-box setting, which avoids the use of surrogate models.

<p>
<img src="figures/Fig_1.png" width="888" >
</p>

<p>
<img src="figures/Table_1.png" width="888" >
</p>

## Overview of the Repository

Our code is based on the open source codes of [Salmanet al.](https://github.com/microsoft/denoised-smoothing). The major contents of our repo are as follows:

* [code/](code) contains the code for our experiments on MNIST, CIFAR-10, STL-10, Restricted ImageNet.

Let us dive into the files in [code/](code):

1. `train_classifier.py`: a generic script for training ImageNet/Cifar-10 classifiers, with Gaussian agumentation option, achieving SOTA.
2. `AE_DS_train.py`: the main code of our paper which is used to train the different AE-DS/DS model with FO/ZO optimization methods used in our paper.

# Multi-Pose Feature Fusion Network

## Introduction
This is an implementation for "animal re-ID"

## Prerequisites
Linux  
Pytorch  
GPU+CUDA

## Dataset
Mendeley Data： He, Zhimin (2022), “Multi-pose dog dataset”, Mendeley Data, V1, doi: 10.17632/v5j6m8dzhv.1


## pretrain files 
Our model train needs a Pose discrimination module weight. Please download it before train model.

You need to put the pre-training files into the 'model' folder before training. Download the Pose discrimination module pre-trained model from [here](https://pan.baidu.com/s/18SO2dsKb4Y8D8LnSczcPdQ ).
提取码：mpdd

## Train

```
python [path to repo]/main.py --config [path to repo]/mpfn_config.yaml --save ''
```

## Test

~~~
python [path to repo]/main.py --test_only --config ./mpfn_config.yaml --pre_train ''
~~~






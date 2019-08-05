# Domain Generalization for biological images 
Link for paper : [Generalizing to Unseen Domains via Adversarial Data Augmentation](https://arxiv.org/abs/1805.12018)

## Overview

### Files

``model.py``: to build tf's graph

``trainOps.py``: to train/test

``exp_configuration``: config file with the hyperparameters

### Prerequisites

Python 2.7, Tensorflow 1.6.0

## How it works

1 - Clone Repository
2 - Go to directory - Domain-Generalization-on-Biological-Data
3 - Make sure you have files - (train.csv , test.csv , train_details.csv and test_details.csv) and folder - (images) in the above mentioned directory.
4 - mkdir data
5 - python download_and_process_mnist.py
6 - sh run_exp

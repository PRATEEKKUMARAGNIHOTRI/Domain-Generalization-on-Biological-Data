# Domain Generalization on biological images 
Link for paper : [Generalizing to Unseen Domains via Adversarial Data Augmentation](https://arxiv.org/abs/1805.12018)

## Overview

### Files

``model.py``: to build tf's graph

``trainOps.py``: to train/test

``exp_configuration``: config file with the hyperparameters

## How it works

1 - Clone Repository
<br>2 - Go to directory - "Domain-Generalization-on-Biological-Data"
<br>3 - Make sure you have files - (train.csv , test.csv , train_details.csv and test_details.csv) and folder - (images) in the above mentioned directory.
<br>4 - mkdir data
<br>5 - python process_data.py
<br>6 - sh run_exp

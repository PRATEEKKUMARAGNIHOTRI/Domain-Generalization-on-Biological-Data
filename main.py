import gc
gc.collect()

import tensorflow as tf
from model import Model
from trainOps import TrainOps
import glob
import os
import cPickle

import numpy.random as npr
import numpy as np

EXP_DIR = "./"

model = Model()
trainOps = TrainOps(model, EXP_DIR)
trainOps.load_exp_config()

print("Training")
trainOps.train()

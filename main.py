import tensorflow as tf
from model import Model
from trainOps import TrainOps
import glob
import os
import cPickle

import numpy.random as npr
import numpy as np

def main(_):

    EXP_DIR = "./"

    model = Model()
    trainOps = TrainOps(model, EXP_DIR)
    trainOps.load_exp_config()
    
    print("Training")
    trainOps.train()       
    print 'Testing'
    trainOps.test('svhn')       

if __name__ == '__main__':
    tf.app.run()



    







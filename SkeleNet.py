# -*- coding: utf-8 -*-
"""
Created on Tue May 24 17:55:44 2022

@author: graha
"""
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import numpy as np
from Skeletize import Skeletize2D,Skeletize3D


##SKELENET will be designed as a whole system which utalizes machine learning 
##in combination with the skeletinzation algorithm to produce acurate skeletons from any data(2D or 3D)


class SkeleNet:
    
    def __init__(self):
        #loads in MachineLearning system on creation, saves and loads its self, 
        #will organize saves incase a test of settings is wanted to be seen
        self.a = 1    








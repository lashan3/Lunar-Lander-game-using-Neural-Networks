import numpy as np
import NN_hyperparams
from NN_hyperparams import nn_lambda, nn_lr, nn_momentum
import random
import math

#Class for defining neuron
class neuron_struct:
    def __init__(self, layer_position, activation_value, no_of_weights):
        self.activation = activation_value
        self.d_weights = []
        self.gradiant = 0
        self.weights = []
        self.initializing_random_weights(no_of_weights)
        self.position = layer_position  # setting neuron index
    
    def initializing_random_weights(self, no_of_weights):
        #Random weights to network
        for i in range(0, no_of_weights):
            self.weights.append(random.random())
            self.d_weights.append(0)
    
    #Calculations of weights
    def weigths_calculation(self, layer):
        total_sum = 0
        for i in range(0, len(layer)):
            total_sum = total_sum + (float(layer[i].activation) * float(layer[i].weights[self.position]))  
        self.sigmoid(total_sum) 
    #Activation Function
    def sigmoid(self, x):
        self.activation = 1 / (1 + math.exp(-(nn_lambda * x )))

import numpy as np
import tensorflow as tf
import pandas as pd

def array_string_to_int_classes(arr, string_classes=[]):
    """
        Converts a string array to int classes ranging from 0 to num of classes

        inputs:
            arr:            an array that holds string classes

        outputs:
            int_classes:    an array of the same shape as arr, but in integer classes
    """
    int_classes = np.zeros(arr.shape[0])

    for i in range(0, arr.shape[0]):
        for int_class in range(0, len(string_classes)):
            if(arr[i] == string_classes[int_class]):
                int_classes[i] = int_class
    
    return int_classes

def sample_weights(y):

    weights = np.zeros(y.shape[0])

    for i in range(0, y.shape[0]):
        if(y[1] == -1):
            weights[i] = 0
        else:
            weights[i] = 1
            
    return weights
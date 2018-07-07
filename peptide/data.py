# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 13:34:42 2018

@author: lux32, jtfl2
"""
import numpy as np
import pandas as pd
import os

aa = list('AILMVFWYNCQSTDERHKGP')

def make_matrix (sequence):
    matrix = np.zeros((20, 30))
    sequence = list(sequence)
    print(sequence)
    for i in range(len(sequence)):
        for j in range(len(aa)):
            if sequence[i] == aa[j]:
                matrix[j, i] = 1
    return np.reshape(matrix, (1, 30, 20))


def load_data (file_path):
    csv_path = os.path.join(file_path, "iedb.csv")
    data = pd.read_csv(csv_path, nrows=20)
    sequ = data['sequence'].values.tolist()
    matrix_list = [make_matrix(x) for x in sequ]
    bind = data['meas'].values.tolist()
    return matrix_list, bind



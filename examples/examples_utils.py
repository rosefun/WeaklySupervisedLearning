# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 19:22:43 2020

@author: rosefun
"""

# normalization
def normalize(x):
	return (x - np.min(x)) / (np.max(x) - np.min(x))

def get_data():
	X, y = datasets.load_breast_cancer(return_X_y=True)
	X = normalize(X)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=0)

	return X_train, X_test, y_train, y_test

from keras.layers.core import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


class DNN(object):
	"""
	Define a DNN model for classification.
	"""

	def __init__(self, batch_size=128):
		self.batch_size = batch_size

	def build_model(self, input_dim, output_dim, hidden_dim_list=[128, 50]):
		'''
		:param inputdim: int type, the dim of input data.
		:param outputdim: int type, the number of class.
		'''
		model = Sequential()
		model.add(Dense(hidden_dim_list[0], input_dim=input_dim, activation='relu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))
		for i in range(1, len(hidden_dim_list)):
			model.add(Dense(hidden_dim_list[i], activation='relu'))
			model.add(BatchNormalization())
			model.add(Dropout(0.5))
		model.add(Dense(output_dim, activation='softmax'))

		return model
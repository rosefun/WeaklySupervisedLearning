# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 19:22:43 2020

@author: rosefun
"""
import keras.backend as K
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.metrics import categorical_accuracy
from keras.layers.core import Dense, Activation, Dropout
import keras
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

import warnings
warnings.simplefilter('error')
warnings.filterwarnings('default', category=PendingDeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)
# filter warning of numpy
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ImportWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

class BootstrappingCallback(Callback):
	def __init__(self, batch_size=128, bootstrapping_type="hard", beta=None):
		super(BootstrappingCallback,self).__init__()
		self.batch_size = batch_size
		self.beta = beta
		self.bootstrapping_type = bootstrapping_type

	def train_generator(self, X, y):
		while True:
			n_batch = X.shape[0] // self.batch_size
			indices = np.arange(y.shape[0])
			np.random.shuffle(indices)
			for i in range(n_batch):
				current_indices = indices[i * self.batch_size: (i + 1) * self.batch_size]
				X_batch = X[current_indices]
				y_batch = y[current_indices]
				yield X_batch, y_batch

	def test_generator(self, X_test, y_test):
		while True:
			indices = np.arange(y_test.shape[0])
			for i in range(X_test.shape[0] // self.batch_size):
				current_indices = indices[i * self.batch_size:(i + 1) * self.batch_size]
				X_batch = X_test[current_indices]
				y_batch = y_test[current_indices]
				yield X_batch, y_batch

	def crossentropy(self, y_true, y_pred):
		'''
		Input:
			y_true: np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0]])
			y_pred: np.array([[0.8, 0.2, 0], [0.4, 0.5, 0.1], [0.1, 0.5, 0.4]])
		Output: np.array [0.22314355 0.69314718 0.69314718]
		'''
		# this gives the same result as using keras.objective.crossentropy
		y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
		y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
		loss = -(K.sum(y_true * K.log(y_pred), axis=-1))

		return loss

	def make_loss(self):
		def bootstrapping_soft(y_true, y_pred, beta=0.95):
			"""
			: y_true, 2-dim tensor, true label.
			: y_pred, 2-dim tensor.
			"""
			if self.beta is not None:
				beta = self.beta
			y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
			y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
			soft_loss = -K.sum((beta * y_true + (1. - beta) * y_pred) *
						  K.log(y_pred), axis=-1)
			return soft_loss

		def bootstrapping_hard(y_true, y_pred, beta=0.80):
			"""
			https://github.com/killthekitten/kaggle-carvana-2017/blob/master/losses.py
			"""
			if self.beta is not None:
				beta = self.beta
			y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
			y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
			hard_label = K.tf.cast(y_pred>0.5, K.tf.float32)
			hard_loss = -K.sum((beta * y_true + (1. - beta) * hard_label) *K.log(y_pred), axis=-1)
			return hard_loss
		if self.bootstrapping_type == "hard":
			return bootstrapping_hard
		elif self.bootstrapping_type == "soft":
			return bootstrapping_soft
		else:
			assert bootstrapping_type in ["soft", "hard"]
		return 

	def accuracy(self, y_true, y_pred):
		return categorical_accuracy(y_true, y_pred)

	def on_epoch_end(self, epoch, logs):
		pass

	def on_train_end(self, logs):
		""""""
		pass

class BootstrappingNeuralNetworkClassifier(object):
	"""
	"""
	def __init__(self, clf, batch_size=128, epochs=40, bootstrapping_type="hard", beta=0.95, patience=0, best_model_name="model_check_point_best_model"):
		"""
		:param clf:
		"""
		self.model = clf
		self.batch_size = batch_size
		self.epochs = epochs
		self.bootstrapping_type = bootstrapping_type
		self.bootstrapping_callback = BootstrappingCallback(batch_size=batch_size, beta=beta, bootstrapping_type=bootstrapping_type)
		self.patience = patience
		self.best_model_name = best_model_name

	def onehot(self, narr, nclass=None):
		"""
		:param narr: np.ndarray
		return onehot ndarray.
		"""
		if not nclass:
			nclass = np.max(narr) + 1
		return np.eye(nclass)[narr]

	def fit(self, X, y, validation_data=None,):
		"""
		:param X: numpy.ndarray, train datasets, 2-ndim
		:param y: numpy.ndarray, label of train datasets, scalar values, 1-ndim. 
					If label == -1, the sample is unlabel.
		"""
		if y.ndim == 1:
			y = self.onehot(y)

		# step 1. train nn clf with labeled datasets.
		self.model.compile(loss=self.bootstrapping_callback.make_loss(), optimizer='adam', 
		metrics=[self.bootstrapping_callback.accuracy])
		if validation_data is not None:
			self.fit_model(X, y, epochs=self.epochs, X_valid=validation_data[0], Y_valid=validation_data[1], patience=self.patience)
		else:
			self.fit_model(X, y, epochs=self.epochs)

	def fit_model(self, X_train, Y_train, X_valid=None, Y_valid=None, epochs=None, patience=0):
		if X_valid is not None:
			early_stopping = EarlyStopping(
				monitor='val_loss', 
				patience=patience, 
			)
			model_check_point_save = ModelCheckpoint('.{}.hdf5'.format(self.best_model_name), save_best_only=True, monitor='val_loss', mode='min')
			hist = self.model.fit_generator(self.bootstrapping_callback.train_generator(X_train, Y_train),
											steps_per_epoch=X_train.shape[0] // self.batch_size,
											validation_data=(X_valid, Y_valid), callbacks=[self.bootstrapping_callback, early_stopping, model_check_point_save],
											validation_steps=X_valid.shape[0] // self.batch_size, epochs=epochs).history
		else:
			hist = self.model.fit_generator(self.bootstrapping_callback.train_generator(X_train, Y_train),
											steps_per_epoch=X_train.shape[0] // self.batch_size,
											callbacks=[self.bootstrapping_callback],
											epochs=epochs
											).history

	def evaluate_model(self, X, Y):
		score = self.model.evaluate(X, Y, batch_size=self.batch_size, verbose=1)
		print('Test score:{:.4f}'.format(score[0]))
		print('Test accuracy:{:.4f}'.format(score[1]))
		return score[1]

	def predict_proba(self, X):
		pred = self.model.predict(X, batch_size=self.batch_size, verbose=1)
		return pred

	def predict(self, X):
		"""
		return 1-dim nd.array, scalar value of prediction. 
		"""
		pred = self.model.predict(X, batch_size=self.batch_size, verbose=1)
		pred = np.argmax(pred, axis=1)
		return pred





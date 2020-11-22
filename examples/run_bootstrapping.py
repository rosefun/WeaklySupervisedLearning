# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 19:22:43 2020

@author: rosefun
"""
from examples_utils import get_data, DNN
from sklearn import metrics
from weaklysupervised import BootstrappingNeuralNetworkClassifier

if __name__ == "__main__":
	X_train, X_test, y_train, y_test = get_data()
	DNN = DNN()
	clf = DNN.build_model(input_dim=30, output_dim=2)
	model = BootstrappingNeuralNetworkClassifier(clf, batch_size=128, epochs=40, bootstrapping_type="soft", 
	beta=0.95, patience=5, best_model_name="model_check_point_best_model")
	model.fit(X_train, y_train, validation_data=(X_test, y_test), )
	predict = model.predict(X_test)
	acc = metrics.accuracy_score(y_test, predict)
	print("bootstrapping accuracy", acc)




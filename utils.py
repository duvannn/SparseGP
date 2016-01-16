import numpy as np
from theano import config

"""
Returns the data as a numpy matrix
"""
def read_dataset(path):
	with open(path, "r") as raw_data:
		data = []
		for row in raw_data:
			nrow = []
			for value in row.split(" "):
				if len(value)>0:
					try:
						val = np.float(value)
						if np.isfinite(val) and not np.isnan(val):
							nrow.append(np.float(value))
						else:
							raise ValueError("bad value")
					except ValueError:
						print("someting bad happend")
			data.append(nrow)
	data = np.asmatrix(data,dtype=config.floatX)
	return data
"""
Returns the training and test set of the desired data set: kin40 or pumadyn32nm.
"""
def get_all_data(data_class):
	if data_class == "kin40k":
		testX = read_dataset("data/kin40k/kin40k_test_data.asc")
		testY = read_dataset("data/kin40k/kin40k_test_labels.asc")
		trainX = read_dataset("data/kin40k/kin40k_train_data.asc")
		trainY = read_dataset("data/kin40k/kin40k_train_labels.asc")
	elif data_class == "pumadyn32nm":
		testX = read_dataset("data/pumadyn-32nm/pumadyn32nm_test_data.asc")
		testY = read_dataset("data/pumadyn-32nm/pumadyn32nm_test_labels.asc")
		trainX = read_dataset("data/pumadyn-32nm/pumadyn32nm_train_data.asc")
		trainY = read_dataset("data/pumadyn-32nm/pumadyn32nm_train_labels.asc")
	else:
		raise Exception("Dataset not found");
	return trainX, trainY, testX, testY;

def mean_square_error(pred_y, y):
	return np.mean(np.power(pred_y-y,2))

def generateSinData(samples):
	domain = np.linspace(-2*np.pi, 2*np.pi, samples)
	X = np.matrix([[point,np.random.randn(1)] for point in domain])
	epsilon = 0.5* np.random.randn(samples)
	Y = np.sin(X[:,0].T)+epsilon
	X, Y = np.array(X),np.array(Y).T

	return X,Y
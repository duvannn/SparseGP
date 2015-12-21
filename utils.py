import numpy as np


#Returns the input X and corresponding output Y as a tuple X,Y 
def read_dataset(path):
	with open(path, "r") as raw_data:
		data = []
		for row in raw_data:
			nrow = []
			for value in row.split(" "):
				if len(value)>0:
					try:
						nrow.append(np.float(value))
					except ValueError:
						print value
			data.append(nrow)
	data = np.asmatrix(data)
	return data

# Returns the training and test set of the desired data set: kin40 or pumadyn32nm.
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

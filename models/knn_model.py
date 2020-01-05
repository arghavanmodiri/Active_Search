import pandas as pd
from datetime import datetime

def knn_model_prob(alpha, weights, positive_train_idx, negative_train_idx,
				 test_idx):
	#written just for binary class
	probs_p = alpha + weights.iloc[test_idx].iloc[:, positive_train_idx].sum(axis=1)
	probs_n = (1.0 - alpha) + weights.iloc[test_idx].iloc[:, negative_train_idx].sum(axis=1)

	#normalize probabilities
	probs = probs_p / (probs_p + probs_n)

	return probs


def knn_search(A, k):
	neighbours_list = []
	distance_list = []
	for row in A.index:
		dist = (A.iloc[row]-A).pow(2).sum(axis=1).sort_values()
		d = pd.Series(dist.tolist())
		d_n = pd.Series(dist.index.tolist())
		neighbours_list.append(dist.index.tolist())
		distance_list.append(dist.tolist())

	neighbours = pd.DataFrame(neighbours_list)
	distance= pd.DataFrame(distance_list)
	neighbours = neighbours.iloc[:,1:k+1]
	distance = distance.iloc[:,1:k+1]
	#neighbours = neighbours.T.iloc[:,1:k+1]
	#distance = distance.T.iloc[:,1:k+1]
	return [neighbours, distance]


#A = pd.DataFrame({'x':[1,2,3,7,2], 'y':[2,5,1,4,2]})
#[neighbours, distance] = knn_search(A,3)
#print("old n:\n", neighbours)
#print("old nd:\n", distance)
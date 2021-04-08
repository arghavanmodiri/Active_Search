import torch
import pandas as pd
import random
from mf_ens_policy_pytorch import mfEns
import logging
from datetime import datetime
import multiprocessing

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(message)s')


data_df = pd.read_csv("./data/toy_problem/toy_example_data_500.csv")#,index_col=0)
#1st column is the index, 2nd is x, 3rd is y, and 4th is the label(0 or 1)
data = torch.tensor(data_df.values)

def knn_search(data_points, k):
    """
    returns the indicies of the k nearest neighbours of each point and their
    distance.
    data_points: contains x,y positions of all the points in type torch
    k: integer showing the number of neighbours to consider
    """
    dist= torch.norm(data_points[:, None] - data_points, dim=2, p=2)
    result = torch.topk(dist, k+1, largest=False)
    dist = result[0][:,1:] #get rid of the distance from itself
    neighbours = result[1][:,1:] #get rid of index of itself
    return [neighbours, dist]

def main():
    #configuration
    num_queries = 20 #equivalent to tatal budget
    alpha = 0.1
    k = 50
    visual = False
    repetition = 2
    policies = ["greedy", "ens", "2-step"]
    cost1 = 0.4
    cost2 = 1
    p1 = 0.7
    p2 = 1

    # initialization
    train = torch.empty(1, 1) #check size later!

    positive_labels = data[data[:,-1]==1,:]
    positive_labels_counts = len(positive_labels)
    num_points = len(data)
    data_learn = data.detach().clone()
    true_label = data_learn[:,-1].clone()
    #removed all the information about labels in data_learn
    data_learn[:,-1] = -1

    # making probabilistic model
    [neighbours, distances] = knn_search(data[:,1:-1], k)
    similarities = 1/distances
    weights = torch.zeros(num_points, num_points, dtype=torch.double)
    for i in range(len(neighbours)):
        weights[i,neighbours[i,:]] = similarities[i]

    train = positive_labels[0]
    data_learn[data_learn[:,0]==train[0], -1] = train[-1]
    '''
    for idx in range(positive_labels_counts):
        #sample = random.randrange(len(positive_labels))
        #train = positive_labels[sample]
        train = positive_labels[idx]
        ##CALL MFENS --> pos_count_mf_ens
    '''
    pool = multiprocessing.Pool()
    mfEns(pool, data_learn, true_label, weights, alpha, visual, num_queries, cost1, cost2, p1, p2)

if __name__ == "__main__":
    main()
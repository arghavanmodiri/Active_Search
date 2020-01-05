import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from selectors.unlabeled_selector import unlabeled_selector
from score_functions.search_expected_utility import search_expected_utility
from models.knn_model import knn_model_prob, knn_search
from query_strategies.argmax import argmax
from label_oracles.lookup_oracle import lookup_oracle

pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', None)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(message)s')
logging.info("{}".format(datetime.now()))
#loading Data
data_df = pd.read_csv("./data/toy_problem/toy_example_data_500.csv",index_col=0)

# configuration
num_queries = 10
alpha = 0.1
k = 50
visual = True
policy = "greedy" #from ["greedy", "ens", "2-step"]

# initialization
train_df = pd.DataFrame()
num_points = data_df.shape[0]
#initial point
positive_labels = data_df.loc[data_df['labels']==1]
train_df = data_df.iloc[
            [((positive_labels['x']-0.5).pow(2)+(positive_labels['y']-0.5).pow(2)).idxmin()]]
# for plotting
clr = {0:'silver',1:'dimgrey'}
clr_update = {0:'blue',1:'red'}
df = data_df["labels"].apply(lambda x: clr[x])

# making probabilistic model
[neighbours, distances] = knn_search(data_df[['x','y']], k)
similarities = 1/distances
weights = pd.DataFrame(0, index=range(num_points), columns=range(num_points))
for i in range(num_points):
    weights.iloc[i,neighbours.iloc[i].tolist()] = similarities.iloc[i].tolist()


if visual:
    plt.ion()

for query in range(num_queries):
    #print("*** query: ",query, " ***")
    logging.info("*** query: {} *** {}".format(query, datetime.now()))

    if visual:
        plt.scatter(data_df['x'], data_df['y'], s=20, color=df, alpha=0.7, marker='o')
        df.iloc[train_df.index]=data_df.loc[train_df.index,"labels"].apply(lambda x: clr_update[x])
        plt.draw()
        plt.waitforbuttonpress(-2)
        plt.clf()

    test_idx = unlabeled_selector(data_df, train_df.index)
    positive_train_idx = train_df.loc[train_df['labels']==1].index.values.tolist()
    negative_train_idx = train_df.loc[train_df['labels']==0].index.values.tolist()

    #Calculate score for all points in test set
    probs = knn_model_prob(alpha, weights,
                        positive_train_idx, negative_train_idx, test_idx)
    scores = search_expected_utility(train_df['labels'], probs)
    # **************GREEDY policy **************
    selected_point_idx = argmax(scores)

    #Ask Oracle
    label = lookup_oracle(data_df, selected_point_idx)
    logging.info("probe : {}".format(probs.loc[selected_point_idx]))
    logging.info("label : {}".format(label))
    logging.info("-------------")

    #Update observed data
    train_df = pd.concat([train_df, data_df.iloc[[selected_point_idx]]])

df.iloc[train_df.index]=data_df.loc[train_df.index,"labels"].apply(lambda x: clr_update[x])
plt.scatter(data_df['x'], data_df['y'], s=20, color=df, alpha=0.7, marker='o')
plt.draw()
plt.waitforbuttonpress(-2)
print("***** greedy policy *****")
print("num_queries : ", num_queries)
print("found target: ",train_df['labels'].sum())
print("train_df : \n", train_df)
print()

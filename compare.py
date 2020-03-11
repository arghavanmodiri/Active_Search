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
from greedy_policy import greedy
from two_step_policy import twoStep
from ens_policy import ens
from mf_ens_policy import mfEns

pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', None)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(message)s')
logging.info("{}".format(datetime.now()))

 #loading Data: has x,y,labels
data_df = pd.read_csv("./data/toy_problem/toy_example_data_500.csv",index_col=0)

def main():
    num_queries = 30
    alpha = 0.1
    k = 50
    visual = False
    repetition = 30
    policies = ["greedy", "ens", "2-step"]
    cost1 = 0.7
    cost2 = 1
    p1 = 0.7
    p2 = 1
    # initialization
    train_df = pd.DataFrame()
    num_points = data_df.shape[0]
    
    #initial point
    positive_labels = data_df.loc[data_df['labels']==1]
    positive_labels_counts = positive_labels['labels'].sum()
    print(positive_labels_counts)
    

    # making probabilistic model
    [neighbours, distances] = knn_search(data_df[['x','y']], k)
    similarities = 1/distances
    weights = pd.DataFrame(0, index=range(num_points), columns=range(num_points))
    for i in range(num_points):
        weights.iloc[i,neighbours.iloc[i].tolist()] = similarities.iloc[i].tolist()

    pos_count_greedy = []
    pos_count_2step = []
    pos_count_ens = []
    pos_count_mf_ens = []
    first_train_point = []
    for idx in range(1):
        train_df = positive_labels.sample()
        #positive_labels.drop(train_df.index, inplace=True)
        logging.info("-------------------------------")
        logging.info(train_df)
        logging.info("-------------------------------")
        first_train_point.append(train_df.index.tolist()[0])
        logging.info("*************** MF ENS ***************")
        #logging.info(train_df)
        pos_count_mf_ens.append(mfEns(data_df, train_df, weights, alpha, visual, num_queries, cost1, cost2, p1, p2))
        logging.info("*************** 2-Step ***************")
        #logging.info(train_df)
        pos_count_2step.append(twoStep(data_df, train_df, weights, alpha, visual, num_queries))
        logging.info("*************** Greedy ***************")
        #logging.info(train_df)
        pos_count_greedy.append(greedy(data_df, train_df, weights, alpha, visual, num_queries))
        logging.info("***************  ENS   ***************")
        #logging.info(train_df)
        pos_count_ens.append(ens(data_df, train_df, weights, alpha, visual, num_queries))


    logging.info("first_train_point : {}".format(first_train_point))
    logging.info("pos_count_greedy : {}".format(pos_count_greedy))
    logging.info("pos_count_2step : {}".format(pos_count_2step))
    logging.info("pos_count_ens : {}".format(pos_count_ens))
    logging.info("pos_count_mf_ens : {}".format(pos_count_mf_ens))
    




if __name__ == "__main__":
    main()

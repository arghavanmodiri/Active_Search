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
#logging.info("{}".format(datetime.now()))

def mfEns(data_df, train_df, weights, alpha, visual, total_cost, cost1, cost2, p1, p2):
    
    num_points = data_df.shape[0]
    
    # for plotting
    clr = {0:'silver',1:'dimgrey'}
    clr_update = {0:'blue',1:'red'}
    df = data_df["labels"].apply(lambda x: clr[x])

    if visual:
        plt.ion()

    budget_left = total_cost
    query = 0

    while budget_left > 0:
        #print("*** query: ",query, " *** ",datetime.now())
        if budget_left < cost1:
            break
        logging.info("*** query: {} *** {}".format(query, datetime.now()))
        query += 1

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

        # **************ENS policy **************
        #scores = search_expected_utility(train_df['labels'], probs)
        scores1 = search_expected_utility(train_df['labels'], probs, p1)
        scores2 = search_expected_utility(train_df['labels'], probs, p2)
        #budget_left = num_queries - train_df.shape[0]

        for point in test_idx:
            temp1 = 0
            temp2 = 0
            #print("point ", point)
            for Y_x in [0, 1]:
                train_fake_df = pd.concat([train_df, data_df.iloc[[point]]])
                train_fake_df.loc[point, 'labels'] = Y_x
                positive_train_fake_idx = train_fake_df.loc[train_fake_df['labels']==1].index.values.tolist()
                negative_train_fake_idx = train_fake_df.loc[train_fake_df['labels']==0].index.values.tolist()
                test_fake_idx = unlabeled_selector(data_df, train_fake_df.index)
                next_probs = knn_model_prob(alpha, weights,
                                positive_train_fake_idx, negative_train_fake_idx, test_fake_idx)
                p_y_x_1 = probs.loc[point]
                p_y_x = ((1-p_y_x_1)*(1-Y_x) + p_y_x_1*Y_x)
                budget_left_fake1 = budget_left - cost1
                budget_left_fake2 = budget_left - cost2
                max_fake1_t1 = int(np.floor(budget_left_fake1/cost1))
                max_fake1_t2 = int(np.floor(budget_left_fake1/cost2))
                max_fake2_t1 = int(np.floor(budget_left_fake2/cost1))
                max_fake2_t2 = int(np.floor(budget_left_fake2/cost2))

                temp_app = -10000
                if budget_left_fake1 < cost1:
                    temp_app = 0
                for t1 in range(max_fake1_t1):
                    t2 = int(budget_left_fake1 - t1 * cost1)
                    if t2 < 0:
                        t2 = 0
                    next_batch = next_probs.nlargest(min(t1+t2, len(test_fake_idx)))
                    temp_sum = p2*next_batch[0:t2].sum() + p1*next_batch[t2:t1+t2].sum()
                    if temp_sum > temp_app:
                        temp_app = temp_sum

                # here aI should add a loop to calculate t1 and t2
                temp1 += p_y_x * temp_app
                temp_app = -10000
                if budget_left_fake2 < cost1:
                    temp_app = 0
                for t1 in range(max_fake2_t1):
                    t2 = int(budget_left_fake2 - t1 * cost1)
                    if t2 < 0:
                        t2 = 0
                    next_batch = next_probs.nlargest(min(t1+t2, len(test_fake_idx)))
                    temp_sum = p2*next_batch[0:t2].sum() + p1*next_batch[t2:t1+t2].sum()
                    if temp_sum > temp_app:
                        temp_app = temp_sum

                # here aI should add a loop to calculate t1 and t2
                temp2 += p_y_x * temp_app
            scores1.loc[point] += temp1
            scores2.loc[point] += temp2

            scores = pd.concat([scores1,scores2], axis =1).max(axis=1)

        selected_point_idx = argmax(scores)
        # logging.info(selected_point_idx)
        # logging.info(scores.loc[selected_point_idx])
        # logging.info(scores1.loc[selected_point_idx])
        # logging.info(scores2.loc[selected_point_idx])

        #Update observed data
        train_df = pd.concat([train_df, data_df.iloc[[selected_point_idx]]])

        if scores1.loc[selected_point_idx] == scores.loc[selected_point_idx]:
            budget_left -= cost1
            logging.info("Oracle1")
            #Ask Oracle
            label = lookup_oracle(data_df, selected_point_idx)
            if np.random.uniform() > p1:
                train_df.loc[selected_point_idx]['labels'] = np.logical_xor(train_df.loc[selected_point_idx]['labels'],1).astype(int)
        else:
            budget_left -= cost2
            logging.info("Oracle2")
            #Ask Oracle
            label = lookup_oracle(data_df, selected_point_idx)
            if np.random.uniform() > p2:
                train_df.loc[selected_point_idx]['labels'] = np.logical_xor(train_df.loc[selected_point_idx]['labels'],1).astype(int)

    if visual:
        logging.info("*** save it ***")
        df.iloc[train_df.index]=data_df.loc[train_df.index,"labels"].apply(lambda x: clr_update[x])
        plt.scatter(data_df['x'], data_df['y'], s=20, color=df, alpha=0.7, marker='o')
        plt.draw()
        plt.waitforbuttonpress(-2)

    '''
    print("***** mfens policy *****")
    print("query : ", query)
    print("budget_left : ", budget_left)
    print("found target: ",train_df['labels'].sum())
    print("real found target: ",data_df.loc[train_df.index]['labels'].sum())
    print("train_df : \n", train_df)
    print()
    '''
    pos_targets_count = data_df.loc[train_df.index]['labels'].sum()

    return pos_targets_count

import pandas as pd
import numpy as np
import torch
import logging
from datetime import datetime
import matplotlib.pyplot as plt

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(message)s')

def knn_model_prob(alpha, data, weights):
    #written just for binary class
    temp_p = alpha + torch.sum(weights[:,data[:,-1]==1], 1)
    temp_n = (1.0 - alpha) + torch.sum(weights[:,data[:,-1]==0], 1)
    #normalize probabilities
    probs = temp_p / (temp_p + temp_n)

    #add a new column to show the index of each probs
    probs_p = torch.stack((data[:,0], probs),dim=1)[data[:,-1]==-1,:]
    #probs_p = alpha + torch.sum(weights[data[:,-1]==-1,:][:,data[:,-1]==1], 1)
    #probs_n = (1.0 - alpha) + torch.sum(weights[data[:,-1]==-1,:][:,data[:,-1]==0], 1)

    #normalize probabilities
    #probs = probs_p / (probs_p + probs_n)
    return probs_p


def mfEns(pool, data, true_labels, weights, alpha, visual, total_cost, cost1, cost2, p1, p2):
    #assume data has all the positions plus label 1 or 0 for train data and -1
    #for unknown data
    false_positive = 0
    true_positive = 0
    false_negative = 0

    visual = False
    if visual:
        plt.ion()
        # for plotting
        clr = {0:'silver',1:'dimgrey'}
        clr_update = {0:'blue',1:'red'}
        #color_points = data[:, -1].apply(lambda x: clr[x])
        color_points = data[:, -1].clone().detach().numpy().astype('str')
        color_points = true_labels.clone().detach().int().numpy().astype('str')
        color_points[color_points == '0'] = clr[0]
        color_points[color_points == '1'] = clr[1]
        logging.info(color_points)

    num_points = len(data)

    #For plotting --> Add later
    #test = data.detach().clone()
    budget_left = total_cost
    query = 0

    #test = [1,2,3,4,5,6,7,8,9,10]
    #logging.info(test)
    #pool.starmap(test_one_loop, [*zip([test]*10, [j for j in range(10)])])
    #logging.info(test)
    #logging.info(scores1)
    #logging.info("***")
    #logging.info(scores2)
    #logging.info("*************************************************")

    while budget_left >= cost1:
        #if budget_left < cost1:
        #    break
        query += 1
        logging.info("*** query: {} *** {}".format(query, datetime.now()))

        if visual:
            plt.scatter(data[:,1], data[:,2], s=20, color=color_points, alpha=0.7, marker='o')
            #df.iloc[train_df.index]=data_df.loc[train_df.index,"labels"].apply(lambda x: clr_update[x])
            plt.draw()
            plt.waitforbuttonpress(-2)
            plt.clf()

        #probsitive_train = data[data[:,-1]==1]
        #negative_train= data[data[:,-1]==0]

        #Calculate score for all points in test set
        probs = knn_model_prob(alpha, data, weights)
        #scores1 =  probs * p1
        #scores1[:,0] = (scores1[:,0]/p1).long()
        #scores2 =  probs * p2
        #scores2[:,0] = (scores2[:,0]/p2).long()
        scores1 = probs.detach().clone()
        scores1[:,1] = scores1[:,1] * p1
        scores2 = probs.detach().clone()
        scores2[:,1] = scores2[:,1] * p2

        budget_left_fake1 = budget_left - cost1
        budget_left_fake2 = budget_left - cost2
        max_fake1_t1 = int(np.floor(budget_left_fake1/cost1))
        max_fake1_t2 = int(np.floor(budget_left_fake1/cost2))
        max_fake2_t1 = int(np.floor(budget_left_fake2/cost1))
        max_fake2_t2 = int(np.floor(budget_left_fake2/cost2))
        logging.info("******************\n{}\n{}    {}\n{}  {}\n\n".format(budget_left,max_fake1_t1,max_fake1_t2,max_fake2_t1,max_fake2_t2))
        if budget_left_fake1 >= cost1:

            logging.info("{}".format(datetime.now()))
            points_count = len(data[data[:,-1].int()==-1])
            points = [point for point in data[data[:,-1].int()==-1]]
            #for point in data[data[:,-1]==-1]:
            inputs = [*zip([alpha]*points_count,
                [data]*points_count,
                [probs]*points_count,
                [weights]*points_count,
                [budget_left_fake1]*points_count,
                [budget_left_fake2]*points_count,
                [max_fake1_t2]*points_count,
                [cost2]*points_count,
                [cost1]*points_count,
                [p1]*points_count, [p2]*points_count,
                [scores1]*points_count,
                [scores2]*points_count,
                points)]
            #one_point_utility(alpha, data, probs, weights, budget_left_fake1, budget_left_fake2,
            #        max_fake1_t2, cost2, cost1, p1, p2, scores1, scores2, point)
            #temp1 = scores1.detach().clone()
            results = pool.starmap(one_point_utility, inputs)
            
            for_score1 = [el[0] for el in results]
            for_score2 = [el[1] for el in results]
            pattern1_lst = [el[2] for el in results]
            pattern2_lst = [el[3] for el in results]
            torch.stack(for_score1, out=scores1)
            torch.stack(for_score2, out=scores2)
            #logging.info("******************\n{}\n{}\n\n".format(scores1.size(),scores2.size()))

        selected_point1 = torch.topk(scores1[:,-1], 1, dim=0)
        selected_point2 = torch.topk(scores2[:,-1], 1, dim=0)
        logging.info("scores: {} and {}".format(selected_point1[0].item(),selected_point2[0].item()))
        if selected_point1[0] > selected_point2[0]:
            budget_left -= cost1
            logging.info("Oracle1")
            #Ask Oracle
            index = int(scores1[selected_point1[1]][0][0].item())
            label = true_labels[index].item()
            if np.random.uniform() > p1:
                data[index, -1] = np.logical_xor(label,1).astype(int)
            else:
                data[index, -1] = int(label)
            logging.info("---------------------------------------Oracle1: {}".format(pattern1_lst[selected_point1[1]]))
        else:
            budget_left -= cost2
            logging.info("Oracle2")
            #Ask Oracle
            index = int(scores2[selected_point2[1]][0][0].item())
            label = true_labels[index].item()
            if np.random.uniform() > p2:
                data[index, -1] = np.logical_xor(label,1).astype(int)
            else:
                data[index, -1] = int(label)
            logging.info("---------------------------------------Oracle2: {}".format(pattern2_lst[selected_point2[1]]))

        logging.info("true label: {}, queried label: {}".format(label,data[index, -1]))
        #Update data
        if label == 1 and data[index, -1] == 1:
            true_positive += 1
        else:
            false_negative += label
            false_positive  += data[index, -1]

        if visual:
            color_points[index] = clr_update[data[index, -1].int().item()]

    #Count the number of success in finding +1
    #Count False Positive Rate

    logging.info("***************************")
    logging.info("true_positive : {}\nfalse_negative : {}\nfalse_positive : {}".format(
            true_positive, false_negative, false_positive))

    if visual:
        logging.info("*** save it ***")
        #df.iloc[train_df.index]=data_df.loc[train_df.index,"labels"].apply(lambda x: clr_update[x])
        plt.scatter(data[:,1], data[:,2], s=20, color=color_points, alpha=0.7, marker='o')
        plt.draw()
        plt.waitforbuttonpress(-2)



def one_point_utility(alpha, data, probs, weights, budget_left_fake1, budget_left_fake2,
                    max_fake1_t2, cost2, cost1, p1, p2, scores1, scores2, point):
    temp1 = 0
    temp2 = 0
    pattern2 = ''
    pattern1 = ''
    for Y_x in [0, 1]:
        #logging.info("point {}".format(point))
        data_fake = data.detach().clone()
        data_fake[data_fake[:,0]==point[0], -1] = Y_x
        next_probs = knn_model_prob(alpha, data_fake, weights)
        p_y_x_1 = probs[probs[:,0]==point[0], -1]
        p_y_x = ((1-p_y_x_1)*(1-Y_x) + p_y_x_1*Y_x)

        #temp, we know that max_fake1_t2>max_fake2_t2
        temp_app1 = -10000
        temp_app2 = -10000
        for t2 in range(max_fake1_t2+1):
            fake1_t1 = int(np.floor((budget_left_fake1 - t2 * cost2)/cost1))
            fake2_t1 = int(np.floor((budget_left_fake2 - t2 * cost2)/cost1))
            fake1_t2 = t2
            fake2_t2 = t2
            if fake2_t1 < 0:
                fake2_t1 = 0
                fake2_t2 = 0
            nlargest1 = min(fake1_t1+fake1_t2, len(data_fake[data_fake[:,-1]==-1]))
            nlargest2 = min(fake2_t1+fake2_t2, len(data_fake[data_fake[:,-1]==-1]))
            next_batch = torch.topk(next_probs[:,-1], max(nlargest1,nlargest2), dim=0)
            idxs = next_probs[next_batch[1]][:,0]

            temp_sum1 = p2*torch.sum(next_batch[0][0:fake1_t2]) + p1*torch.sum(next_batch[0][fake1_t2:fake1_t1+fake1_t2])
            temp_sum2 = p2*torch.sum(next_batch[0][0:fake2_t2]) + p1*torch.sum(next_batch[0][fake2_t2:fake2_t1+fake2_t2])
            if temp_sum1 > temp_app1:
                temp_app1 = temp_sum1
                pattern1 = "{}-{}".format(fake1_t1, fake1_t2)
            if temp_sum2 > temp_app2:
                temp_app2 = temp_sum2
                pattern2 = "{}-{}".format(fake2_t1, fake2_t2)
        temp2 += p_y_x * temp_app1
        temp1 += p_y_x * temp_app2
    #logging.info("scores1 {}\n {}\n".format(scores1[:,0]==point[0], point))
    #scores2[scores2[:,0]==point[0], -1] += temp2
    #scores1[scores1[:,0]==point[0], -1] += temp1
    new_score1_for_point = torch.zeros_like(scores1[0, :])
    new_score2_for_point = torch.zeros_like(scores1[0, :])
    new_score1_for_point[-1] = scores1[scores1[:,0]==point[0], -1] + temp1
    new_score2_for_point[-1] = scores2[scores2[:,0]==point[0], -1] + temp2
    logging.info("{}-{}  {}-{}".format(temp1[-1],temp2[-1],new_score1_for_point[-1], new_score2_for_point[-1]))
    new_score1_for_point[0] = point[0]
    new_score2_for_point[0] = point[0]
    if len(scores1[scores1[:,0]==point[0], :])==0:
        logging.info(point[0])
        logging.info(scores1[:,0])
        #logging.info("scores1 {}\n {}\n{}\n\n".format(scores1[scores1[:,0]==point[0], :], point,scores1))
    #print(next_probs.size())

    return [new_score1_for_point, new_score2_for_point, pattern1, pattern2]



def test_one_loop(test, i):
    test[i] = 0
    return test[i]
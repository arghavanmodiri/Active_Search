import pandas as pd
import numpy as np
import torch
import logging
from datetime import datetime
from label_oracles.lookup_oracle import lookup_oracle

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


def mfEns(data, true_labels, weights, alpha, visual, total_cost, cost1, cost2, p1, p2):
    #assume data has all the positions plus label 1 or 0 for train data and -1
    #for unknown data
    false_positive = 0
    true_positive = 0
    false_negative = 0

    num_points = len(data)

    #For plotting --> Add later
    #test = data.detach().clone()
    budget_left = total_cost
    query = 0

    while budget_left >= cost1:
        #if budget_left < cost1:
        #    break
        query += 1
        logging.info("*** query: {} *** {}".format(query, datetime.now()))

        #probsitive_train = data[data[:,-1]==1]
        #negative_train= data[data[:,-1]==0]

        #Calculate score for all points in test set
        probs = knn_model_prob(alpha, data, weights)

        scores1 =  probs * p1
        scores1[:,0] = scores1[:,0]/p1
        scores2 =  probs * p2
        scores2[:,0] = scores2[:,0]/p2

        budget_left_fake1 = budget_left - cost1
        budget_left_fake2 = budget_left - cost2
        max_fake1_t1 = int(np.floor(budget_left_fake1/cost1))
        max_fake1_t2 = int(np.floor(budget_left_fake1/cost2))
        max_fake2_t1 = int(np.floor(budget_left_fake2/cost1))
        max_fake2_t2 = int(np.floor(budget_left_fake2/cost2))
        
        if budget_left_fake1 >= cost1:
            for point in data[data[:,-1]==-1]:
                temp1 = 0
                temp2 = 0
                for Y_x in [0, 1]:
                    data_fake = data.detach().clone()
                    data_fake[data_fake[:,0]==point[0], -1] = Y_x
                    next_probs = knn_model_prob(alpha, data_fake, weights)
                    p_y_x_1 = probs[probs[:,0]==point[0], -1]
                    p_y_x = ((1-p_y_x_1)*(1-Y_x) + p_y_x_1*Y_x)
                    '''
                    temp_app = -10000
                    for t2 in range(max_fake1_t2):
                        t1 = `int((budget_left_fake1 - t2 * cost2)/cost1)
                        if t1 < 0:
                            t1 = 0
                        nlargest = min(t1+t2, len(data_fake[data_fake[:,-1]==-1]))
                        next_batch = torch.topk(next_probs[:,-1], nlargest, dim=0)
                        #next_batch = next_probs.nlargest(min(t1+t2, len(test_fake_idx)))
                        idxs = next_probs[next_batch[1]][:,0]
                        temp_sum = p2*torch.sum(next_batch[0][0:t2]) + p1*torch.sum(next_batch[0][t2:t1+t2])
                        if temp_sum > temp_app:
                            temp_app = temp_sum
                    temp1 += p_y_x * temp_app
                    scores1[scores1[:,0]==point[0], -1] += temp1

                    temp_app = -10000
                    for t2 in range(max_fake2_t2):
                        t1 = int((budget_left_fake2 - t2 * cost2)/cost1)
                        if t1 < 0:
                            t1 = 0
                        nlargest = min(t1+t2, len(data_fake[data_fake[:,-1]==-1]))
                        next_batch = torch.topk(next_probs[:,-1], nlargest, dim=0)
                        idxs = next_probs[next_batch[1]][:,0]
                        temp_sum = p2*torch.sum(next_batch[0][0:t2]) + p1*torch.sum(next_batch[0][t2:t1+t2])
                        if temp_sum > temp_app:
                            temp_app = temp_sum
                    temp2 += p_y_x * temp_app
                    scores2[scores2[:,0]==point[0], -1] += temp2'''

                    #temp, we know that max_fake1_t2>max_fake2_t2
                    temp_app1 = -10000
                    temp_app2 = -10000
                    for t2 in range(max_fake1_t2+1):
                        t1_fake1 = int(np.floor((budget_left_fake1 - t2 * cost2)/cost1))
                        t1_fake2 = int(np.floor((budget_left_fake2 - t2 * cost2)/cost1))
                        t2_fake1 = t2
                        t2_fake2 = t2
                        if t1_fake2 < 0:
                            t1_fake2 = 0
                            t2_fake2 = 0
                        nlargest1 = min(t1_fake1+t2_fake1, len(data_fake[data_fake[:,-1]==-1]))
                        nlargest2 = min(t1_fake2+t2_fake2, len(data_fake[data_fake[:,-1]==-1]))
                        next_batch = torch.topk(next_probs[:,-1], max(nlargest1,nlargest2), dim=0)
                        idxs = next_probs[next_batch[1]][:,0]
                        temp_sum1 = p2*torch.sum(next_batch[0][0:t2_fake1]) + p1*torch.sum(next_batch[0][t2_fake1:t1_fake1+t2_fake1])
                        temp_sum2 = p2*torch.sum(next_batch[0][0:t2_fake2]) + p1*torch.sum(next_batch[0][t2_fake2:t1_fake2+t2_fake2])
                        if temp_sum1 > temp_app1:
                            temp_app1 = temp_sum1
                        if temp_sum2 > temp_app2:
                            temp_app2 = temp_sum2
                    temp2 += p_y_x * temp_app1
                    scores2[scores2[:,0]==point[0], -1] += temp2
                    temp1 += p_y_x * temp_app2
                    scores1[scores1[:,0]==point[0], -1] += temp1
                    #print(next_probs.size())
            logging.info("******************\n{}\n{}\n\n".format(scores1,scores2))

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

        logging.info("true label: {}, queried label: {}".format(label,data[index, -1]))
        #Update data
        if label == 1 and data[index, -1] == 1:
            true_positive += 1
        else:
            false_negative += label
            false_positive  += data[index, -1]

    #Count the number of success in finding +1
    #Count False Positive Rate

    logging.info("***************************")
    logging.info("true_positive : {}\nfalse_negative : {}\nfalse_positive : {}".format(
            true_positive, false_negative, false_positive))






import random
import numpy as np
import pandas as pd
from pyGPs import cov


def generate_data(num_points, random_seed = None, save_to_file = False):

    random.seed(random_seed)

    #generate grid points
    # x , y = np.meshgrid(range(side_length), range(side_length))

    x = np.random.random(num_points)
    y = np.random.random(num_points)
    df_rand = pd.DataFrame(np.random.randint(4, size=num_points))
    df_rand[df_rand[0] != 0] = 1
    df_rand = np.logical_xor(df_rand,1).astype(int)

    points = np.concatenate((x,y)).reshape(2,num_points).T
    to_int_vectorized = np.vectorize(to_int)
    labels = to_int_vectorized(pow(x,2)+pow(y,2) < pow(0.25,2))
    labels += to_int_vectorized(pow(x-1,2)+pow(y,2) < pow(0.25,2))
    labels += to_int_vectorized(pow(x,2)+pow(y-1,2) < pow(0.25,2))
    labels += to_int_vectorized(pow(x-1,2)+pow(y-1,2) < pow(0.25,2))
    labels += to_int_vectorized(pow(x-0.5,2)+pow(y-0.5,2) < pow(0.125,2))

    labels = df_rand*pd.DataFrame(labels)

    if save_to_file:
        data_df = pd.DataFrame(points, columns=['x','y'])
        data_df["labels"] = labels
        data_df.to_csv("toy_example_data_{}.csv".format(num_points))

    return [points, labels]





def to_int(x):
    return np.int(x)

generate_data(10000,20,True)
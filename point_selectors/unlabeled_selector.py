
def unlabeled_selector(data_df, train_idx):
    '''UNLABELED_SELECTOR selects points not yet observed.'''
    '''
        data_df: DataFrame contains observed and non-observed data
        train_idx: list of the indices observed so far

        Output:
        test_idx: list of the indices that can be queried at this iteration
    '''

    test_idx = [x for x in list(data_df.index) if x not in train_idx]
    return test_idx


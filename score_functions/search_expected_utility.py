def search_expected_utility(observed_labels, probs, oracle_p = 1):
	#expected utility after adding x is simply u(D) + p(y = 1 | x, D)
	return past_utility(observed_labels) + probs * oracle_p


def past_utility(observed_labels):
	return sum(observed_labels)


def search_two_step_utility(data_df, train_df, test_idx, probs):
	temp = past_utility(observed_labels) + probs


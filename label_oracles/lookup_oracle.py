def lookup_oracle(data_df, query_idx):
	return data_df['labels'].iloc[query_idx]
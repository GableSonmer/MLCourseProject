"""
Models for the time series forecasting.

Input shape: (batch_size, seq_len[96], n_features[7])
Output shape: (batch_size, pred_len[336], n_output[7])
The [~] means the default value, in our dataset.
"""
MODEL_NAMES = ['lstm', 'transformer', 'segrnn', 'patchmixer']

import os
import h5py
import logging
import numpy as np
import pandas as pd
import math
import tensorflow as tf

from data import utils
from config import DATA_PATH, NUM_NODES

def get_soc_feature(dataset):
	n_neighbors = dataset['n_neighbors']

	# get urbanicity factors and calculates z-scores
	urb = utils.state_urbanicity()
	urb = (urb - np.mean(urb, axis=0)) / np.std(urb, axis=0)

	# get white evangelical percentages and calculate z-scores
	evangel = utils.state_white_evang_pct()
	evangel = (evangel - np.mean(evangel, axis=0)) / np.std(evangel, axis=0)

	# get educational attainment percentages and calculate z-scores
	edu = utils.state_education()
	edu = (edu - np.mean(edu, axis=0)) / np.std(edu, axis=0)

	# get prev election party percentages and calculate z-scores
	prev = utils.state_prev_results()
	prev = (prev - np.mean(prev, axis=0)) / np.std((prev), axis=0)

	# get state median ages and calculate z-scores
	age = utils.state_median_age()
	age = (age - np.mean(age, axis=0)) / np.std((age), axis=0)

	# get state median household incomes and calculate z-scores
	income = utils.state_median_income()
	income = (income - np.mean(income, axis=0)) / np.std((income), axis=0)

	# get state race distributions and calculate z-scores
	race = utils.state_races()
	race = (race - np.mean(race, axis=0)) / np.std((race), axis=0)

	# get correlation matrix
	corr, e_bi = utils.correlation_matrix(n_neighbors)

	# normalize distance matrix
	n = urb.shape[0]
	"""
	n = loc.shape[0]
	edge = np.zeros((n, n))
	for i in range(n):
		for j in range(n_neighbors):
			edge[e_in[i][j], i] = edge[i, e_out[i][j]] = 1
	corr[edge == 0] = np.inf

	values = dist.flatten()
	values = values[values != np.inf]
	dist_mean = np.mean(values)
	dist_std = np.std(values)
	dist = np.exp(-(dist - dist_mean) / dist_std)
	"""
	# merge features (pop size, urban density, evangelical pct, correlation values of n nearest states)
	features = []
	for i in range(n):
		f = np.concatenate([urb[i], evangel[i], edu[i], prev[i], age[i], income[i], race[i], corr[e_bi[i],i]])
		features.append(f)
	features = np.stack(features)
	return features, (corr, e_bi)

def series_to_supervised(data, n_in=3, n_out=3, dropnan=True):
    #n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = pd.concat(cols, axis=1)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values

def dataloader(dataset):
	data = pd.read_csv(os.path.join(DATA_PATH, 'pollavg.csv'))
	#data = data.drop(['DC'], axis=1)
	# split df, one for each candidate
	half_idx = int(len(data) /2)
	b_data = data[:half_idx]
	t_data = data[half_idx:]
	#b_data = b_data.iloc[::-1] #earliest date first
	#t_data = t_data.iloc[::-1]

	#b_data = b_data.dropna(thresh=43)
	#t_data =t_data.dropna(thresh=43)

	b_data = utils.fill_missing(b_data)
	t_data = utils.fill_missing(t_data)

	n_timestamp = b_data.shape[0]

	num_train = int(n_timestamp * dataset['train_prop'])
	num_eval = int(n_timestamp * dataset['eval_prop'])
	num_test = n_timestamp - num_train - num_eval

	b_train = b_data[:num_train].copy()
	t_train = t_data[:num_train].copy()
	if num_eval != 0:
		b_eval = b_data[num_train: num_train + num_eval].copy()
		t_eval = t_data[num_train: num_train + num_eval].copy()
		b_test = b_data[-num_test:].copy()
		t_test = t_data[-num_test:].copy()
	else:
		b_eval = t_eval = None
		b_test = b_data[num_train:].copy()
		t_test = t_data[num_train:].copy()

	return b_train, t_train, b_eval, t_eval, b_test, t_test

def dataiter_all_sensors_seq2seq(b_df, t_df, scaler, setting, shuffle=True):
	dataset = setting['dataset']
	training = setting['training']
	#b_df_fill = utils.fill_missing(b_df) # fill before splitting the datasets
	#t_df_fill = utils.fill_missing(t_df)
	#print('b_df_fill', b_df_fill)

	b_df_fill, t_df_fill = scaler.transform(b_df, t_df)
	print('biden fill', b_df_fill)
	n_timestamp = b_df_fill.shape[0]

	data_list = [np.expand_dims(b_df_fill.values, axis=-1)] # allows arrays to be stored in each df 'cell' (df expanded in last axis and wrapped in array type)
	data_list.append(np.expand_dims(t_df_fill.values, axis=-1))

	# time in day
	"""
	time_idx = (df_fill.index.values - df_fill.index.values.astype('datetime64[D]')) / np.timedelta64(1, 'D')
	time_in_day = np.tile(time_idx, [1, NUM_NODES, 1]).transpose((2, 1, 0))
	data_list.append(time_in_day)
	"""
	""" ****look into implementing days count until election****
	# day in week (hot encoded 3D array of day in week associated w/ timestamp)
	day_in_week = np.zeros(shape=(n_timestamp, NUM_NODES, 7))
	day_in_week[np.arange(n_timestamp), :, df_fill.index.dayofweek] = 1
	data_list.append(day_in_week) # hot encoded array appended to each cell
	"""
	# temporal feature ()
	temporal_feature = np.concatenate(data_list, axis=-1) # concatenate arrays in each 'cell', last axis
	#print('temporal feature', temporal_feature.shape)
	soc_feature, _ = get_soc_feature(dataset)

	input_len = dataset['input_len']
	output_len = dataset['output_len']
	feature, data, mask, label  = [], [], [], []
	for i in range(n_timestamp - input_len - output_len + 1): # gathers all possible input + output sequences then appends and stacks them into data + label
		data.append(temporal_feature[i: i + input_len])

		# obtains rows of boolean values (1,0) where values are not nan/zero in label
		_mask = np.array(b_df.iloc[i + input_len: i + input_len + output_len] > 1e-5, dtype=np.float32)
		mask.append(_mask)

		label.append(temporal_feature[i + input_len: i + input_len + output_len])

		feature.append(soc_feature)

		if i % 20 == 0:
			logging.info('Processing %d timestamps', i)
			# if i > 0: break
	data = tf.convert_to_tensor(np.stack(data)) # (sequences, timesteps, nodes, candidate pcts)
	#print('data stacked', data.shape)
	label = tf.convert_to_tensor(np.stack(label))
	mask = tf.convert_to_tensor(np.expand_dims(np.stack(mask), axis=3))
	feature = tf.convert_to_tensor(np.stack(feature))

	data = tf.data.Dataset.from_tensor_slices(data)
	label = tf.data.Dataset.from_tensor_slices(label)
	feature = tf.data.Dataset.from_tensor_slices(feature)
	mask = tf.data.Dataset.from_tensor_slices(mask)

	logging.info('shape of feature: %s', feature)
	logging.info('shape of data: %s', data)
	logging.info('shape of mask: %s', mask)
	logging.info('shape of label: %s', label)

	data = data.batch(training['batch_size'])
	#print('batched data', data)
	label = label.batch(training['batch_size'])
	feature = feature.batch(training['batch_size'])
	mask = mask.batch(training['batch_size'])

	dataset = tf.data.Dataset.zip((feature, data, label, mask))
	batched_dataset = dataset.shuffle(buffer_size=1000) if shuffle else dataset
	#print('batched dataset', batched_dataset)
	return batched_dataset
	"""
	return DataLoader(
		ArrayDataset(feature, data, label),
		shuffle		= shuffle,
		batch_size	= training['batch_size'],
		num_workers	= 4,
		last_batch	= 'rollover',
	)
	"""

def dataloader_all_sensors_seq2seq(setting):
	b_train, t_train, b_eval, t_eval, b_test, t_test = dataloader(setting['dataset'])
	scaler = utils.Scaler(b_train, t_train)
	return dataiter_all_sensors_seq2seq(b_train, t_train, scaler, setting), \
		   dataiter_all_sensors_seq2seq(b_eval, t_eval, scaler, setting, shuffle=False), \
		   dataiter_all_sensors_seq2seq(b_test, t_test, scaler, setting, shuffle=False), \
		   scaler

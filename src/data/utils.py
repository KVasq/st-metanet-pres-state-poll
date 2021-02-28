import os
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf

from config import DATA_PATH

class Scaler:
	def __init__(self, bdata, tdata):
		self.biden_mean = np.mean(bdata)
		self.trump_mean = np.mean(tdata)
		self.biden_std = np.std(bdata)
		self.trump_std = np.std(tdata)

	def transform(self, bdata, tdata):
		#print('biden mean', self.biden_mean)
		#print('biden std', self.biden_std)
		#print('trump mean', self.trump_mean)
		#print('trump std', self.trump_std)

		return ((bdata - self.biden_mean) / self.biden_std), ((tdata - self.trump_mean) / self.trump_std)

	def inverse_transform(self, bdata, tdata):
		inverse_b = np.zeros(tf.shape(bdata))
		inverse_t = np.zeros(tf.shape(tdata))
		for i in range(tf.shape(bdata)[0]):
			inverse_b[i] = bdata[i] * self.biden_std[i] + self.biden_mean[i]
			inverse_t[i] = tdata[i] * self.trump_std[i] + self.trump_mean[i]
		return inverse_b, inverse_t

def state_index(): #retrieves the states list (ordered for correlation mat) and assigns index to state abbreviation
	with open(os.path.join(DATA_PATH, 'states.txt')) as f:
		states = f.read().strip().split(',')
	state_ids = {}
	for i, state in enumerate(states):
		state_ids[state] = i
	return state_ids

def correlation_matrix(n_neighbors): #creates a correlation matrix restricted to n nearest neighbors
	filename = os.path.join(DATA_PATH, 'correlation_matrix_%d.h5' % n_neighbors) #assigns a file to read/write matrix per n neighbors
	# if not os.path.exists(filename):
	state_idx = state_index()
	graph = pd.read_csv(os.path.join(DATA_PATH, 'state_correlation_matrix.csv'))

    # initialize correlation matrix
	n = len(state_idx)
	corr = np.zeros((n, n))
	corr[:] = np.inf

	"""
	cnt = 0
	for state_id, row in enumerate(graph.values):
        for idx, state in enumerate(state_idx):
			corr[state_id, idx] = row[idx]
			cnt += 1"""
	# print('# edges', cnt)

	e_bi = [] # bi-directional edges
	for i in range(n): # obtains the n nearest state ids (ids with highest correlation values)
		e_bi.append(np.argsort(graph.values[:, i])[-(n_neighbors + 1):])
	e_bi = np.array(e_bi, dtype=np.int32)

	f = h5py.File(filename, 'w')
	f.create_dataset('corr', data=graph.values)
	f.create_dataset('e_bi', data=e_bi)
	f.flush()
	f.close()

	f = h5py.File(filename, 'r')
	adj_mat = np.array(f['corr'])
	e_bi = np.array(f['e_bi'])
	f.close()
	return adj_mat, e_bi

def state_urbanicity():
	state_idx = state_index()
	state_urb = pd.read_csv(os.path.join(DATA_PATH, 'urbanicity_index.csv'))

	n = len(state_idx)
	urb = np.zeros((n, 2))
	for i in range(n):
		state = state_urb.values[i, 3]
		urb[state_idx[state], :] = state_urb.values[i, 1:3]
	return urb

def state_white_evang_pct():
	state_idx = state_index()
	state_evangel = pd.read_csv(os.path.join(DATA_PATH, 'white_evangel_pct.csv'))

	n = len(state_idx)
	evangel_pct = np.zeros((n, 1))
	for i in range(n):
		state = state_evangel.values[i, 0]
		evangel_pct[state_idx[state]] = state_evangel.values[i, 1]
	return evangel_pct

def fill_missing(data):
	data = data.copy()
	data[np.absolute(data) < 1e-8] = float('nan')
	data = data.fillna(method='pad')
	data = data.fillna(method='bfill')
	return data

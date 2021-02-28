import tensorflow as tf
import numpy as np

def tf_sum_exclude(x, axis):
	total = tf.reduce_sum(x, axis=[i for i in range(np.ndim(x)) if i != axis])
	return total

def tf_mean_exclude(x, axis):
	mean = tf.reduce_mean(x, axis=[i for i in range(np.ndim(x)) if i != axis])
	return mean

class Metric:
	def __init__(self, name):
		self.name = name
		self.cnt = None
		self.loss = None

	def reset(self):
		self.cnt = None
		self.loss = None

class RMSE(Metric):
	def __init__(self, scaler, name='rmse'):
		super(RMSE, self).__init__(name)
		self.scaler = scaler

	def update(self, data, label, mask):
		bdata = data[:,:,:,0]
		tdata = data[:,:,:,1]
		blabel = label[:,:,:,0]
		tlabel = label[:,:,:,1]
		bdata, tdata = self.scaler.inverse_transform(bdata, tdata)
		blabel, tlabel = self.scaler.inverse_transform(blabel, tlabel)
		data = tf.stack([bdata, tdata], axis=-1)
		#print('data', data)
		label = tf.stack([blabel, tlabel], axis=-1)
		_cnt = tf.reduce_sum(mask)
		_loss = tf.reduce_sum((data - label) ** 2 * mask)
		if self.cnt is None:
			self.cnt = 0
			self.loss = 0
		self.cnt += _cnt
		self.loss += _loss

	def get_value(self):
		print('RMSE loss', self.loss)
		print('RMSE cnt', self.cnt + 1e-8)
		return { self.name: tf.math.sqrt(self.loss / (self.cnt + 1e-8)) }

class MAE(Metric):
	def __init__(self, scaler, name='mae'):
		super(MAE, self).__init__(name)
		self.scaler = scaler

	def update(self, data, label, mask):

		bdata = data[:,:,:,0]
		tdata = data[:,:,:,1]
		#print('metric bdata', bdata)
		#print('metric tdata', tdata)
		blabel = label[:,:,:,0]
		tlabel = label[:,:,:,1]
		bdata, tdata = self.scaler.inverse_transform(bdata, tdata)
		blabel, tlabel = self.scaler.inverse_transform(blabel, tlabel)
		#print('bdata inversed', tf.shape(bdata))
		data = tf.stack([bdata, tdata], axis=-1)
		label = tf.stack([blabel, tlabel], axis=-1)
		#print('metric data', data)
		#print('metric label', label)
		#print('metric mask', mask)
		_cnt = tf.reduce_sum(mask)
		_loss = tf.reduce_sum(tf.abs(data - label) * mask)
		if self.cnt is None:
			self.cnt = _cnt
			self.loss = _loss
		else:
			self.cnt += _cnt
			self.loss += _loss

	def get_value(self):
		print('MAE loss', self.loss)
		print('MAE cnt', self.cnt + 1e-8)
		return { self.name: self.loss / (self.cnt + 1e-8) }

class IndexRMSE(Metric):
	def __init__(self, scaler, indices, name='rmse-index'):
		super(IndexRMSE, self).__init__(name)
		self.scaler = scaler
		self.indices = indices

	def update(self, data, label, mask):
		bdata = data[:,:,:,0]
		tdata = data[:,:,:,1]
		blabel = label[:,:,:,0]
		tlabel = label[:,:,:,1]
		bdata, tdata = self.scaler.inverse_transform(bdata, tdata)
		blabel, tlabel = self.scaler.inverse_transform(blabel, tlabel)
		data = tf.stack([bdata, tdata], axis=-1)
		label = tf.stack([blabel, tlabel], axis=-1)

		_cnt = tf_sum_exclude(mask, axis=2)
		_loss = tf_sum_exclude((data - label) ** 2 * mask, axis=2)
		if self.cnt is None:
			self.cnt = _cnt
			self.loss = _loss
		else:
			self.cnt += _cnt
			self.loss += _loss

	def get_value(self):
		#print('IndexRMSE indices', self.indices)
		print('IndexRMSE loss', self.loss)
		print('IndexRMSE cnt', self.cnt + 1e-8)
		return { self.name: tf.gather(tf.math.sqrt((self.loss / (self.cnt + 1e-8))),[self.indices]) }

class IndexMAE(Metric):
	def __init__(self, scaler, indices, name='mae-index'):
		super(IndexMAE, self).__init__(name)
		self.scaler = scaler
		self.indices = indices

	def update(self, data, label, mask):
		bdata = data[:,:,:,0]
		tdata = data[:,:,:,1]
		blabel = label[:,:,:,0]
		tlabel = label[:,:,:,1]
		bdata, tdata = self.scaler.inverse_transform(bdata, tdata)
		blabel, tlabel = self.scaler.inverse_transform(blabel, tlabel)
		data = tf.stack([bdata, tdata], axis=-1)
		label = tf.stack([blabel, tlabel], axis=-1)

		_cnt = tf_sum_exclude(mask, axis=2)
		_loss = tf_sum_exclude(tf.abs(data - label) * mask, axis=2)
		if self.cnt is None:
			self.cnt = _cnt
			self.loss = _loss
		else:
			self.cnt += _cnt
			self.loss += _loss

	def get_value(self):
		#print('IndexMAE indices', self.indices)
		print('IndexMAE loss', self.loss)
		print('IndexMAE cnt', self.cnt + 1e-8)
		#print('IndexMAE return', tf.gather(tf.math.divide(self.loss, (self.cnt + 1e-8)),[self.indices]))
		return { self.name: tf.gather(tf.math.divide(self.loss, (self.cnt + 1e-8)),[self.indices]) }

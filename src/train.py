import os
import math
import yaml
import random
import logging
logging.basicConfig(level=logging.INFO)

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)

import numpy as np
np.set_printoptions(formatter={'float': '{: 0.2f}'.format})


import data.dataloader
import config
from config import PARAM_PATH
from helper.callback import Speedometer, Logger
from helper.metric import MAE, RMSE, IndexMAE, IndexRMSE
import model

class ModelTrainer:
	def __init__(self, net, learning_schedule, clip_gradient, logger, ctx):
		#super(ModelTrainer, self).__init__()
		self.net = net
		self.learning_schedule = learning_schedule
		self.clip_gradient = clip_gradient
		self.logger = logger
		self.ctx = ctx
		self.optimizer = tf.keras.optimizers.Adam(
				learning_rate	= learning_schedule)
		self.net.compile(self.optimizer)

	"""
	def train_step(self, batch_size): # train step with gradient clipping
		# self.trainer.allreduce_grads()

		for param in self.net.trainable_weights:
			if param.grad_req == 'write':
				grads += param.list_grad()

		import math
		tf.clip_by_norm(grads, self.clip_gradient * math.sqrt(len(self.ctx)))
		# self.trainer.update(batch_size, ignore_stale_grad=True)
"""
	def process_data(self, epoch, dataset, metrics=None, is_training=True, title='[TRAIN]'):
		speedometer = Speedometer(title, epoch, frequent=50)
		speedometer.reset()
		if metrics is not None:
			for metric in metrics:
				metric.reset()
		#print('callbacks and metrics set')
		dataset = dataset.enumerate()
		#print('dataset enumerated')
		for nbatch, batch_data in dataset.as_numpy_iterator():
			# inputs = [gluon.utils.split_and_load(x, self.ctx) for x in batch_data]
			feat_, data_, label_, mask_ = batch_data
			#print('label_', label_)
			if is_training:
				self.net.decoder.global_steps += 1.0
				#print('calculating loss')

				with tf.GradientTape(persistent=True) as tape:
					outputs = [self.net(feat_, data_, label_, mask_, is_training=is_training)] # loss, prediction, label, mask

				#print('loss calculated')
				gradients = []
				#print('weights', self.net.trainable_weights)
				#print('outputs', outputs)
				for out, w in zip(outputs, self.net.trainable_weights):
					print('nbatch', nbatch)
					#if len(tf.shape(gradients)) < 2:
					#	gradients = tf.reshape(gradients, (0, 2, 96))
					#print('w', w)
					#print('out', out[0])
					#print('gradients', gradients)
					gradients += tape.gradient(out[0], self.net.trainable_weights) # out[0] is loss array
					# out[0].backward()
				#print('weights', self.net.trainable_weights)
				#print('gradients after', gradients)
				# self.train_step(batch_data[0].shape[0], grads)
				grads = tf.clip_by_global_norm(gradients, self.clip_gradient)
				"""
				print('grads size', len(grads))
				for g in grads[0]:
					print(tf.shape(g))

				print('weight shapes')
				for w in self.net.trainable_weights:
					print(tf.shape(w))
					"""
				self.optimizer.apply_gradients(zip(grads[0], self.net.trainable_weights))
				#print('trainable weights', self.net.trainable_weights)
			else:
				#print('calculating loss alt')
				outputs = [self.net(feat_, data_, label_, mask_, False)]
			#print('net output shape', tf.shape(outputs))
			if metrics is not None:
				for metric in metrics:
					for out in outputs:
						_d, _l, _m = out[1]
						metric.update(_d, _l, _m) #[output, label, mask]
			speedometer.log_metrics(nbatch + 1, metrics)

		speedometer.finish(metrics)

	def fit(self, begin_epoch, num_epochs, train, eval, test, metrics=None):
		for epoch in range(begin_epoch, begin_epoch + num_epochs):
			if train is not None:
				#print('processing training data')
				self.process_data(epoch, train, metrics)

			if eval is not None:
				#print('processing eval data')
				self.process_data(epoch, eval, metrics, is_training=False, title='[EVAL]')
				if (train is not None) and (metrics is not None):
					self.logger.log(epoch, metrics)

			if test is not None:
				self.process_data(epoch, test, metrics, is_training=False, title='[TEST]')

			print('')

def main(args):
	with open(args.file, 'r') as f:
		settings = yaml.load(f)
	assert args.file[:-5].endswith(settings['model']['name']), \
		'The model name is not consistent! %s != %s' % (args.file[:-5], settings['model']['name'])

	tf.random.set_seed(settings['seed'])
	np.random.seed(settings['seed'])
	random.seed(settings['seed'])

	dataset_setting = settings['dataset']
	model_setting = settings['model']
	train_setting = settings['training']

	### set meta hiddens
	if 'meta_hiddens' in model_setting.keys():
		config.MODEL['meta_hiddens'] = model_setting['meta_hiddens']

	name = os.path.join(PARAM_PATH, model_setting['name'])
	model_type = getattr(model, model_setting['type'])
	net = model_type.net(settings)

	try:
		logger = Logger.load('%s.yaml' % name)
		net.load_weights('%s-%04d.h5' % (name, logger.best_epoch()))
		logger.set_net(net)
		#print('Successfully loading the model %s [epoch: %d]' % (model_setting['name'], logger.best_epoch()))

		num_params = 0
		for i in range(len(net.weights)):
			num_params += np.prod(net.weights[i].shape)
		#print('weights', net.weights)
		print('NUMBER OF PARAMS:', num_params)
	except:
		logger = Logger(name, net, train_setting['early_stop_metric'], train_setting['early_stop_epoch'])
		### net.build((settings['dataset']['input_dim'], model_setting['batch_size' ]))
		#print('Initialize the model')

	# net.hybridize()
	learning_scheduler=tf.keras.optimizers.schedules.ExponentialDecay(
					initial_learning_rate	= train_setting['lr'],
					decay_steps			= train_setting['lr_decay_step'] * len(args.gpus),
					decay_rate			= train_setting['lr_decay_factor'],
					staircase=True
					)

	model_trainer = ModelTrainer(
		net = net,
		learning_schedule = learning_scheduler,
		clip_gradient = train_setting['clip_gradient'],
		logger = logger,
		ctx = args.gpus
	)
	#print('ModelTrainer initialized') # retrieves batched dataset
	train, eval, test, scaler = getattr(data.dataloader, dataset_setting['dataloader'])(settings)
	#print('data loaded')
	#print('train',train)
	model_trainer.fit(
		begin_epoch = logger.best_epoch(),
		num_epochs	= args.epochs,
		train		= train,
		eval		= eval,
		test		= test,
		metrics		= [MAE(scaler), RMSE(scaler), IndexMAE(scaler, [0,1,2]), IndexRMSE(scaler, [0,1,2])],
	)
	#print('ModelTrainer fit finished 1')
	net.load_weights('%s-%04d.params' % (name, logger.best_epoch()))
	model_trainer.fit(
		begin_epoch	= 0,
		num_epochs	= 1,
		train		= None,
		eval		= eval,
		test		= test,
		metrics		= [MAE(scaler), RMSE(scaler), IndexMAE(scaler, [2,5,11]), IndexRMSE(scaler, [2,5,11])]
	)
	#print('ModelTrainer fit finished 2')
if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--file', type=str)
	parser.add_argument('--epochs', type=int)
	parser.add_argument('--gpus', type=str)
	args = parser.parse_args()

	args.gpus = [tf.config.list_physical_devices('GPU')[0]]
	main(args)

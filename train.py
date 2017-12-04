import os, glob, re, signal, sys, argparse, threading, time, random, h5py
from random import shuffle
import tensorflow as tf
import numpy as np
from PIL import Image
import scipy.io

from model import model
from psnr import psnr
from utils import dataset
import hyper

hyper_param = hyper.hyper

parser = argparse.ArgumentParser()
parser.add_argument("--model_path")
args = parser.parse_args()
model_path = args.model_path


def main():
	# define input 
	if not hyper_param['use_queue_loading']:
		# do not use asynchronous data loading
		print "not use queue loading, just sequential loading ...\n"

		train_input = tf.placeholder(
			tf.float32, 
			shape=(hyper_param['batch_size'], hyper_param['size_input'], hyper_param['size_input'],1)
			)
		train_label = tf.placeholder(
			tf.float32, 
			shape=(hyper_param['batch_size'], hyper_param['size_input'], hyper_param['size_input'],1)
			)
	else:
		# use asynchronous data loading
		print "use queue loading..."

		train_input_single = tf.placeholder(
			tf.float32, 
			shape=(hyper_param['size_input'], hyper_param['size_input'],1)
			)
		train_label_single = tf.placeholder(
			tf.float32, 
			shape=(hyper_param['size_input'], hyper_param['size_input'],1)
			)
		q = tf.FIFOQueue(
			10000,
			[tf.float32,tf.float32], 
			[[hyper_param['size_input'], hyper_param['size_input'],1],[hyper_param['size_input'], hyper_param['size_input'],1]]
			)
		enqueue_op = q.enqueue([train_input_single,train_label_single])

		train_input, train_label = q.dequeue_many(hyper_param['batch_size'])

	# the model and loss
	shared_model = tf.make_template('shared_model', model)
	train_output, weights = shared_model(train_input)
	loss = tf.reduce_sum(tf.nn.l2_loss(tf.substract(train_output, train_label)))

	# normlization weight.
	for w in weights:
		loss += tf.nn.l2_loss(w)*1e-4

	# record loss
	tf.summary.scalar("loss", loss)

	# training step
	global_step = tf.Variable(0, trainable=False)
	learning_rate = tf.train.exponential_decay(
		hyper_param['base_lr'], 
		global_step*hyper_param['batch_size'],
		)


if __name__ == '__main__':
	main()
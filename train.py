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

image_size = hyper_param['size_input']
batch_size = hyper_param['batch_size']
train_path = hyper_param['data_path']
base_learning_rate = hyper_param['base_lr']
lr_step_size = hyper_param['lr_step_size']
lr_decay = hyper_param['lr_rate']
max_epoch = hyper_param['max_epoch']

def main():
	# define dataset object
	train_set = dataset(train_path, batch_size)
	# define input 
	train_input = tf.placeholder(
		tf.float32, 
		shape=(image_size,image_size,batch_size)
		)
	train_label = tf.placeholder(
		tf.float32, 
		shape=(image_size,image_size,batch_size)
		)

	# the model and loss
	shared_model = tf.make_template('shared_model', model)
	train_output, weights = shared_model(train_input)
	loss = tf.reduce_sum(tf.nn.l2_loss(tf.substract(train_output, train_label)))

	# normlization weight.
	for w in weights:
		loss += tf.nn.l2_loss(w)*1e-4

	# record loss
	tf.summary.scalar("loss", loss)

	# training step and learning rate
	global_step = tf.Variable(0, trainable=False)
	learning_rate = tf.train.exponential_decay(
		base_learning_rate, 
		global_step*batch_size,
		train_set.instance_num*lr_step_size,
		lr_decay,
		staircase=True
		)
	tf.summary.scalar("learning_rate", learning_rate)

	# Optimizer
	optimizer = tf.train.AdamOptimizer(learning_rate)
	opt = optimizer.minimize(loss, global_step=global_step)

	saver = tf.train.Saver(weights, max_to_keep=0)
	config = tf.ConfigProto()

	# training
	with tf.Session(config=config) as sess:
		#TensorBoard open log with "tensorboard --logdir=logs"
		if not os.path.exists('logs'):
			os.mkdir('logs')
		merged = tf.summary.merge_all()
		file_writer = tf.summary.FileWriter('logs',sess.graph)

		# var initializaion
		tf.initialize_all_variables().run()

		if model_path:
			print "restore model..."
			saver.restore(sess,model_path)
			print "successfully restore previous model."

		# train
		for epoch in xrange(0,max_epoch):
			for step in range(train_set.instance_num//batch_size):
				data, label = train_set.next_batch()
				feed_dict = {train_input : data, train_label : label}
				_,l,output,lr,g_step = sess.run([opt, loss, train_output, learning_rate, global_step],feed_dict=feed_dict)
				print "[epoch %2.4f] loss %.4f\t lr %.5f" % (epoch+(float(step)*batch_size/train_set.instance_num), np.sum(l)/batch_size, lr)
				del data, label

			saver.save(sess, "./checkpoints/VDSR_const_clip_0.01_epoch_%03d.ckpt" % epoch ,global_step=global_step)


if __name__ == '__main__':
	main()
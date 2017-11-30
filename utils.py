import numpy as np
import h5py

class dataset(object):
	"""the data lavel"""
	def __init__(self, path, batch_size):
		super(dataset, self).__init__()
		self.path = path
		self.batch_size = batch_size
		with h5py.File(path,'a') as file:
			self.num_instance = file['data'].shape[-1]
		self.num_batch = self.num_instance / self.batch_size
		self.batch_array = np.random.permutation(self.num_batch)
		self.curr_bat_idx = 0

	def next_batch(self):
		with h5py.File(self.path, 'r') as file:
			#file = h5py.File(self.path, 'r')
			batch_idx = self.curr_bat_idx
			self.curr_bat_idx = (self.curr_bat_idx + 1) % num_batch
			batch_data = file['data'][:,:,0,batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
			batch_labels = file['labels'][:,:,0,batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
			return batch_data, batch_labels

	def next_instance(self):
		with h5py.File(self.path,'r') as file:
			#file = h5py.File(self.path,'r')
			idx = np.random.randint(self.num_instance)
			return file['data'][:,:,0,idx], file['labels'][:,:,0,idx]

		
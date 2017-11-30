import numpy as np
import os
import h5py

def store2hdf5(
	filename, 
	data, 
	labels, 
	count,
	create, 
	startloc, 
	chunksz):
	'''
	Save a chunk of data into hdf5 files
	*filename*: hdf5 file path
	*data*: data, W*H*C*N matrix of images should be normalized (e.g. to lie between 0 and 1) before
	*label*: a D*N matrix of labels
	*create*: 0-1 var specifies whether to create file newly or to append to existed file.
		if create=1 (create mode), startloc.data=[1 1 1 1], and startloc.lab=[1 1]; 
  		if create=0 (append mode), startloc.data=[1 1 1 K+1], and startloc.lab = [1 K+1]; where K is the current number of samples stored in the HDF
  	*count*: total instances count
  	*chunksz* (used only in create mode), specifies number of samples to be stored per chunk (see HDF5 documentation on chunking) for creating HDF5 files with unbounded maximum size - TLDR; higher chunk sizes allow faster read-write operations
	'''
	# judge if the format is right
	dat_dims = list(data.shape)
	lab_dims = list(labels.shape)
	num_samples = dat_dims[-1]
	assert lab_dims[-1] == num_samples, 'number of instances is not equal to the number of labels'

	dat_dims = dat_dims[:-1]
	lab_dims = lab_dims[:-1]

	if create:
		if os.path.exists(filename):
			print 'the file is already existed, replace it with new file.'
		hdf_file = h5py.File(filename, 'w')
		data_set = hdf_file.create_dataset(
			"data",
			tuple(dat_dims+[count]),
			maxshape=tuple(dat_dims+[None]),
			chunks=tuple(dat_dims+[chunksz]),
			dtype='float32')

		labels_set = hdf_file.create_dataset(
			"labels",
			tuple(lab_dims+[count]),
			maxshape=tuple(lab_dims+[None]),
			chunks=tuple(lab_dims+[chunksz]),
			dtype='float32')
	else:
		hdf_file = h5py.File(filename, 'r+')
		data_set = hdf_file['data']
		labels_set = hdf_file['labels']

	data_set[:,:,:,startloc['data']:startloc['data']+chunksz] = data
	labels_set[:,:,:,startloc['label']:startloc['label']+chunksz] = labels
	
	hdf_file.close()

	return startloc['data']+chunksz
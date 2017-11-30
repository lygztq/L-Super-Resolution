import numpy as np
import cv2
import os
import h5py

import hyper
import modcrop
from store2hdf5 import store2hdf5
from im2double import im2double

img_format = ['bmp','png','jpg']
root_path = os.path.abspath(os.curdir)
# mode 0:	train
# mode 1:	test
def main():
	for mode in range(2):
		# settings
		#folder = 'tmp'
		folder = 'Train'
		savepath = 'train.h5'
		size_input = hyper.hyper['size_input']
		size_label = hyper.hyper['size_label']
		scale = hyper.hyper['scale']
		stride = hyper.hyper['stride']
		chunk_size = hyper.hyper['chunksz']

		if mode == 1:
			folder = 'Test/Set5'
			savepath = 'test.h5'
			size_input = hyper.test_hyper['size_input']
			size_label = hyper.test_hyper['size_label']
			scale = hyper.test_hyper['scale']
			stride = hyper.test_hyper['stride']
			chunk_size = hyper.test_hyper['chunksz']

		print "-----------\config:\n"
		print "folder: %s\nsavepath: %s\nmode: %d\nsize_input: %d\nsize_label: %d\nscale: %d\nstride: %d" \
		 % (folder,savepath,mode,size_input,size_label,scale,stride)
		print "\n---------\n"

		# initialization
		data = np.zeros([size_input,size_input,1,1])
		label = np.zeros([size_label,size_label,1,1])
		padding = abs(size_input - size_label) / 2
		count = 0

		# generate data
		items = os.listdir(os.path.join(root_path,folder))
		filepaths = []
		for i in items:
			if os.path.isdir(i) or i[-3:] not in img_format:
				continue
			filepaths.append(i)

		for i in range(len(filepaths)):
			print "processing file: %s, \ntotal count: %d" % (filepaths[i],i)
			image = cv2.imread(os.path.join(folder,filepaths[i]))
			image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
			image = im2double(image[:,:,0])

			im_label = modcrop.modcrop(image,scale)
			[height, width] = im_label.shape
			#[width, height] = im_label.shape
			im_input = cv2.resize(
					cv2.resize(im_label, (width/scale,height/scale), interpolation=cv2.INTER_CUBIC),
					(width,height),
					interpolation=cv2.INTER_CUBIC
				)

			for x in range(0, height - size_input, stride):
				for y in range(0, width - size_input, stride):

					subim_input = im_input[x : x+size_input, y : y+size_input]
					subim_label = im_label[x+padding : x+padding+size_label, y+padding : y+padding+size_label]

					data[:,:,0,count] = subim_input
					label[:,:,0,count] = subim_label
					count+=1
					data = np.insert(data,count,values=np.zeros([size_input,size_input,1]),axis=3)
					label = np.insert(label,count,values=np.zeros([size_label,size_label,1]),axis=3)

		order = np.random.permutation(count)
		data = data[:,:,:,order]
		label = label[:,:,:,order]

		filename = os.path.join('dataset', savepath)

		# writing to HDF5
		first_flag = True
		total_count = 0
		batch_num = count/chunk_size

		for batch_No in range(batch_num):
			last_read = batch_No*chunk_size
			batch_data = data[:,:,0,last_read:last_read+chunk_size]
			batch_labels = label[:,:,0,last_read:last_read+chunk_size]

			startloc = {'data':total_count,'label':total_count}
			curr_dat_sz = store2hdf5(filename,batch_data,batch_labels,count,first_flag,startloc,chunk_size)
			first_flag = False
			total_count = curr_dat_sz

		if mode==0:
			print "\ntraining set size: %d\n" % total_count
		else:
			print "\ntest set size: %d\n" % total_count


if __name__ == '__main__':
	main()



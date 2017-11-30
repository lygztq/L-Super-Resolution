import numpy as np

import utils

def main():
	data_path = 'dataset/train.h5'
	dataset = utils.dataset(data_path,128)

	one_data,one_label = dataset.next_instance()
	print one_data
	print one_label
if __name__ == '__main__':
	main()
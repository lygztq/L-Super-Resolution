hyper = {
	'size_input' : 42,
	'size_label' : 42,
	'scale' : 3,
	'stride' : 14,
	'chunksz' : 128,
	'data_path' : './dataset/train.h5',
	'batch_size' : 128,
	'base_lr' : 0.0001,
	'lr_rate' : 0.1,
	'lr_step_size' : 120,
	'max_epoch' : 120,
	'use_queue_loading' : True
}

test_hyper = {
	'size_input' : 42,
	'size_label' : 42,
	'scale' : 3,
	'stride' : 21,
	'chunksz' : 2,
	'data_path' : './dataset/test.h5'
}
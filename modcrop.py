def modcrop(imgs, modulo):
	dim1 = imgs.shape[0] - (imgs.shape[0] % modulo)
	dim2 = imgs.shape[1] - (imgs.shape[1] % modulo)
	if len(imgs.shape) == 2:
		imgs = imgs[:dim1,:dim2]
	else:
		imgs = imgs[:dim1,:dim2,:]
	return imgs

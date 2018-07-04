import os
import zipfile
import scipy.ndimage
import scipy.misc
import numpy as np
import math
import log
import pickle

def extract_file(path):
	zipfiles = zipfile.ZipFile(path)
	zipfiles.extractall()
	zipfiles.close()


def extract_data():
	if not os.path.isfile('train.zip'):
		raise FileNotFoundError('train.zip')
	if not os.path.isfile('test.zip'):
		raise FileNotFoundError('test.zip')
	if not os.path.isdir('train'):
		extract_file('train.zip')
	if not os.path.isdir('test'):
		extract_file('test.zip')


def shuffle_list(x):
	arr = np.arange(len(x))
	np.random.shuffle(arr)
	return [x[index] for index in arr]


def shuffle_list_2(x, y):
	if len(x) != len(y):
		raise Exception('input size not match: %d & %d' % (len(x), len(y)))
	arr = np.arange(len(x))
	np.random.shuffle(arr)
	return x[arr], y[arr]


def get_all_images_in(category, need_filter, size = 224):
	files = os.listdir('train/%s' % category)
	images = []
	for i, file in enumerate(files):
		log.log("%-25s loading from file: %d%%" % (category, math.ceil((i+1)/len(files)*100)), end="\r")
		path = 'train/%s/%s' % (category, file)
		data = read_image(path, need_filter=need_filter, size = size)
		if data.shape == (size, size, 3):
			images.append(data)
	print()
	return images


def load_data(split=(0.8, 0.2, 0), shuffle=True, need_filter=False, fast_load=True, size = 224):
	extract_data()
	categories = os.listdir('train')
	
	if len(split) != 3:
		raise Exception('split size must be 3')
	summary = sum(split)
	first = split[0] / summary
	second = (split[0] + split[1]) / summary
	
	train_x = []
	train_y = []
	validation_x = []
	validation_y = []
	test_x = []
	test_y = []
	for index, category in enumerate(categories):
		if fast_load:
			file = 'dumps_%d/%s.dump' % (size, category)
			if os.path.isfile(file):
				log.log('%-25s loading from dump...' % category, end='\r')
				images = pickle.load(open(file, 'rb'))
				log.log('%-25s loading from dump...[Done],' % category)
			else:
				if not os.path.isdir(file.split('/')[0]):
					os.mkdir(file.split('/')[0])
				images = get_all_images_in(category, need_filter=need_filter, size = size)
				log.log('%-25s saving dumps...' % category, end='\r')
				pickle.dump(images, open(file, 'wb'))
				log.log('%-25s saving dumps...[Done]' % category)
		else:
			images = get_all_images_in(category, need_filter=need_filter, size = size)

		if shuffle:
			images = shuffle_list(images)
		length = len(images)
		first_index = int(length * first)
		second_index = int(length * second)
		# split data
		train_x += images[:first_index]
		train_y += [index] * first_index
		validation_x += images[first_index : second_index]
		validation_y += [index] * (second_index - first_index)
		test_x += images[second_index:]
		test_y += [index] * (length - second_index)

	train_x = np.array(train_x)
	train_y = np.array(train_y)
	validation_x = np.array(validation_x)
	validation_y = np.array(validation_y)
	test_x = np.array(test_x)
	test_y = np.array(test_y)
	
	log.log('read file finished!')
	return train_x, train_y, validation_x, validation_y, test_x, test_y, categories


def read_image(path, size=224, need_filter=False):
	if type(path) is str:
		data = scipy.ndimage.imread(path)
		data = scipy.misc.imresize(data, (size, size, 3))/255
		if need_filter:
			filter_image(data)
		return data
	if type(path) is list:
		return [read_image(p, size) for p in path]


def filter_image(image):
	image[image[:,:,2] > 0.2] = 0
	image[image[:,:,0] > image[:,:,1]] = 0
	image[image[:,:,2] > image[:,:,1]] = 0

# x: list of path of the images
# y: list of label of the images
def next_batch(x, y, batch_size):
	xx, yy = shuffle_list_2(x, y)
	batch_count = int(len(xx) / batch_size)
	for index in range(batch_count):
		batch_x = xx[index * batch_size: (index + 1) * batch_size]
		batch_y = yy[index * batch_size: (index + 1) * batch_size]
		yield batch_x, batch_y



if __name__ == '__main__':
	train_x, train_y, validation_x, validation_y, test_x, test_y, categories = load_data(shuffle=True)
	print(len(train_x))
	# testing
	#for idx in range(0, 20):
	#    preview_image(train_x[idx], idx)
	#for batch_x, batch_y in next_batch(train_x, train_y, 10):
	#	print(batch_y)

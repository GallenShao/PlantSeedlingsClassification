import os
import zipfile
import scipy.ndimage
import scipy.misc
import numpy as np


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
    return [x[index] for index in arr], [y[index] for index in arr]


def load_data(split=(0.8, 0.1, 0.1), shuffle=True, shuffe_when_return=False):
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
        files = os.listdir('train/%s' % category)
        if shuffle:
            files = shuffle_list(files)
        files = ['train/%s/%s' % (category, file) for file in files]
        length = len(files)
        first_index = int(length * first)
        second_index = int(length * second)
        # split data
        train_x += files[:first_index]
        train_y += [index] * first_index
        validation_x += files[first_index : second_index]
        validation_y += [index] * (second_index - first_index)
        test_x += files[second_index:]
        test_y += [index] * (length - second_index)

    if shuffe_when_return:
        train_x, train_y = shuffle_list_2(train_x, train_y)
        validation_x, validation_y = shuffle_list_2(validation_x, validation_y)
        test_x, test_y = shuffle_list_2(test_x, test_y)
    
    return train_x, train_y, validation_x, validation_y, test_x, test_y, categories


def read_image(path, size=[224, 224]):
    if type(path) is str:
        data = scipy.ndimage.imread(path)
        data = scipy.misc.imresize(data, size)/255
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
        yield np.array(read_image(batch_x)), np.array(batch_y)


def preview_image(path, id):
    image = read_image(path)
    scipy.misc.imsave('%d_original.png' % id, image)
    filter_image(image)
    scipy.misc.imsave('%d_preview.png' % id, image)


if __name__ == '__main__':
    train_x, train_y, validation_x, validation_y, test_x, test_y, categories = load_data(shuffle=False, shuffe_when_return=False)
    # testing
    for idx in range(0, 20):
        preview_image(train_x[idx], idx)
    batch_x, batch_y = next_batch(train_x, train_y, 10).__next__()

import tensorflow as tf
import data

input_dim = 224
input_size = [input_dim, input_dim, 3]
epoch_size = 1000
batch_size = 60
learning_rate = 0.001
regularizer_rate = 0.01


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)
	
def conv2d(x, W, strides):
	return tf.nn.conv2d(x, W, strides=strides, padding='SAME')
  
def max_pool_2x2(x, ksize, strides):
	return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding='SAME')
	
def batch_normalization(input, bias_shape):
	with tf.variable_scope('BN'):
	    mean, var = tf.nn.moments(input, list(range(len(input.shape)-1)), keep_dims=True)
	    shift = tf.Variable(tf.zeros(bias_shape), name='shift')
	    scale = tf.Variable(tf.ones(bias_shape), name='scale')
	    epsilon = 1e-3
	    output = tf.nn.batch_normalization(input, mean, var, shift, scale, epsilon)
	    return output
	
def conv_layer(x, shape, conv_strides):
	with tf.variable_scope('conv_%d_%d' % (shape[0], shape[1])):
		bias_shape = shape[-1:]
		weight = weight_variable(shape)
		biase = bias_variable(bias_shape)
		conv = conv2d(x, weight, [1, conv_strides, conv_strides, 1]) + biase
		conv_BN = batch_normalization(conv, bias_shape)
		conv_relu = tf.nn.relu(conv_BN) 
		return conv_relu

def pool_layer(x, pool_ksize, pool_strides):
	with tf.variable_scope('pool_%d_%d' % (pool_ksize, pool_strides)):
		pool = max_pool_2x2(x, [1, pool_ksize, pool_ksize, 1], [1, pool_strides, pool_strides, 1])
		return pool

def dense_layer(x, shape, activator, keep_prob):
	with tf.variable_scope('dense'):
		bias_shape = shape[-1:]
		weight = weight_variable(shape)
		biase = bias_variable(bias_shape)
		input = tf.nn.dropout(x, keep_prob)
		fc = tf.matmul(input, weight) + biase
		fc_BN = batch_normalization(fc, bias_shape)
		fc_out = activator(fc_BN)
		return fc_out
	
def build_net():
	input_x = tf.placeholder("float",[None, input_dim, input_dim, 3])
	input_y = tf.placeholder("int32",[None])
	transfer_y = tf.one_hot(input_y,12)
	keep_prob = tf.placeholder("float")

	conv1 = conv_layer(input_x, [11, 11 ,3, 96], 4)
	pool1 = pool_layer(conv1, 3, 2)
	conv2 = conv_layer(pool1, [5, 5, 96, 256], 1)
	pool2 = pool_layer(conv2, 3, 2)
	conv3 = conv_layer(pool2, [3, 3, 256, 384], 1)
	conv4 = conv_layer(conv3, [3, 3, 384 ,384], 1)
	conv5 = conv_layer(conv4, [3, 3, 384, 256], 1)
	pool5 = pool_layer(conv5, 3, 2)
	
	flat = tf.reshape(pool5, [-1, 7*7*256])
	fc1 = dense_layer(flat, [7*7*256, 4096], tf.nn.relu, keep_prob)
	fc2 = dense_layer(fc1, [4096, 4096], tf.nn.relu, keep_prob)
	output_y = dense_layer(fc2, [4096, 12], tf.nn.softmax, keep_prob)
	
	entropy = -tf.reduce_sum(transfer_y * tf.log(output_y))
	tf.add_to_collection('losses', entropy)
	cost = tf.add_n(tf.get_collection('losses')) 
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
	
	correct = tf.equal(tf.argmax(output_y,1), tf.argmax(transfer_y,1))
	accuracy = tf.reduce_mean(tf.cast(correct, "float"))
	
	return input_x, input_y, keep_prob, cost, optimizer, accuracy, output_y

def main():
	Xtrain, Ytrain, XCV, YCV, Xtest, Ytest, categories = data.load_data(shuffle=False)
	input_x, input_y, keep_prob, cost, optimizer, accuracy, output_y = build_net()
	saver = tf.train.Saver()
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)
	
	for epoch in range(epoch_size):
		for batch_x, batch_y in data.next_batch(Xtrain, Ytrain, batch_size):
			loss, _ = sess.run([cost,optimizer], feed_dict={input_x:batch_x, input_y:batch_y, keep_prob:0.5})
		print("epoch %d: %f" % (epoch, loss))
		CV_acc = sess.run(accuracy, feed_dict={input_x: XCV, input_y: YCV, keep_prob:1.0})
		print("CV accuracy: %.2f%%" % (CV_acc*100))
		print()
		if CV_acc > 0.92:
			saver.save(sess, "model/alexnet.model")
	
	
if __name__ == '__main__':
	main()

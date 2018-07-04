import tensorflow as tf
import data

input_dim = 224
input_size = [input_dim, input_dim, 3]
epoch_size = 1000
batch_size = 60
learning_rate = 0.001
regularizer_rate = 0.1


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

def conv_layer(x, shape, conv_strides):
	bias_shape = shape[-1:]
	weight = weight_variable(shape)
	biase = bias_variable(bias_shape)
	conv = conv2d(x, weight, [1, conv_strides, conv_strides, 1]) + biase
	conv_mean, conv_var = tf.nn.moments(conv, [0, 1, 2], keep_dims=True)
	conv_shift = tf.Variable(tf.zeros(bias_shape))
	conv_scale = tf.Variable(tf.ones(bias_shape))
	conv_epsilon = 1e-3
	conv_BN = tf.nn.batch_normalization(conv, conv_mean, conv_var, conv_shift, conv_scale, conv_epsilon)
	conv_relu = tf.nn.relu(conv_BN)
	return conv_relu

def pool_layer(x, pool_ksize, pool_strides):
	pool = max_pool_2x2(x, [1, pool_ksize, pool_ksize, 1], [1, pool_strides, pool_strides, 1])
	return pool

def dense_layer(x, shape, activator):
	bias_shape = shape[-1:]
	weight = weight_variable(shape)
	biase = bias_variable(bias_shape)
	fc = tf.matmul(x, weight) + biase
	fc_mean, fc_var = tf.nn.moments(fc, [0], keep_dims=True)
	fc_shift = tf.Variable(tf.zeros(bias_shape))
	fc_scale = tf.Variable(tf.ones(bias_shape))
	fc_epsilon = 1e-3
	fc_BN = tf.nn.batch_normalization(fc, fc_mean, fc_var, fc_shift, fc_scale, fc_epsilon)	
	fc_out = activator(fc_BN)
	return fc_out
	
def build_net():
	input_x = tf.placeholder("float",[None, input_dim, input_dim, 3])
	input_y = tf.placeholder("int32",[None])
	transfer_y = tf.one_hot(input_y,12)
	keep_prob = tf.placeholder("float")

	conv1 = conv_layer(input_x, [3, 3 ,3, 64], 1)
	conv11 = conv_layer(conv1, [3, 3 ,64, 64], 1)
	pool1 = pool_layer(conv11, 2, 2)
	
	conv2 = conv_layer(pool1, [3, 3, 64, 128], 1)
	conv22 = conv_layer(conv2, [3, 3, 128, 128], 1)
	pool2 = pool_layer(conv22, 2, 2)
	
	conv3 = conv_layer(pool2, [3, 3, 128, 256], 1)
	conv33 = conv_layer(conv3, [3, 3, 256 ,256], 1)
	conv333 = conv_layer(conv33, [3, 3, 256 ,256], 1)
	conv3333 = conv_layer(conv333, [3, 3, 256 ,256], 1)
	pool3 = pool_layer(conv3333, 2, 2)
	
	conv4 = conv_layer(pool3, [3, 3, 256, 512], 1)
	conv44 = conv_layer(conv4, [3, 3, 512, 512], 1)
	conv444 = conv_layer(conv44, [3, 3, 512, 512], 1)
	conv4444 = conv_layer(conv444, [3, 3, 512, 512], 1)
	pool4 = pool_layer(conv4444, 2, 2)
	
	conv5 = conv_layer(pool4, [3, 3, 512, 512], 1)
	conv55 = conv_layer(conv5, [3, 3, 512, 512], 1)
	conv555 = conv_layer(conv55, [3, 3, 512, 512], 1)
	conv5555 = conv_layer(conv555, [3, 3, 512, 512], 1)
	pool5 = pool_layer(conv5555, 2, 2)
	
	flat = tf.reshape(pool5, [-1, 7*7*512])
	
	fc1 = dense_layer(flat, [7*7*512, 4096], tf.nn.relu)
	fc1_drop = tf.nn.dropout(fc1, keep_prob)
	fc2 = dense_layer(fc1_drop, [4096, 4096], tf.nn.relu)
	fc2_drop = tf.nn.dropout(fc2, keep_prob)
	output_y = dense_layer(fc2_drop, [4096, 12], tf.nn.softmax)
	
	entropy = -tf.reduce_sum(transfer_y * tf.log(output_y))
	tf.add_to_collection('losses', entropy)
	cost = tf.add_n(tf.get_collection('losses')) 
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
	
	correct = tf.equal(tf.argmax(output_y,1), tf.argmax(transfer_y,1))
	accuracy = tf.reduce_mean(tf.cast(correct, "float"))
	
	return input_x, input_y, keep_prob, cost, optimizer, accuracy

def main():
	Xtrain, Ytrain, XCV, YCV, Xtest, Ytest, categories = data.load_data(shuffle=False, need_filter = True)
	input_x, input_y, keep_prob, cost, optimizer, accuracy = build_net()
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)
	
	for epoch in range(epoch_size):
		for batch_x, batch_y in data.next_batch(Xtrain, Ytrain, batch_size):
			loss, _ = sess.run([cost,optimizer], feed_dict={input_x:batch_x, input_y:batch_y, keep_prob:0.5})
		if epoch%1 == 0:
			print("epoch %d: %f" % (epoch, loss))
			acc_count = 0
			acc_sum = 0.0
			for batch_x, batch_y in data.next_batch(XCV, YCV, batch_size):
				CV_acc = sess.run(accuracy, feed_dict={input_x: batch_x, input_y: batch_y, keep_prob:1.0})
				acc_count += 1
				acc_sum += CV_acc
			print("CV accuracy: %.2f%%" % (acc_sum/acc_count*100))
			print()
	
	
if __name__ == '__main__':
	main()
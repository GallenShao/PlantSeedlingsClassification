import tensorflow as tf
import log

learning_rate = 1e-3


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
#    initial = tf.random_normal(shape)
    return tf.Variable(initial)


def bias_variable(shape, constant=0.2):
    initial = tf.constant(constant, shape=shape)
#    initial = tf.random_normal(shape)
    return tf.Variable(initial)


def batch_normalization(input, bias_shape):
    mean, var = tf.nn.moments(input, list(range(len(input.shape)-1)), keep_dims=True)
    shift = tf.Variable(tf.zeros(bias_shape))
    cale = tf.Variable(tf.ones(bias_shape))
    epsilon = 1e-3
    output = tf.nn.batch_normalization(input, mean, var, shift, scale, epsilon)
    return output


def conv2d(input, weight_shape, stride=1, padding='SAME', bias_init_contant=0.2, activate=tf.nn.relu):
    output = tf.nn.conv2d(input, weight_variable(weight_shape), strides=[1, stride, stride, 1], padding=padding)
    output = tf.nn.bias_add(output, bias_variable(weight_shape[-1:], bias_init_contant))
    output = batch_normalization(output, weight_shape[-1:])
    output = activate(output)
    return output


def fc(input, input_size, output_size, bias_init_contant=0.2):
    output = tf.reshape(input, [-1, input_size])
    output = tf.nn.bias_add(tf.matmul(output, weight_variable([input_size, output_size])), bias_variable([output_size], bias_init_contant))
    output = batch_normalization(output, [output_size])
    return output


def maxpool(input, ksize, stride=1, padding='SAME'):
    return tf.nn.max_pool(input, ksize=[1,ksize,ksize,1], strides=[1,stride,stride,1], padding=padding)


def avgpool(input, ksize, stride=1, padding='VALID'):
    return tf.nn.avg_pool(input, ksize=[1,ksize,ksize,1], strides=[1,stride,stride,1], padding=padding)


def inception(input, input_size, output_1, reduce_3, output_3, reduce_5, output_5, output_pool):
    branch_0 = conv2d(input, [1, 1, input_size, output_1])

    branch_1 = conv2d(input, [1, 1, input_size, reduce_3])
    branch_1 = conv2d(branch_1, [3, 3, reduce_3, output_3])

    branch_2 = conv2d(input, [1, 1, input_size, reduce_5])
    branch_2 = conv2d(branch_2, [5, 5, reduce_5, output_5])
    
    branch_3 = maxpool(input, ksize=3)
    branch_3 = conv2d(branch_3, [1, 1, input_size, output_pool])

    return tf.concat([branch_0, branch_1, branch_2, branch_3], 3)


def build_googlenet(input):
    output = conv2d(input, [7, 7, 3, 64], stride=2)
    output = maxpool(output, ksize=3, stride=2)
    output = tf.nn.local_response_normalization(output, depth_radius=5/2.0, bias=2.0, alpha=1e-4, beta=0.75)

    output = conv2d(output, [1, 1, 64, 64])
    output = conv2d(output, [3, 3, 64, 192])
    output = tf.nn.local_response_normalization(output, depth_radius=5/2.0, bias=2.0, alpha=1e-4, beta=0.75)
    output = maxpool(output, ksize=3, stride=2)

    output = inception(output, 192, 64, 96, 128, 16, 32, 32)
    output = inception(output, 256, 128, 128, 192, 32, 96, 64)

    output = maxpool(output, ksize=3, stride=2)

    output = inception(output, 480, 192, 96, 208, 16, 48, 64)
    
    softmax0 = avgpool(output, ksize=5, stride=3)
    softmax0 = conv2d(softmax0, [1, 1, 512, 128])
    softmax0 = fc(softmax0, 4 * 4 * 128, 1024)
    softmax0 = tf.nn.relu(softmax0)
    softmax0 = tf.nn.dropout(softmax0, keep_prob=0.7)
    softmax0 = fc(softmax0, 1024, 12, bias_init_contant=0.0)
#    softmax0 = tf.nn.softmax(softmax0)

    output = inception(output, 512, 160, 112, 224, 24, 64, 64)
    output = inception(output, 512, 128, 128, 256, 24, 64, 64)
    output = inception(output, 512, 112, 144, 288, 32, 64, 64)
    
    softmax1 = avgpool(output, ksize=5, stride=3)
    softmax1 = conv2d(softmax1, [1, 1, 528, 128])
    softmax1 = fc(softmax1, 4 * 4 * 128, 1024)
    softmax1 = tf.nn.relu(softmax1)
    softmax1 = tf.nn.dropout(softmax1, keep_prob=0.7)
    softmax1 = fc(softmax1, 1024, 12, bias_init_contant=0.0)
#    softmax1 = tf.nn.softmax(softmax1)

    output = inception(output, 528, 256, 160, 320, 32, 128, 128)
    
    output = maxpool(output, ksize=3, stride=2)

    output = inception(output, 832, 256, 160, 320, 32, 128, 128)
    output = inception(output, 832, 384, 192, 384, 48, 128, 128)

    softmax2 = avgpool(output, ksize=7)
    softmax2 = tf.nn.dropout(softmax2, keep_prob=0.4)
    softmax2 = fc(softmax2, 1 * 1 * 1024, 12, bias_init_contant=0.0)
#    softmax2 = tf.nn.softmax(softmax2)

    return softmax0, softmax1, softmax2


def build_net():
    log.log('building googlenet...', end='\r')
    input_x = tf.placeholder(tf.float32, [None, 224, 224, 3])
    input_y = tf.placeholder(tf.int32, [None])
    keep_prob = tf.placeholder(tf.float32)
    transfer_y = tf.one_hot(input_y, 12)
    
    softmax0, softmax1, softmax2 = build_googlenet(input_x)
    
    loss0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=softmax0, labels=transfer_y))
    loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=softmax1, labels=transfer_y))
    loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=softmax2, labels=transfer_y))
#    loss0 = -tf.reduce_sum(transfer_y * tf.log(softmax0))
#    loss1 = -tf.reduce_sum(transfer_y * tf.log(softmax1))
#    loss2 = -tf.reduce_sum(transfer_y * tf.log(softmax2))
    loss = loss0 * 0.3 + loss1 * 0.3 + loss2

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    prediction = tf.equal(tf.argmax(tf.nn.softmax(softmax2), 1), tf.argmax(transfer_y, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    log.log('building googlenet...[Done]')
    return input_x, input_y, keep_prob, loss, optimizer, accuracy


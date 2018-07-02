import tensorflow as tf
import data
import log
import googlenet
import alexnet
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

epoch_size = 1000
batch_size = 60

if __name__ == '__main__':
    Xtrain, Ytrain, XCV, YCV, Xtest, Ytest, categories = data.load_data(split=(0.8, 0.2, 0), shuffle=True, need_filter=True, fast_load=True)
    
    if len(sys.argv) < 2:
        network = alexnet.build_net()
    elif (sys.argv[1] == 'alexnet'):
        network = alexnet.build_net()
    elif (sys.argv[1] == 'googlenet'):
        network = googlenet.build_net()
    else:
        network = alexnet.build_net()
    
    input_x, input_y, keep_prob, cost, optimizer, accuracy = network
    
    # start sess
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    for epoch in range(epoch_size):
        print()
        for batch, (batch_x, batch_y) in enumerate(data.next_batch(Xtrain, Ytrain, batch_size)):
            loss, _ = sess.run([cost, optimizer], feed_dict={input_x:batch_x, input_y:batch_y, keep_prob:0.5})
            if batch % 10 == 0:
                log.log("epoch %d batch %d: %f" % (epoch, batch, loss))
                CV_acc = sess.run(accuracy, feed_dict={input_x: XCV, input_y: YCV, keep_prob:1.0})
                log.log("CV    accuracy: %.2f%%" % (CV_acc*100))
        if epoch % 1 == 0:
            log.log("epoch %d: %f" % (epoch, loss))
            CV_acc = sess.run(accuracy, feed_dict={input_x: XCV, input_y: YCV, keep_prob:1.0})
            #train_acc = sess.run(accuracy, feed_dict={input_x: Xtrain, input_y: Ytrain, keep_prob:1.0})
            #print("Train accuracy: %.2f%%" % (train_acc*100))
            log.log("CV    accuracy: %.2f%%" % (CV_acc*100))
    test_acc = sess.run(accuracy, feed_dict={input_x: Xtest, input_y: Ytest, keep_prob:1.0})
    log.log("Test  accuracy: %.2f%%" % (test_acc*100))

import tensorflow as tf
import data
import log
import googlenet
import alexnet
import shaonet
import sys
import os
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

epoch_size = 1000
batch_size = 60

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="main script for training")
    parser.add_argument('--network', '-n', required=True, help='select network', choices=['alexnet', 'googlenet', 'shaonet'])
    parser.add_argument('--save', '-s', type=float, default=0, help='save model when accuracy is higher than')
    parser.add_argument('--filter', '-f', action='store_true', help='use filter')
    parser.add_argument('--reload', '-r', action='store_true', help='don\'t use fast load')
    args = parser.parse_args()
    
    if args.network == 'alexnet':
        network = alexnet.build_net()
    elif args.network == 'googlenet':
        network = googlenet.build_net()
    elif args.network == 'shaonet':
        network = shaonet.build_net()
    else:
        raise Exception('Illegal network type')
    
    input_x, input_y, keep_prob, cost, optimizer, accuracy, output = network

    Xtrain, Ytrain, XCV, YCV, Xtest, Ytest, categories = data.load_data(
        split = (0.8, 0.2, 0),
        shuffle = args.save == 0,
        need_filter = args.filter,
        fast_load = not args.reload,
        size = int(input_x.shape[1]))

    loss_summary = tf.summary.scalar('loss', cost)
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)

    # start sess
    sess = tf.Session()
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter('%s_logs/' % args.network, sess.graph)
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
            writer.add_summary(sess.run(loss_summary, feed_dict={input_x:batch_x, input_y:batch_y, keep_prob:1.0}), epoch)
            CV_acc = sess.run(accuracy, feed_dict={input_x: XCV, input_y: YCV, keep_prob:1.0})
#            test_acc = sess.run(accuracy, feed_dict={input_x: Xtest, input_y: Ytest, keep_prob:1.0})
            writer.add_summary(sess.run(accuracy_summary, feed_dict={input_x: XCV, input_y: YCV, keep_prob:1.0}), epoch)
            log.log("CV    accuracy: %.2f%%" % (CV_acc*100))
#            log.log("test  accuracy: %.2f%%" % (test_acc*100))
            # save model
            if args.save != 0 and CV_acc > args.save:
                log.log('saving model...', end='\r')
                if not os.path.isdir('model'):
                    os.mkdir('model')
                saver.save(sess, 'model/%s.model' % args.network)
                log.log('saving model...[Done]')
                break


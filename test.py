import tensorflow as tf
import numpy as np
import data
import log
import googlenet
import alexnet
import shaonet
import sys
import os
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def get_outputs(networks, args):
	outputs = []
	for network in networks:
		if network == 'alexnet':
			NN = alexnet.build_net()
		elif network == 'googlenet':
			NN = googlenet.build_net()
		elif network == 'shaonet':
			NN = shaonet.build_net()
			
		input_x, input_y, keep_prob, cost, optimizer, accuracy, output = NN
		Xtrain, Ytrain, XCV, YCV, Xtest, Ytest, categories = data.load_data(
			split = (0.8, 0.2, 0),
			shuffle = False,
			need_filter = False,
			fast_load = True,
			size = int(input_x.shape[1]))
			
		sess = tf.Session()
		saver = tf.train.Saver()
		saver.restore(sess, 'model/%s.model' % network)
	
		CV_output, CV_acc = sess.run([output,accuracy], feed_dict={input_x: XCV, input_y: YCV, keep_prob:1.0})
		log.log("%s accuracy: %.2f%%\n" % (network, CV_acc*100))
		outputs.append(CV_output)
		
		sess.close()
		tf.reset_default_graph()	
		
	return outputs, YCV


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="main script for testing")
	parser.add_argument('--network', '-n', required=True, help='select network', choices=['alexnet', 'googlenet', 'shaonet', 'all'])
	parser.add_argument('--policy', '-p', help='select policy when network is all', choices=['1','2'], default='1')
	args = parser.parse_args()
	
	if args.network == 'alexnet':
		networks = ['alexnet']
	elif args.network == 'googlenet':
		networks = ['googlenet']
	elif args.network == 'shaonet':
		networks = ['shaonet']
	elif args.network == 'all':
		networks = ['alexnet', 'googlenet', 'shaonet']
	else:
		raise Exception('Illegal network type')
	
	outputs, YCV = get_outputs(networks, args)
	if len(outputs) > 1:
		if args.policy == '1':
			outputs = sum(outputs)
			classification = outputs.argmax(axis = 1)
			acc = sum(classification == YCV)
		elif args.policy == '2':
			classifications = []
			for output in outputs:
				classification = output.argmax(axis = 1)
				classifications.append(classification)
			res = []
			for i in range(len(res[0])):
				res.append(classifications[0][i] if classifications[0][i] == classifications[2][i] else classifications[1][i])
			res = np.array(res)
			acc = sum(res == YCV)

		log.log("Final accuracy: %.2f%%" % (acc/len(YCV)*100))
	
	
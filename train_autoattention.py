# coding: utf-8
import os, sys


import pandas as pd
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score
from tensorflow.python.keras import backend as K
from deepctr.utils import SingleFeat


from utils import *
from models import AutoAttention, DotProduct

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
K.set_session(tf.Session(config=tfconfig))

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth',100)
def binary_crossentropy(y_true, y_pred):
	return K.mean(K.binary_crossentropy(y_true, y_pred + 1e-5), axis=-1)

if __name__ == "__main__":
	FRAC = FRAC
	#FRAC=None
	SESS_MAX_LEN = DIN_SESS_MAX_LEN
	fd = pd.read_pickle(ROOT_DATA+'model_input/din_fd_' +
						str(FRAC) + '_' + str(SESS_MAX_LEN) + '.pkl')
	key_length = pd.read_pickle(
		ROOT_DATA+'model_input/din_input_len_' + str(FRAC) + '_' + str(SESS_MAX_LEN) + '.pkl')
	model_input = pd.read_pickle(
		ROOT_DATA+'model_input/din_input_' + str(FRAC) + '_' + str(SESS_MAX_LEN) + '.pkl')
	model_input += key_length
	label = pd.read_pickle(ROOT_DATA+'model_input/din_label_' +
						   str(FRAC) + '_' + str(SESS_MAX_LEN) + '.pkl')
	#fd = pd.read_pickle(ROOT_DATA+'model_input/din_fd_' +
	#					 str(SESS_MAX_LEN) + '.pkl')
	#key_length = pd.read_pickle(
	#    ROOT_DATA+'model_input/din_input_len_'+str(SESS_MAX_LEN)+'.pkl')
	#model_input = pd.read_pickle(
	#	ROOT_DATA+'model_input/din_input_'  + str(SESS_MAX_LEN) + '.pkl')
	#model_input += key_length
	#label = pd.read_pickle(ROOT_DATA+'model_input/din_label_' +
	#					    str(SESS_MAX_LEN) + '.pkl')
	#sample_sub = pd.read_pickle(
	#	ROOT_DATA+'sampled_data/sampled_data1' + '.pkl')
	sample_sub = pd.read_pickle(
		ROOT_DATA+'sampled_data/raw_sample_' + str(FRAC) + '.pkl')
	#sample_num = len(label)
	#idx = list(range(sample_num))
	# train_num = int(sample_num * 0.7)
	# train_num = 4895270 # tencent_v3
	#train_num = 6666928 # tencent_v4
	#train_num = 5547569 # 6666928
	#valid_num = 6666928
	#train_idx = idx[:train_num]
	#valid_idx = idx[train_num:valid_num]
	#test_idx = idx[valid_num:]
	#train_input = [i[train_idx] for i in model_input]
	#valid_input = [i[valid_idx] for i in model_input]
	#test_input = [i[test_idx] for i in model_input]
	#train_label = label[train_idx]
	#valid_label = label[valid_idx]
	#test_label = label[test_idx]
	#print(train_input)
	sess_len_max = SESS_MAX_LEN
	BATCH_SIZE = 4096

	sample_sub['idx'] = list(range(sample_sub.shape[0]))
	train_idx = sample_sub.loc[sample_sub.time_stamp <
								   1494547200, 'idx'].values

	valid_idx = sample_sub.loc[(sample_sub.time_stamp >=
								1494547200) & (sample_sub.time_stamp <
											   1494633600), 'idx'].values
	test_idx = sample_sub.loc[sample_sub.time_stamp >=
							  1494633600, 'idx'].values

	train_input = [i[train_idx] for i in model_input]
	valid_input = [i[valid_idx] for i in model_input]
	test_input = [i[test_idx] for i in model_input]
	train_label = label[train_idx]
	valid_label = label[valid_idx]
	test_label = label[test_idx]


	sess_len_max = SESS_MAX_LEN
	sess_feature = ['cate_id', 'brand']
	#sess_feature = ['ad_id', 'campaign_id', 'creative_id']
	
	BATCH_SIZE = 4096
	TEST_BATCH_SIZE = 2 ** 14
    
	print('train len: %d\ttest_len: %d' % (train_label.shape[0], test_label.shape[0]))

	is_prun = False
	sparse_rate = 0.5
	model_type = sys.argv[1]
	for i in range(5):
		print('########################################')
		if model_type == 'DotProduct':
			print('Start training DotProduct: ' + str(i))
			log_path = ROOT_DATA + 'log/DotProduct_log_' + str(i) + '.txt'
			best_model_path = ROOT_DATA + 'best_model/DotProduct.h5'
			model = DotProduct(fd, sess_feature, embedding_size=16, hist_len_max=sess_len_max)
		elif model_type == 'AutoAttention':
			print('Start training AutoAttention: ' + str(i))
			log_path = ROOT_DATA + 'log/AutoAttention_log_' + str(i) + '.txt'
			best_model_path = ROOT_DATA + 'best_model/AutoAttention.h5'
			model = AutoAttention(fd, sess_feature, embedding_size=64, hist_len_max=sess_len_max)
		elif model_type == 'AutoAttention_Prun':
			is_prun = True
			sparse_rate = float(sys.argv[2])
			print('Start training AutoAttention_Prun: ' + str(i))
			log_path = ROOT_DATA + 'log/AutoAttention_Prun_log_' + str(i) + '.txt'
			best_model_path = ROOT_DATA + 'best_model/AutoAttention_Prun.h5'
			model = AutoAttention(fd, sess_feature, embedding_size=16, hist_len_max=sess_len_max)
		else:
			print("Wrong argument model type!")
			sys.exit(0)

		# model.compile(optimizer='adagrad', loss='binary_crossentropy')
		# opt = tf.keras.optimizers.Adam(lr=0.0001)
		opt = tf.keras.optimizers.Adagrad(lr=0.01)


		#model.compile(optimizer=opt, loss=binary_crossentropy)
		model.compile('adagrad', 'binary_crossentropy')
		log_dir="/cephfs/group/file-teg-datamining-wx-dm-intern/tianyihu/AutoAttention"
		tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
		hist_ = model.fit(train_input[:], train_label,
						  batch_size=BATCH_SIZE, epochs=10, initial_epoch=0, verbose=0,
						  callbacks=[LossHistory(log_path),tensorboard_callback,
									 auc_callback(training_data=[train_input, train_label],
												  valid_data=[valid_input, valid_label],
												  test_data=[test_input, test_label],
												  best_model_path=best_model_path,
												  is_prun=is_prun, target_sparse=sparse_rate)])
		K.clear_session()
		del model
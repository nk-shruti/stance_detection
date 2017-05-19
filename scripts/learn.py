from keras.models import Sequential, load_model, Model
from keras.layers import Input, LSTM, concatenate
from keras.layers.embeddings import Embedding
from keras.layers.core import Flatten, Dense, Dropout, Activation, Reshape
from keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.layers.recurrent import LSTM, GRU
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau
from keras.utils import np_utils
from os.path import dirname, abspath
from os import listdir
import numpy as np
import h5py, pickle
from random import randint, choice, shuffle, sample
from sys import argv
from keras.layers.advanced_activations import PReLU
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold
from data import max_text_len, max_claim_len, process_data, EMBEDDING_DIM, ROOT
from sklearn.metrics import confusion_matrix

batch_size = 64


def get_confusion_matrix(y_true, y_pred):
	# preds = []
	# for i, x in enumerate(y_pred):
	# 	preds.append(1. if x[0] > 0.5 else 0.)
	# return confusion_matrix(y_true, preds)
	return confusion_matrix(y_true,y_pred)


def visualizer(modell):
	plot(modell, to_file=ROOT + '/vis.png', show_shapes=True)

def init_model(preload=None, declare=True, data=None):
	print 'Compiling model...'
	if not declare and preload: return load_model(preload)

	else:

		embedding_matrix, len_word_index = data
		print('building the model...')

		inp1 = Input(shape=(max_claim_len,))
		embedding_1 = Embedding(len_word_index + 1,
							EMBEDDING_DIM,
							name='word_vectors1',
							input_length=max_claim_len,
							weights=[embedding_matrix],
							trainable=True) (inp1)

		inp2 = Input(shape=(max_text_len,))
		embedding_2 = Embedding(len_word_index + 1,
							EMBEDDING_DIM,
							name='word_vectors2',
							input_length=max_text_len,
							weights=[embedding_matrix],
							trainable=True) (inp2)

		lstm = LSTM(64)

		out_1 = lstm(embedding_1)
		out_2 = lstm(embedding_2)

		merged = concatenate([out_1, out_2])

		dense = Dense(128) (merged)
		elu = ELU() (dense)
		output = Dense(4, activation='softmax') (elu)
		
		model = Model(inputs=[inp1, inp2], outputs=output)

		if preload:
			model.load_weights(preload)

		return model

def get_data():
	data = None
	if 'pd' not in listdir('{}/processedData/'.format(ROOT)):
		data = process_data()
		pickle.dump(data, open('{}/processedData/pd'.format(ROOT),'w'))
	else:
		data = pickle.load(open('{}/processedData/pd'.format(ROOT)))
	# print "shape of data:",data.shape
	return data

def runner(epochs):
	ne = epochs
	x_train, y_train, x_val, y_val = [None] * 4
	data = get_data()
	embedding_matrix = data['embedding_matrix']
	len_word_index = data['len_word_index']
	x_train, y_train = data['x_train'], data['y_train']
	x_val, y_val = data['x_val'], data['y_val']
	model = init_model(data=[embedding_matrix, len_word_index])
	model.compile(optimizer='adam', metrics=['acc'], loss='categorical_crossentropy')
	val_checkpoint = ModelCheckpoint('bestval.h5','val_acc', 1, True)
	cur_checkpoint = ModelCheckpoint('current.h5')
	print 'Model compiled.'

	# x_train = np.asarray(x_train, dtype=np.float32)
	# x_val = np.asarray(x_val,dtype=np.float32)

	# print x_train.shape
	# print y_train.shape
	y_train = np.asarray(y_train,dtype=np.float32)
	x_train_1 = np.zeros(((y_train.shape[0]),max_claim_len))
	x_train_2 = np.zeros(((y_train.shape[0]),max_text_len))
	x_val_1 = np.zeros(((y_val.shape[0]),max_claim_len))
	x_val_2 = np.zeros(((y_val.shape[0]),max_text_len))

	# print x_train[1][1]
	for i in range(0,len(x_train)):
		x_train_1[i] = x_train[i][0]
	for i in range(0,len(x_train)):
		x_train_2[i] = x_train[i][1]
	
	for i in range(0,len(x_val)):
		x_val_1[i] = x_val[i][0]
	for i in range(0,len(x_val)):
		x_val_2[i] = x_val[i][1]
	

	class_weights = dict()
	y = dict()
	values = [0,0,0,0]
	for i in range(0,y_train.shape[0]):
		for j in range(0,4):
			if y_train[i][j] == 1:
				values[j] += 1
	for i in range(0,4):
		y[i] = values[i]

	c = np.asarray([0.]*values[0] + [1.]*values[1] + [2.]*values[2] + [3.]*values[3])	
	for i, wt in enumerate(class_weight.compute_class_weight('balanced', np.unique(c), c)):
		class_weights[i] = wt
 	# cw = compute_class_weight('balanced', np.unique(y_train), y_train)

	model.fit([x_train_1,x_train_2], y_train, class_weight = class_weights, validation_data=([x_val_1,x_val_2 ], y_val),batch_size=32, epochs=20,
                 verbose=1, callbacks=[val_checkpoint])


def main(args):
	mode = None
	if len(args) == 2: mode, preload = args
	else: raise ValueError('Incorrect number of args.')

	if preload == 'none': preload = None
	if mode == 'vis':
		data = get_data()
		model = init_model(data=[data['embedding_matrix'], data['len_word_index']])
		return visualizer(model)
	if mode == 'train':
		return runner(50)
	if mode == 'confusion':
		model = init_model(preload=preload, declare=False)
		data = process_data()
		embedding_matrix = data['embedding_matrix']
		len_word_index = data['len_word_index']
		x_train, y_train = data['x_train'], data['y_train']
		x_val, y_val = data['x_val'], data['y_val']
		y_train = np.asarray(y_train,dtype=np.float32)
		x_train_1 = np.zeros(((y_train.shape[0]),max_claim_len))
		x_train_2 = np.zeros(((y_train.shape[0]),max_text_len))
		x_val_1 = np.zeros(((y_val.shape[0]),max_claim_len))
		x_val_2 = np.zeros(((y_val.shape[0]),max_text_len))

		# print x_train[1][1]
		for i in range(0,len(x_train)):
			x_train_1[i] = x_train[i][0]
		for i in range(0,len(x_train)):
			x_train_2[i] = x_train[i][1]
		
		for i in range(0,len(x_val)):
			x_val_1[i] = x_val[i][0]
		for i in range(0,len(x_val)):
			x_val_2[i] = x_val[i][1]
		y_true = data['y_val']
		y_pred = model.predict([x_val_1,x_val_2])
		y_t = []
		for i in range(0,len(y_true)):
			for j in range(0,4):
				if y_true[i][j] == 1:
					y_t.append(j)
		y_p = []
		for i in range(0,len(y_true)):
			for j in range(0,4):
				if y_pred[i][j] == max(y_pred[i]):
					y_p.append(j)

		print len(y_t)
		print len(y_p)


		print get_confusion_matrix(y_t, y_p)
	else: 
		raise ValueError('Incorrect mode')

if __name__ == '__main__':
	main(argv[1:])
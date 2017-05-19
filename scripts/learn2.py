from keras.models import Sequential, load_model, Model
from keras.layers import Input, LSTM, Merge
from keras.layers.embeddings import Embedding
from keras.layers.core import Flatten, Dense, Dropout, Activation, Reshape
from keras.layers.convolutional import Convolution2D, Convolution1D, MaxPooling2D, MaxPooling1D, \
		ZeroPadding2D, UpSampling2D, UpSampling1D
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
from data import max_text_len, max_claim_len, process_data, EMBEDDING_DIM, ROOT
from utils import *

batch_size = 32

def pop_layer(model):
	if not model.outputs:
		raise Exception('Sequential model cannot be popped: model is empty.')

	model.layers.pop()
	if not model.layers:
		model.outputs = []
		model.inbound_nodes = []
		model.outbound_nodes = []
	else:
		model.layers[-1].outbound_nodes = []
		model.outputs = [model.layers[-1].output]
	model.built = False
	# print 'Last layer is now: ' + model.layers[-1].name
	return model

def init_model(preload=None, declare=True, data=None):
	print 'Compiling model...'
	if not declare and preload: return load_model(preload)

	else:

		embedding_matrix, len_word_index = data
		model = Sequential()
		print('building the model...')
		model1 = Sequential()
		model1.add(Embedding(len_word_index + 1,
							EMBEDDING_DIM,
							name='word_vectors1',
							# input_length=max_len,
							weights=[embedding_matrix],
							trainable=True))	
		model1.add(LSTM(16, dropout_W=0.2, dropout_U=0.2))	
		model2 = Sequential()
		model2.add(Embedding(len_word_index + 1,
							EMBEDDING_DIM,
							name='word_vectors2',
							# input_length=max_len,
							weights=[embedding_matrix],
							trainable=True))	
		model2.add(LSTM(16, dropout_W=0.2, dropout_U=0.2))	
		model.add(Merge([model1,model2],mode='concat'))
		model.add(BatchNormalization())

		model.add(Dense(16))
		model.add(PReLU())
		model.add(Dropout(0.2))
		model.add(BatchNormalization())
		model.add(Dense(4,activation='sigmoid', name='sentiment_class'))
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
	print x_train.shape
	print y_train.shape
	x_train_1 = np.zeros(((x_train.shape[0]),max_claim_len))
	x_train_2 = np.zeros(((x_train.shape[0]),max_text_len))
	print x_train_1.shape
	print x_train_2.shape
	print x_train[1][0]
	for i in range(0,x_train.shape[0]):
		x_train_1[i] = x_train[i][0]
	for i in range(0,x_train.shape[0]):
		x_train_2[i] = x_train[i][1]
		# x_train_1.append(np.asarray(x_train[i][0],dtype=np.float32))
		# x_train_2.append(np.asarray(x_train[i][1],dtype=np.float32))

	x_train_2 = np.asarray(x_train_2,dtype=np.float32)
	x_train_1 = np.asarray(x_train_1,dtype=np.float32)
	print x_train_2.shape
	print x_train_1.shape
	model.fit([x_train_1,x_train_2], y_train, batch_size=32, epochs=20,
                 verbose=1, validation_split=0.1, shuffle=True)


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
		data = category_data(src, target=dest)
		y_true = data['y_val']
		y_pred = model.predict(data['x_val'])
		print get_confusion_matrix(y_true, y_pred)
	else: raise ValueError('Incorrect mode')

if __name__ == '__main__':
	main(argv[1:])

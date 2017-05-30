from keras.models import Sequential, load_model, Model
from keras.layers import Input, LSTM, concatenate
from keras.layers.embeddings import Embedding
from keras.layers.core import Flatten, Dense, Dropout, Activation, Reshape, RepeatVector
from keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import Bidirectional
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau
from keras.utils import np_utils
from os.path import dirname, abspath
from keras.preprocessing.sequence import pad_sequences
from os import listdir
import numpy as np
import h5py, pickle
from random import randint, choice, shuffle, sample
from sys import argv
from keras.layers.advanced_activations import PReLU
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold
from data_multimodel import max_text_len, max_claim_len, process_data, EMBEDDING_DIM, ROOT
from sklearn.metrics import confusion_matrix
from keras.utils import plot_model
from keras.layers.wrappers import TimeDistributed

batch_size = 64


def get_confusion_matrix(y_true, y_pred):
	return confusion_matrix(y_true,y_pred)


def visualizer(modell):
	plot_model(modell, to_file=ROOT + '/vis.png', show_shapes=True)

def init_model(data=None):
	print 'Compiling model...'
	embedding_matrix, len_word_index = data
	print('building the model...')
	inp1 = Input(shape=(max_claim_len,), name='main_input')
	embedding_1 = Embedding(len_word_index + 1,
						EMBEDDING_DIM,
						name='word_vectors1',
						input_length=max_claim_len,
						weights=[embedding_matrix],
						trainable=True) (inp1)
	# drop_1 = Dropout(0.1, name='input1_dropout') (embedding_1)
	inp2 = Input(shape=(max_text_len,))
	embedding_2 = Embedding(len_word_index + 1,
						EMBEDDING_DIM,
						name='word_vectors2',
						input_length=max_text_len,
						weights=[embedding_matrix],
						trainable=True) (inp2)
	# drop_2 = Dropout(0.1, name='input2_dropout') (embedding_2)
	gru = GRU(64)
	out_1 = gru(embedding_1)
	out_2 = gru(embedding_2)

	merged = concatenate([out_1, out_2])

	aux_output = Dense(1, activation='sigmoid',name='aux_output') (merged)
	#here we have to expand the output into a 128 tensor 
	next_inp = RepeatVector(128) (aux_output)
	merged = Reshape((128,1)) (merged)
	m = concatenate([next_inp,merged],axis=2)
	m = Reshape((256,1)) (m)

	m = Flatten() (m)
	x = Dense(128) (m)
	e = ELU() (x)
	main_output = Dense(4,activation='softmax',name='main_output') (e)
	model = Model(inputs=[inp1, inp2], outputs=[main_output,aux_output])

	return model

def get_data():
	data = None
	if 'pd' not in listdir('{}/processedData/'.format(ROOT)):
		data = process_data()
		pickle.dump(data, open('{}/processedData/pd'.format(ROOT),'w'))
	else:
		data = pickle.load(open('{}/processedData/pd'.format(ROOT)))
	return data

def runner(epochs):
	ne = epochs
	
	data = get_data()
	embedding_matrix = data['embedding_matrix']
	len_word_index = data['len_word_index']

	# x_train, y_train = data['x_train'], data['y_train']
	# x_val, y_val = data['x_val'], data['y_val']

	x_train_1 = data['x_train_claim']
	x_train_2 = data['x_train_text']
	x_val_1 = data['x_val_claim']
	x_val_2 = data['x_val_text']
	y_train = data['y1_train']
	y_val = data['y1_val']
	labels2_train = data['y2_train']
	labels2_val = data['y2_val']

 
	model = init_model(data=[embedding_matrix, len_word_index])
	model.compile(optimizer='adam', metrics=['acc'], 
		loss={'main_output': 'categorical_crossentropy', 'aux_output': 'binary_crossentropy'})
	val_checkpoint = ModelCheckpoint('bestval.h5','val_main_output_acc', 1, True)
	cur_checkpoint = ModelCheckpoint('current.h5')
	print 'Model compiled.'

	#FOR ASSIGNING CLASS WEIGHTS
	class_weights = dict()
	class_weights_2 = dict()
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

	values2 = [0,0]
	for i in range(0,y_train.shape[0]):
		if labels2_train[i]==0:
			values2[0] += 1
		else:
			values2[1] += 1
	c2 = np.asarray([0.]*values2[0] + [1.]*values2[1])
	for i, wt in enumerate(class_weight.compute_class_weight('balanced', np.unique(c2), c2)):
		class_weights_2[i] = wt

	model.fit([x_train_1,x_train_2], [y_train,labels2_train], 
			# class_weight = {'main_output':class_weights,'aux_output':class_weights_2}, 
			# uncomment the previous line to include class weights
			validation_data=([x_val_1,x_val_2 ], [y_val,labels2_val]),batch_size=32, epochs=ne,
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
		return runner(20)
	if mode == 'confusion':
		model = init_model(preload=preload)
		data = process_data()
		embedding_matrix = data['embedding_matrix']
		len_word_index = data['len_word_index']
		y1_train, y2_train = data['y1_train'], data['y2_train']
		y1_val, y2_val = data['y1_val'], data['y2_val']
		x_val_1,x_val_2 = data['x_val_claim'], data['x_val_text']
		#for computing the confusion matrix
		y_true = y1_val
		y_pred_1,y_pred_2 = model.predict([x_val_1,x_val_2])
		#array of the class labels:
		#0 - unrelated, 1:discuss, 2:agree, 3:disagree
		y_t = []
		for i in range(0,len(y_true)):
			for j in range(0,4):
				if y_true[i][j] == 1:
					y_t.append(j)
		y_p = []
		for i in range(0,len(y_true)):
			for j in range(0,4):
				if y_pred_1[i][j] == max(y_pred_1[i]):
					y_p.append(j)
		print get_confusion_matrix(y_t, y_p)
	else: 
		raise ValueError('Incorrect mode')

if __name__ == '__main__':
	main(argv[1:])
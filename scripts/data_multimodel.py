from os.path import dirname, abspath
from keras.preprocessing.text import Tokenizer,one_hot
import csv
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import pickle
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = dirname(dirname(abspath(__file__)))
DATA_DIR = ROOT + '/fnc-1/'
max_text_len = 300
max_claim_len = 50
EMBEDDING_DIM = 50
# max_nb_reviews = 200000
MAX_NB_WORDS = 20000

classes = { 'unrelated':0, 'discuss':1,'agree':2,'disagree':3 }


def read_file_body(fname):
	body_index_text = {}
	with open(fname,'r') as f:
		reader = csv.DictReader(f)
		for row in reader:
			body_index_text[row['Body ID']] = row['articleBody']
	return body_index_text


def read_file_stance(fname1,fname2):
	claims = []
	texts = []
	labels1 = []
	labels2 = []
	body_index_text = read_file_body(fname2)
	# agree, disagree, discuss, unrelated = 0
	with open(fname1,'r') as f:
		reader = csv.DictReader(f)
		for row in reader:
			claims.append(row['Headline'])
			texts.append(body_index_text[row['Body ID']])
			labels1.append(row['Stance'])
			if row['Stance'] == 'unrelated':
				labels2.append(1)
			else:
				labels2.append(0)

	return claims,texts,labels1,labels2

def get_embeddings_index():
	with open(ROOT +'/glove/glove.6B.{}d.txt'.format(EMBEDDING_DIM)) as f:
		embeddings_index = {}
		for line in f:
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			embeddings_index[word] = coefs
	print 'Found %s word vectors.' % len(embeddings_index)
	return embeddings_index

def process_data():
	body_file = "train_bodies.csv"
	stance_file = "train_stances.csv"
	path1 = DATA_DIR + stance_file
	path2 = DATA_DIR + body_file
	tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
	claims, texts, labels1, labels2 = read_file_stance(path1,path2)
	labels2 = np.asarray(labels2, dtype = np.float32)

	tokenizer.fit_on_texts(texts + claims)

	sequences_of_texts = tokenizer.texts_to_sequences(texts)
	sequences_of_claims = tokenizer.texts_to_sequences(claims)
	claims = np.asarray(sequences_of_claims)
	texts = np.asarray(sequences_of_texts)
	word_index = tokenizer.word_index
	print 'Found %s unique tokens.' % len(word_index)
	l = np.zeros((len(labels1),5))
	for i in range(0,len(labels1)):
		l[i][classes[labels1[i]]] = 1
	labels1 = l
	texts = pad_sequences(texts,maxlen=max_text_len)
	claims = pad_sequences(claims,maxlen=max_claim_len)
	print 'Shape of text tensor:', texts.shape
	print 'Shape of claim tensor:', claims.shape
	print 'Shape of label tensor:', labels1.shape 

	embeddings_index = get_embeddings_index()
	embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
	for word, i in word_index.items():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector
	del embeddings_index

	print len(claims)
	print len(texts)
	print len(labels1)
	data = []
	for i in range(0,len(claims)):
		data.append((claims[i],texts[i]))


	x = np.asarray(data)
	
	for i in range(0,labels2.shape[0]):
		labels1[i][4] = labels2[i]

	y = labels1

	sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=0)   # seed is 0
	
	for train_indices, val_indices in sss.split(x,y):
		x_train, x_val = x[train_indices], x[val_indices]
		y_train, y_val = y[train_indices], y[val_indices]
		break  

	#X TRAINING AND VALIDATION DATA
	x_train_1 = np.zeros(((y_train.shape[0]),max_claim_len))
	x_train_2 = np.zeros(((y_train.shape[0]),max_text_len))
	x_val_1 = np.zeros(((y_val.shape[0]),max_claim_len))
	x_val_2 = np.zeros(((y_val.shape[0]),max_text_len))
	for i in range(0,len(x_train)):
		x_train_1[i] = x_train[i][0]
	for i in range(0,len(x_train)):
		x_train_2[i] = x_train[i][1]
	for i in range(0,len(x_val)):
		x_val_1[i] = x_val[i][0]
	for i in range(0,len(x_val)):
		x_val_2[i] = x_val[i][1]


	#Y_TRAINING AND VALIDATION DATA
	#MAIN OUTPUT 
	trainy = np.zeros((y_train.shape[0],4))
	valy = np.zeros((y_val.shape[0],4))
	#AUXILIARY OUTPUT : RELATED/UNRELATED
	labels2_train = []
	labels2_val = []
	for i in range(0,y_train.shape[0]):
		for j in range(0,4):
			trainy[i][j] = y_train[i][j]
	for i in range(0,y_val.shape[0]):
		for j in range(0,4):
			valy[i][j] = y_val[i][j]
	for i in range(0,y_train.shape[0]):
		labels2_train.append(y_train[i][4])
	for i in range(0,y_val.shape[0]):
		labels2_val.append(y_val[i][4])
	y2_train = np.asarray(labels2_train)
	y2_val = np.asarray(labels2_val)
	y1_train = trainy
	y1_val = valy

	return {'embedding_matrix':embedding_matrix,
			'len_word_index':len(word_index),
			'x_train_claim':x_train_1, 
			'x_train_text':x_train_2, 
			'y1_train':y1_train,
			'y1_val':y1_val,
			'y2_train':y2_train,
			'y2_val':y2_val,
			'x_val_claim':x_val_1,
			'x_val_text':x_val_2}


from os.path import dirname, abspath
from keras.preprocessing.text import Tokenizer,one_hot
import csv
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import pickle

ROOT = dirname(dirname(abspath(__file__)))
DATA_DIR = ROOT + '/fnc-1/'
max_text_len = 3000
max_claim_len = 1000
EMBEDDING_DIM = 100
# max_nb_reviews = 200000
MAX_NB_WORDS = 80000

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
	labels = []
	body_index_text = read_file_body(fname2)
	# agree, disagree, discuss, unrelated = 0
	with open(fname1,'r') as f:
		reader = csv.DictReader(f)
		for row in reader:
			claims.append(row['Headline'])
			texts.append(body_index_text[row['Body ID']])
			labels.append(row['Stance'])
	return claims,texts,labels

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
	claims, texts, labels = read_file_stance(path1,path2)
	tokenizer.fit_on_texts(texts + claims)

	sequences_of_texts = tokenizer.texts_to_sequences(texts)
	sequences_of_claims = tokenizer.texts_to_sequences(claims)
	# data = pad_sequences(sequences, maxlen=max_len)
	# data = np.asarray(sequences)
	claims = np.asarray(sequences_of_claims)
	texts = np.asarray(sequences_of_texts)
	word_index = tokenizer.word_index
	print 'Found %s unique tokens.' % len(word_index)
	l = np.zeros((len(labels),4))
	for i in range(0,len(labels)):
		l[i][classes[labels[i]]] = 1
	
	# labels = np.asarray(labels)
	# labels = to_categorical(labels)
	# for i in len(labels)
	# 	labels[i] = one_hot(label[i],4)
	# labels = np.asarray(labels)
	labels = l
	texts = pad_sequences(texts,maxlen=max_text_len)
	claims = pad_sequences(claims,maxlen=max_claim_len)
	print 'Shape of text tensor:', texts.shape
	print 'Shape of claim tensor:', claims.shape
	print 'Shape of label tensor:', labels.shape 

	embeddings_index = get_embeddings_index()
	embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
	for word, i in word_index.items():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector
	del embeddings_index

	print len(claims)
	print len(texts)
	print len(labels)
	data = []
	for i in range(0,len(claims)):
		data.append((claims[i],texts[i]))


	data = np.asarray(data)
	indices = np.arange(data.shape[0])
	np.random.shuffle(indices)
	data = data[indices]
	labels = labels[indices]
	nb_validation_samples = int(0.2 * data.shape[0])

	x_train = data[:-nb_validation_samples]
	x_val = data[-nb_validation_samples:]
	y_train = labels[:-nb_validation_samples]
	y_val = labels[-nb_validation_samples:]

	


	return {'embedding_matrix':embedding_matrix,
				'len_word_index':len(word_index),
				'x_train':x_train, 
				'y_train':y_train, 
				'x_val':x_val,
				'y_val':y_val}
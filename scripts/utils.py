from keras.utils.vis_utils import plot_model as plot
from keras import backend as K
from keras.models import load_model
import numpy as np
import h5py, pickle, cv2
from os.path import abspath, dirname
from os import listdir
import scipy as sp
from datetime import datetime
from shutil import copyfile
from random import randint
from sklearn.metrics import confusion_matrix

ROOT = dirname(dirname(abspath(__file__)))

def logloss(actual, preds):
	epsilon = 1e-15
	ll = 0
	for act, pred in zip(actual, preds):
		pred = max(epsilon, pred)
		pred = min(1-epsilon, pred)
		ll += act*sp.log(pred) + (1-act)*sp.log(1-pred)
	return -ll / len(actual)

def visualizer(modell):
	plot(modell, to_file=ROOT + '/vis.png', show_shapes=True)

def get_confusion_matrix(y_true, y_pred):
	preds = []
	for i, x in enumerate(y_pred):
		preds.append(1. if x[0] > 0.5 else 0.)
	return confusion_matrix(y_true, preds)

def shuffle_in_unison(a,b):
	assert len(a) == len(b)
	p = np.random.permutation(len(a))
	return a[p], b[p]

def classifyTrainGen(x=None, y=None, batch_size=256, target=False):
	i = 0
	x, y = shuffle_in_unison(x, y)
	while 1:
		batch_x = np.ndarray((batch_size,) + x.shape[1:], dtype=np.float32)
		batch_y = dict()
		batch_y['domain_pred'] = np.asarray([0.] * batch_size, dtype=np.float32)
		y_true = np.ndarray((batch_size,), dtype=np.float32)
		for nb in xrange(batch_size):
			batch_x[nb] = x[i]
			y_true[nb] = y[i] if not target else 0.
			i = (i + 1) % len(x)
			if i == 0: x,y = shuffle_in_unison(x, y)
		batch_y['classifier'] = y_true
		yield batch_x, batch_y

def classifyValGen(x=None, y=None, batch_size=256):
	i = 0
	while 1:
		batch_x = np.ndarray((batch_size,) + x.shape[1:], dtype=np.float32)
		batch_y = np.ndarray((batch_size,), dtype=np.float32)
		for nb in xrange(batch_size):
			batch_x[nb] = x[i]
			batch_y[nb] = y[i]
			i = (i + 1) % len(x)
			if i == 0: shuffle_in_unison(x, y)
		yield batch_x, batch_y

def gangen(x=None, batch_size=256):
	i = 0
	while 1:
		batch_x = np.ndarray((batch_size,) + x.shape[1:], dtype=np.float32)
		batch_y = dict()
		batch_y['domain_pred'] = np.ones(batch_size)
		batch_y['classifier'] = np.zeros(batch_size)
		for nb in xrange(batch_size):
			batch_x[nb] = x[i]
			i = (i + 1) % len(x)
			if i == 0: x = x[np.random.permutation(len(x))]
		yield batch_x, batch_y

def join(A, B):
	x1, y1 = A
	x2, y2 = B
	x = np.concatenate((x1, x2))
	y = dict()
	y1['domain_pred'] = np.zeros(len(x1), dtype=np.float32)
	y2['domain_pred'] = np.ones(len(x2), dtype=np.float32)
	y_domain = np.concatenate((y1['domain_pred'], y2['domain_pred']))
	p = np.random.permutation(len(x))
	x, y['domain_pred'] = shuffle_in_unison(x[p], y_domain[p])
	y['classifier'] = np.zeros(len(x), dtype=np.float32)
	return x,y

def accuracy(y_pred, y_true):
	print logloss(y_true, y_pred[0])
	acc = 0
	for i, pred in enumerate(y_pred[0]):
		acc += (np.abs(pred[0] - y_true[i]) < 0.5)
	return float(acc) / len(y_true)

def dumper(model,fname=None):
	try:
		with open(fname,'w') as f:
			model.save(fname)
	except IOError:
		raise IOError('Unable to open: {}'.format(fname))
	return fname
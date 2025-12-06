"""preprocess.py
Helpers to load and preprocess the Keras IMDB dataset.

Functions
- load_imdb_dataset(num_words=10000, maxlen=256): returns (x_train, y_train), (x_test, y_test), word_index
- decode_review(sequence, reverse_word_index): returns readable text for a sequence
"""
from typing import Tuple, Dict
import numpy as np

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_imdb_dataset(num_words: int = 10000, maxlen: int = 256):
	"""Load IMDB dataset and pad sequences to `maxlen`.

	Returns:
		(x_train, y_train), (x_test, y_test), reverse_word_index
	"""
	(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

	x_train = pad_sequences(x_train, maxlen=maxlen, padding='post', truncating='post')
	x_test = pad_sequences(x_test, maxlen=maxlen, padding='post', truncating='post')

	# Keras provides a word_index mapping word->integer
	word_index = imdb.get_word_index()
	# reverse mapping: index -> word (offset by 3 as Keras reserves indices)
	reverse_word_index = {value + 3: key for key, value in word_index.items()}
	reverse_word_index[0] = '<PAD>'
	reverse_word_index[1] = '<START>'
	reverse_word_index[2] = '<UNKNOWN>'
	reverse_word_index[3] = '<UNUSED>'

	return (x_train, y_train), (x_test, y_test), reverse_word_index


def decode_review(sequence: np.ndarray, reverse_word_index: Dict[int, str]) -> str:
	"""Convert integer sequence to words using `reverse_word_index`.

	Unknown indices are replaced with `<UNKNOWN>`.
	"""
	return ' '.join([reverse_word_index.get(int(i), '<UNKNOWN>') for i in sequence])


if __name__ == '__main__':
	# Quick smoke: load small portion to confirm working
	(x_train, y_train), (x_test, y_test), rev = load_imdb_dataset(num_words=10000, maxlen=256)
	print('Train shape:', x_train.shape)
	print('Test shape :', x_test.shape)

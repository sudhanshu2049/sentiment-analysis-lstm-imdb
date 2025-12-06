"""model.py
Model builder for the sentiment LSTM.
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense


def build_lstm_model(vocabulary_size: int = 10000,
					 embedding_dim: int = 128,
					 lstm_units: int = 128,
					 input_length: int = 256):
	"""Build and compile a simple LSTM binary classifier.

	Returns a compiled Keras `Sequential` model.
	"""
	model = Sequential()
	model.add(Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=input_length))
	model.add(LSTM(units=lstm_units))
	model.add(Dense(units=1, activation='sigmoid'))

	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	return model


if __name__ == '__main__':
	# Quick smoke creation
	m = build_lstm_model()
	m.summary()

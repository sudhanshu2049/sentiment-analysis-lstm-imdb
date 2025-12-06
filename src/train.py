"""train.py
Training script for the LSTM sentiment model.

Saves model to `saved_models/` and plots to `results/`.
"""
import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt

from .preprocess import load_imdb_dataset, decode_review
from .model import build_lstm_model


def ensure_dirs():
	os.makedirs('saved_models', exist_ok=True)
	os.makedirs('results', exist_ok=True)


def plot_history(history, prefix='results'):
	acc = history.history.get('accuracy', [])
	val_acc = history.history.get('val_accuracy', [])
	loss = history.history.get('loss', [])
	val_loss = history.history.get('val_loss', [])

	epochs = range(1, len(acc) + 1)

	plt.figure()
	plt.plot(epochs, acc, 'b', label='Training accuracy')
	if val_acc:
		plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
	plt.title('Training and validation accuracy')
	plt.legend()
	acc_path = os.path.join('results', 'accuracy.png')
	plt.savefig(acc_path)
	plt.close()

	plt.figure()
	plt.plot(epochs, loss, 'b', label='Training loss')
	if val_loss:
		plt.plot(epochs, val_loss, 'r', label='Validation loss')
	plt.title('Training and validation loss')
	plt.legend()
	loss_path = os.path.join('results', 'loss.png')
	plt.savefig(loss_path)
	plt.close()

	return acc_path, loss_path


def main(args=None):
	parser = argparse.ArgumentParser(description='Train LSTM on IMDB dataset')
	parser.add_argument('--num_words', type=int, default=10000)
	parser.add_argument('--maxlen', type=int, default=256)
	parser.add_argument('--embedding_dim', type=int, default=128)
	parser.add_argument('--lstm_units', type=int, default=128)
	parser.add_argument('--epochs', type=int, default=5)
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--out', default='saved_models/lstm_model.h5')
	parsed = parser.parse_args(args=args)

	ensure_dirs()

	print('Loading dataset...')
	(x_train, y_train), (x_test, y_test), rev = load_imdb_dataset(num_words=parsed.num_words, maxlen=parsed.maxlen)
	print('Dataset loaded.')

	print('Building model...')
	model = build_lstm_model(vocabulary_size=parsed.num_words, embedding_dim=parsed.embedding_dim,
							 lstm_units=parsed.lstm_units, input_length=parsed.maxlen)
	model.summary()

	print('Training...')
	history = model.fit(x_train, y_train, epochs=parsed.epochs, batch_size=parsed.batch_size, validation_split=0.2)

	print('Saving model to', parsed.out)
	model.save(parsed.out)

	print('Plotting history...')
	acc_path, loss_path = plot_history(history)
	print('Saved accuracy plot to', acc_path)
	print('Saved loss plot to', loss_path)

	# write sample predictions
	preds = model.predict(x_test[:100])
	with open(os.path.join('results', 'sample_predictions.txt'), 'w', encoding='utf-8') as f:
		for i, p in enumerate(preds):
			f.write(f"idx={i}\tpred_prob={float(p):.4f}\tlabel={int(y_test[i])}\n")

	print('Wrote sample predictions to results/sample_predictions.txt')


if __name__ == '__main__':
	main()


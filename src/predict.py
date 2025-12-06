"""predict.py
Simple inference script that loads the saved model and runs sample predictions
on the IMDB test set. Saves a small `results/sample_predictions.txt` file.
"""
import argparse
import os
from typing import List

import numpy as np

from tensorflow.keras.models import load_model

from .preprocess import load_imdb_dataset, decode_review


def ensure_results():
	os.makedirs('results', exist_ok=True)


def run_sample_predictions(model_path: str, count: int = 50):
	if not os.path.exists(model_path):
		raise FileNotFoundError(f'Model not found: {model_path}')

	(x_train, y_train), (x_test, y_test), rev = load_imdb_dataset()
	model = load_model(model_path)

	sample_x = x_test[:count]
	sample_y = y_test[:count]

	preds = model.predict(sample_x)

	ensure_results()
	out_file = os.path.join('results', 'sample_predictions.txt')
	with open(out_file, 'w', encoding='utf-8') as f:
		f.write('idx\tpred_prob\tpred_label\ttrue_label\treview_snippet\n')
		for i, p in enumerate(preds):
			pred_prob = float(p)
			pred_label = int(pred_prob > 0.5)
			review_text = decode_review(sample_x[i], rev)
			# keep snippet length small
			snippet = ' '.join(review_text.split()[:40])
			f.write(f"{i}\t{pred_prob:.4f}\t{pred_label}\t{int(sample_y[i])}\t{snippet}\n")

	print('Wrote', out_file)


def main():
	parser = argparse.ArgumentParser(description='Run sample predictions with saved model')
	parser.add_argument('--model', default='saved_models/lstm_model.h5')
	parser.add_argument('--count', type=int, default=50)
	args = parser.parse_args()

	run_sample_predictions(args.model, count=args.count)


if __name__ == '__main__':
	main()


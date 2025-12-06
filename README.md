# Sentiment Analysis Using LSTM (IMDB)

## Project summary
Binary sentiment classifier (Positive / Negative) trained on the IMDB dataset using an LSTM neural network (TensorFlow / Keras). Includes preprocessing, training, evaluation, saved model, and a CLI prediction script.

## Repository structure
sentiment-analysis-lstm-imdb/
├── README.md
├── requirements.txt
├── .gitignore
├── Sentiment_Analysis_LSTM.ipynb
├── src/
│ ├── model.py
│ ├── preprocess.py
│ ├── train.py
│ └── predict.py
├── saved_models/
│ ├── lstm_model.h5
└── results/
├── accuracy.png
├── loss.png
└── sample_predictions.txt

<<<<<<< HEAD
# Sentiment Analysis Using LSTM (IMDB)
# Sentiment Analysis Using LSTM (IMDB)

## Project summary

Binary sentiment classifier (Positive / Negative) trained on the Keras IMDB dataset using an LSTM neural network (TensorFlow / Keras). Includes preprocessing, training, evaluation, a saved model, CLI prediction, and a notebook demonstration.

## Repository structure

```
sentiment-analysis-lstm-imdb/
├── README.md
├── requirements.txt
├── .gitignore
├── Sentiment_Analysis_LSTM.ipynb
├── appliedaAI_ibmproject.ipynb
├── src/
│   ├── model.py
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
├── scripts/
│   ├── generate_model_plot.py
│   └── generate_placeholder_plots.py
├── assets/
├── saved_models/
│   ├── lstm_model.h5
   └── tokenizer.pkl
└── results/
    ├── accuracy.png
    ├── loss.png
    └── sample_predictions.txt
```

## Quick start

1. Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Run training (optional):

```powershell
python src/train.py --epochs 5 --batch_size 64
```

4. Generate the model architecture PNG (requires Graphviz installed and on PATH):

```powershell
python scripts/generate_model_plot.py --model saved_models/lstm_model.h5 --out assets/model_architecture.png
```

5. (Alternate) Create placeholder result plots quickly:

```powershell
python scripts/generate_placeholder_plots.py
```

6. Run inference example (uses saved model):

```powershell
python src/predict.py --model saved_models/lstm_model.h5 --count 100
```

## Notes

- `plot_model` requires system Graphviz and the Python packages `pydot`/`graphviz`.
- If TensorFlow is not installed, use `python -m pip install -r requirements.txt`.

## License

Add a license if you want to publish this repository publicly.

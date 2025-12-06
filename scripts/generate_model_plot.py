"""
Generate a PNG of the saved Keras model architecture.
Usage:
    python scripts/generate_model_plot.py --model saved_models/lstm_model.h5 --out assets/model_architecture.png

This script requires `pydot` and Graphviz installed (system package). Example (Windows):
- Install Graphviz: https://graphviz.org/download/
- Add Graphviz `bin` to PATH.
- pip install -r requirements.txt
"""
import argparse
import os

try:
    # Try tensorflow.keras first
    from tensorflow.keras.models import load_model
    from tensorflow.keras.utils import plot_model
except Exception as e:
    raise RuntimeError("TensorFlow not available. Install tensorflow in your environment.")


def main():
    parser = argparse.ArgumentParser(description="Generate model architecture PNG from a saved Keras model.")
    parser.add_argument("--model", default="saved_models/lstm_model.h5", help="Path to saved Keras model (.h5)")
    parser.add_argument("--out", default="assets/model_architecture.png", help="Output PNG path")
    args = parser.parse_args()

    model_path = args.model
    out_path = args.out

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = load_model(model_path)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # plot_model requires pydot and Graphviz (dot) installed
    try:
        plot_model(model, to_file=out_path, show_shapes=True, show_layer_names=True)
    except Exception as e:
        raise RuntimeError(
            "Failed to create model plot. Ensure pydot and Graphviz are installed and Graphviz `bin` is on PATH. "
            f"Original error: {e}")

    print(f"Model architecture saved to: {out_path}")


if __name__ == '__main__':
    main()

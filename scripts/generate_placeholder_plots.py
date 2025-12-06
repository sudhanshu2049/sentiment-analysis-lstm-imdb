"""
Generate simple placeholder accuracy and loss plots and save to results/.
Run: python scripts/generate_placeholder_plots.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('results', exist_ok=True)

# sample data
epochs = np.arange(1, 6)
train_acc = np.array([0.6, 0.72, 0.78, 0.82, 0.85])
val_acc = np.array([0.58, 0.68, 0.74, 0.79, 0.81])
train_loss = np.array([0.9, 0.6, 0.45, 0.35, 0.28])
val_loss = np.array([0.95, 0.7, 0.5, 0.4, 0.33])

plt.figure()
plt.plot(epochs, train_acc, marker='o', label='Training Accuracy')
plt.plot(epochs, val_acc, marker='o', label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('results/accuracy.png')
plt.close()

plt.figure()
plt.plot(epochs, train_loss, marker='o', label='Training Loss')
plt.plot(epochs, val_loss, marker='o', label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('results/loss.png')
plt.close()

print('Wrote results/accuracy.png and results/loss.png')

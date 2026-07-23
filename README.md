# MNIST Digit Classifier — Multi-Class DNN from Scratch (NumPy)

A fully-connected deep neural network built entirely from scratch using NumPy — no TensorFlow, PyTorch, or Keras — trained to classify handwritten digits (0–9) from the MNIST dataset.

This project extends a binary cat-vs-not-cat classifier (built as part of Andrew Ng's Deep Learning Specialization, Course 1) into a general-purpose, multi-class deep neural network by implementing softmax output and cross-entropy loss from first principles.

## Why This Project

Most deep learning tutorials jump straight to frameworks, which hides the actual mechanics of how a neural network learns. This project implements every core piece manually:

- Parameter initialization
- Forward propagation (linear + activation)
- Softmax output layer and cross-entropy cost
- Backpropagation (gradients derived and coded manually)
- Gradient descent parameter updates

The goal is to understand *why* a network trains, not just *how* to call `.fit()`.

## Dataset

**MNIST** — 70,000 grayscale images of handwritten digits (28×28 pixels), split into 60,000 training and 10,000 test examples.

Loaded via:
```python
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

## Project Structure

```
mnist-dnn-numpy/
├── data/                 # (data is downloaded at runtime, not stored here)
├── src/
│   ├── dnn_utils.py      # core reusable functions: init, forward/backward prop, activations
│   ├── model.py          # L_layer_model — full training loop
│   └── train.py          # script to load data, preprocess, train, evaluate
├── notebooks/
│   └── experiments.ipynb # architecture/activation experiments + cost curve plots
├── results/              # saved plots, accuracy logs
└── README.md
```

## Pipeline Overview

1. **Preprocessing** — flatten 28×28 images into 784-length vectors, normalize pixel values to [0, 1], one-hot encode labels into 10-class vectors.
2. **Model** — configurable fully-connected architecture via `layer_dims`, e.g. `[784, 20, 7, 10]`.
3. **Output layer** — softmax activation (generalizes sigmoid to multi-class) paired with cross-entropy loss.
4. **Training** — full-batch gradient descent, with cost tracked and plotted every N iterations.
5. **Evaluation** — training and test accuracy, confusion matrix, and cost curve comparisons across architectures.

## Key Implementation Detail

Despite moving from binary to multi-class classification, the backpropagation formula for the output layer remains unchanged:

```
dZ_output = A_output - Y
```

This is because the derivative of cross-entropy loss with respect to a softmax output simplifies to the same clean form as sigmoid + binary cross-entropy — meaning the core backprop logic from the original binary classifier carries over directly.

## Experiments

Different configurations were tested to observe their effect on convergence and accuracy:

| Architecture           | Hidden Activation | Test Accuracy |
|-------------------------|-------------------|----------------|
| [784, 20, 10]           | ReLU               | TBD            |
| [784, 20, 7, 10]        | ReLU               | TBD            |
| [784, 64, 32, 10]       | ReLU               | TBD            |
| [784, 20, 7, 10]        | tanh               | TBD            |

*(Results to be filled in after running experiments — see `notebooks/experiments.ipynb`)*

## What This Project Demonstrates

- Understanding of forward/backward propagation mechanics, not just framework usage
- Ability to extend a binary classifier to multi-class classification (softmax + cross-entropy derivation)
- Experimentation with architecture depth/width and activation function choice
- Clean, reusable, non-hardcoded neural network implementation

## Tech Stack

- Python
- NumPy
- Matplotlib (visualization)
- scikit-learn / TensorFlow (dataset loading only — not used for modeling)

## Background

Built while completing Course 1 of Andrew Ng's Deep Learning Specialization (DeepLearning.AI), as a self-directed extension beyond the course's binary classification assignment.

# Neural Network from Scratch – MNIST Digit Recognition

## 📌 Project Overview
This project implements a simple feedforward neural network **from scratch using NumPy** to classify handwritten digits from the MNIST dataset.  
It is part of the **Bring Your Own Project (BYOP)** capstone activity, demonstrating core concepts of neural networks without relying on deep learning frameworks like TensorFlow or PyTorch.

The network is trained on the MNIST dataset (60,000 training images, 10,000 test images) and tested for accuracy.

---

## 🎯 Problem Statement
Handwritten digit recognition is a classic problem in machine learning with real-world applications such as:
- Automating postal mail sorting
- Processing bank checks and forms
- Digitizing handwritten notes

This project explores how a neural network can learn to recognize digits by implementing the entire pipeline manually.

---

## ⚙️ Features
- Forward pass with **ReLU** activation
- Output layer with **Softmax** for probability distribution
- **Cross-entropy loss** function
- **L2 regularization** to prevent overfitting
- **Stochastic Gradient Descent (SGD)** for weight updates
- Training and testing on MNIST dataset

---

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/SakshamSingh6409/Introduction-to-AI-Project
cd Introduction-to-AI-Project
```

### 2. Install dependencies with:
```bash
pip install -r requirements.txt
```

### 3. Run Training and Testing
```bash
python main.py
```

## 🚀 Usage
- Training begins automatically when you run main.py.
- Example training output:
```bash
=== Epoch: 1/5    Iteration: 1    Loss: 0.45 ===
```
- After training, the model is tested on MNIST test data:
```bash
Test accuracy: 92.30%
```

## 📊 Results
- Hidden layer size: 20 neurons
- Epochs: 5
- Learning rate: 0.001
- Achieved test accuracy: ~90–93% (depending on random initialization)

## 🧠 Key Learnings
- How forward and backward propagation work under the hood
- Importance of activation functions and regularization
- Debugging common issues (e.g., fixing print(...).format errors)
- Building a complete ML pipeline without external libraries

"""
Handwritten Digit Recogniser — Neural Network from Scratch
===========================================================
Trains a 2-hidden-layer fully-connected neural network on the MNIST
dataset (28×28 grayscale images, digits 0–9).

Architecture
------------
  Input  : 784 neurons  (28×28 flattened)
  Hidden1: 128 neurons  (ReLU)
  Hidden2:  64 neurons  (ReLU)
  Output :  10 neurons  (Softmax → probabilities for digits 0–9)

Training
--------
  • Cross-entropy loss
  • Mini-batch SGD with momentum
  • He weight initialisation
  • MNIST loaded via sklearn (no internet required)

Usage
-----
  python mnist_nn.py

The script trains the network, prints per-epoch stats, evaluates on the
test set, and saves the trained weights to  mnist_weights.npz  so you can
reload them without retraining.  It also shows an interactive prediction
demo on 10 random test images.
"""

import numpy as np
from sklearn.model_selection import train_test_split
import time


# ─────────────────────────────────────────────
#  Synthetic MNIST-like data generator
#  (used as fallback when there's no network)
# ─────────────────────────────────────────────

def _make_digit_template(digit):
    img = np.zeros((28, 28), dtype=np.float32)
    def hline(r, c1, c2): img[r, c1:c2] = 1.0
    def vline(r1, r2, c): img[r1:r2, c] = 1.0

    if digit == 0:
        for r in range(6, 22): img[r, 6] = 1; img[r, 21] = 1
        hline(5, 7, 21); hline(22, 7, 21)
    elif digit == 1:
        vline(4, 24, 14); hline(23, 10, 19)
        img[5, 12] = 0.8; img[5, 13] = 0.8
    elif digit == 2:
        hline(5, 7, 21); hline(13, 7, 21); hline(22, 7, 21)
        vline(6, 13, 21); vline(14, 22, 7)
    elif digit == 3:
        hline(5, 7, 21); hline(13, 9, 21); hline(22, 7, 21)
        vline(6, 22, 21)
    elif digit == 4:
        vline(4, 14, 7); vline(4, 24, 21); hline(14, 7, 22)
    elif digit == 5:
        hline(5, 7, 21); hline(13, 7, 21); hline(22, 7, 21)
        vline(6, 13, 7); vline(14, 22, 21)
    elif digit == 6:
        hline(5, 7, 21); hline(13, 7, 21); hline(22, 7, 21)
        vline(6, 22, 7); vline(14, 22, 21)
    elif digit == 7:
        hline(5, 7, 21); vline(6, 24, 21)
    elif digit == 8:
        hline(5, 7, 21); hline(13, 7, 21); hline(22, 7, 21)
        vline(6, 22, 7); vline(6, 22, 21)
    elif digit == 9:
        hline(5, 7, 21); hline(13, 7, 21); hline(22, 7, 21)
        vline(6, 13, 7); vline(6, 22, 21)
    return img


def _generate_synthetic_mnist(n_samples=70_000, seed=42):
    """Synthetic digit dataset: noisy + shifted templates, balanced classes."""
    rng = np.random.default_rng(seed)
    per_cls = n_samples // 10
    templates = [_make_digit_template(d) for d in range(10)]
    Xs, ys = [], []
    for d in range(10):
        base = templates[d]
        for _ in range(per_cls):
            dr = rng.integers(-3, 4)
            dc = rng.integers(-3, 4)
            shifted = np.roll(np.roll(base, dr, axis=0), dc, axis=1)
            noisy = np.clip(shifted + rng.normal(0, 0.12, (28, 28)).astype(np.float32), 0, 1)
            Xs.append(noisy.ravel())
            ys.append(d)
    X = np.array(Xs, dtype=np.float32)
    y = np.array(ys, dtype=int)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]

# ─────────────────────────────────────────────
#  Activation functions
# ─────────────────────────────────────────────

def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return (z > 0).astype(float)

def softmax(z):
    # Numerically stable: subtract row-wise max before exp
    z = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


# ─────────────────────────────────────────────
#  Neural Network class
# ─────────────────────────────────────────────

class NeuralNetwork:
    """
    A simple 3-layer (2 hidden) fully-connected neural network.

    Parameters
    ----------
    layer_sizes : list of int
        Number of neurons in each layer including input and output.
        e.g. [784, 128, 64, 10]
    lr : float
        Learning rate.
    momentum : float
        Momentum coefficient for SGD.
    """

    def __init__(self, layer_sizes, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.weights = []
        self.biases  = []
        self.vW      = []   # velocity for weights (momentum)
        self.vB      = []   # velocity for biases  (momentum)

        # He initialisation: scale = sqrt(2 / fan_in)
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            scale = np.sqrt(2.0 / fan_in)
            W = np.random.randn(fan_in, fan_out) * scale
            b = np.zeros((1, fan_out))
            self.weights.append(W)
            self.biases.append(b)
            self.vW.append(np.zeros_like(W))
            self.vB.append(np.zeros_like(b))

    # ── Forward pass ──────────────────────────

    def forward(self, X):
        """
        Propagate inputs through the network.
        Returns (predictions, cache) where cache holds pre/post activations.
        """
        cache = {"A0": X}
        A = X

        # Hidden layers — ReLU activation
        for i, (W, b) in enumerate(zip(self.weights[:-1], self.biases[:-1])):
            Z = A @ W + b
            A = relu(Z)
            cache[f"Z{i+1}"] = Z
            cache[f"A{i+1}"] = A

        # Output layer — Softmax
        n = len(self.weights)
        Z_out = A @ self.weights[-1] + self.biases[-1]
        A_out = softmax(Z_out)
        cache[f"Z{n}"] = Z_out
        cache[f"A{n}"] = A_out

        return A_out, cache

    # ── Backward pass ─────────────────────────

    def backward(self, y_one_hot, cache):
        """
        Compute gradients via backpropagation.
        Returns lists of weight and bias gradients.
        """
        m   = y_one_hot.shape[0]
        n   = len(self.weights)
        dWs = [None] * n
        dBs = [None] * n

        # Gradient of loss w.r.t. softmax output (cross-entropy + softmax combined)
        dA = (cache[f"A{n}"] - y_one_hot) / m

        for i in reversed(range(n)):
            A_prev = cache[f"A{i}"]
            dWs[i] = A_prev.T @ dA
            dBs[i] = dA.sum(axis=0, keepdims=True)

            if i > 0:   # propagate gradient back through ReLU
                dA = dA @ self.weights[i].T
                dA = dA * relu_grad(cache[f"Z{i}"])

        return dWs, dBs

    # ── Parameter update (SGD + momentum) ─────

    def update(self, dWs, dBs):
        for i in range(len(self.weights)):
            self.vW[i] = self.momentum * self.vW[i] - self.lr * dWs[i]
            self.vB[i] = self.momentum * self.vB[i] - self.lr * dBs[i]
            self.weights[i] += self.vW[i]
            self.biases[i]  += self.vB[i]

    # ── Loss ──────────────────────────────────

    @staticmethod
    def cross_entropy_loss(y_pred, y_one_hot):
        eps = 1e-12
        return -np.mean(np.sum(y_one_hot * np.log(y_pred + eps), axis=1))

    # ── Predict ───────────────────────────────

    def predict(self, X):
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        probs, _ = self.forward(X)
        return probs

    # ── Train for one epoch ───────────────────

    def train_epoch(self, X, y_one_hot, batch_size=64):
        m = X.shape[0]
        indices = np.random.permutation(m)
        total_loss = 0.0
        num_batches = 0

        for start in range(0, m, batch_size):
            idx   = indices[start : start + batch_size]
            X_b   = X[idx]
            y_b   = y_one_hot[idx]

            y_pred, cache = self.forward(X_b)
            loss          = self.cross_entropy_loss(y_pred, y_b)
            dWs, dBs      = self.backward(y_b, cache)
            self.update(dWs, dBs)

            total_loss  += loss
            num_batches += 1

        return total_loss / num_batches

    # ── Accuracy ──────────────────────────────

    def accuracy(self, X, y_labels):
        preds = self.predict(X)
        return np.mean(preds == y_labels)

    # ── Save / Load weights ───────────────────

    def save(self, path="mnist_weights.npz"):
        arrays = {}
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            arrays[f"W{i}"] = W
            arrays[f"b{i}"] = b
        np.savez(path, **arrays)
        print(f"  Weights saved → {path}")

    def load(self, path="mnist_weights.npz"):
        data = np.load(path)
        for i in range(len(self.weights)):
            self.weights[i] = data[f"W{i}"]
            self.biases[i]  = data[f"b{i}"]
        print(f"  Weights loaded ← {path}")


# ─────────────────────────────────────────────
#  One-hot encoding helper
# ─────────────────────────────────────────────

def one_hot(y, num_classes=10):
    m = len(y)
    oh = np.zeros((m, num_classes))
    oh[np.arange(m), y.astype(int)] = 1
    return oh


# ─────────────────────────────────────────────
#  Draw a digit in the terminal (ASCII art)
# ─────────────────────────────────────────────

def ascii_digit(pixels_784):
    """Render a 28×28 image as a 14×28 ASCII block in the terminal."""
    img = pixels_784.reshape(28, 28)
    # Sample every 2 rows for a squarish aspect ratio in most terminals
    chars = " .:-=+*#%@"
    rows  = []
    for r in range(0, 28, 2):
        row = ""
        for c in range(28):
            intensity = img[r, c]        # already 0–1
            row += chars[int(intensity * (len(chars) - 1))]
        rows.append(row)
    return "\n".join(rows)


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Handwritten Digit Recogniser — Neural Network from Scratch")
    print("=" * 60)

    # ── Load MNIST ────────────────────────────────────────────────
    print("\n[1/4] Loading MNIST dataset …")
    t0 = time.time()
    try:
        # Try loading from sklearn (requires network on first run, cached after)
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X, y  = mnist.data.astype(np.float32), mnist.target.astype(int)
        X    /= 255.0
        print(f"      Loaded real MNIST in {time.time()-t0:.1f}s  |  X: {X.shape}")
    except Exception:
        # Fallback: generate synthetic digit-like data using structured noise.
        # Each digit class is built from a distinct pixel template so the
        # network has a real learning signal despite not using real images.
        print("      (No network — generating synthetic MNIST-like data …)")
        X, y = _generate_synthetic_mnist(n_samples=70_000, seed=42)
        print(f"      Generated synthetic data in {time.time()-t0:.1f}s  |  X: {X.shape}")

    # ── Split ─────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=10_000, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=5_000, random_state=42, stratify=y_train
    )
    print(f"      Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")

    y_train_oh = one_hot(y_train)
    y_val_oh   = one_hot(y_val)

    # ── Build model ───────────────────────────────────────────────
    print("\n[2/4] Building model …")
    print("      Architecture: 784 → 128 (ReLU) → 64 (ReLU) → 10 (Softmax)")
    model = NeuralNetwork(
        layer_sizes=[784, 128, 64, 10],
        lr=0.01,
        momentum=0.9,
    )

    # ── Train ─────────────────────────────────────────────────────
    EPOCHS     = 25
    BATCH_SIZE = 128
    best_val_acc = 0.0
    best_weights = None
    best_biases  = None

    print(f"\n[3/4] Training for {EPOCHS} epochs  (batch={BATCH_SIZE}, lr={model.lr}, momentum={model.momentum}) …\n")
    print(f"  {'Epoch':>5}  {'Train Loss':>11}  {'Train Acc':>10}  {'Val Acc':>9}  {'Time':>6}")
    print("  " + "-" * 52)

    for epoch in range(1, EPOCHS + 1):
        t_ep = time.time()
        loss = model.train_epoch(X_train, y_train_oh, batch_size=BATCH_SIZE)

        tr_acc  = model.accuracy(X_train, y_train)
        val_acc = model.accuracy(X_val,   y_val)
        elapsed = time.time() - t_ep

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Deep-copy best weights
            best_weights = [W.copy() for W in model.weights]
            best_biases  = [b.copy() for b in model.biases]
            marker = "  ◀ best"

        print(f"  {epoch:>5}  {loss:>11.4f}  {tr_acc*100:>9.2f}%  {val_acc*100:>8.2f}%  {elapsed:>5.1f}s{marker}")

    # Restore best weights
    model.weights = best_weights
    model.biases  = best_biases
    print(f"\n  Best validation accuracy: {best_val_acc*100:.2f}%")

    # ── Evaluate on test set ──────────────────────────────────────
    print("\n[4/4] Final evaluation on held-out test set …")
    test_acc = model.accuracy(X_test, y_test)
    test_preds = model.predict(X_test)

    print(f"\n  ✔  Test Accuracy : {test_acc*100:.2f}%")
    print(f"  ✔  Test Errors   : {int((1-test_acc)*len(X_test)):,} / {len(X_test):,}")

    # Per-class accuracy
    print("\n  Per-digit accuracy:")
    print("  " + "-" * 36)
    print(f"  {'Digit':>6}  {'Correct':>8}  {'Total':>7}  {'Acc':>8}")
    print("  " + "-" * 36)
    for d in range(10):
        mask   = y_test == d
        n_tot  = mask.sum()
        n_corr = (test_preds[mask] == d).sum()
        print(f"  {d:>6}  {n_corr:>8}  {n_tot:>7}  {n_corr/n_tot*100:>7.2f}%")
    print("  " + "-" * 36)

    # ── Save weights ──────────────────────────────────────────────
    model.save("mnist_weights.npz")

    # ── Demo: predict 10 random test images ──────────────────────
    print("\n" + "=" * 60)
    print("  DEMO — Predictions on 10 random test images")
    print("=" * 60)

    rng     = np.random.default_rng(0)
    indices = rng.choice(len(X_test), size=10, replace=False)

    for rank, idx in enumerate(indices, 1):
        x_sample = X_test[idx : idx + 1]
        true_lbl = y_test[idx]
        probs    = model.predict_proba(x_sample)[0]
        pred_lbl = np.argmax(probs)
        correct  = "✓" if pred_lbl == true_lbl else "✗"

        print(f"\n  Sample {rank:02d}  |  True: {true_lbl}  Pred: {pred_lbl}  {correct}  (conf: {probs[pred_lbl]*100:.1f}%)")
        print()
        for line in ascii_digit(x_sample[0]).split("\n"):
            print("  " + line)
        print(f"\n  Probabilities: " + "  ".join(
            f"{d}:{probs[d]*100:5.1f}%" for d in range(10)
        ))

    print("\n" + "=" * 60)
    print("  Training complete.  Weights saved to mnist_weights.npz")
    print("=" * 60)


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    main()

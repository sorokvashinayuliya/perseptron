import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

train_file    = "train.csv"
model_file    = "model.pkl"
target_col    = "diagnosis"
random_state  = 42

HIDDEN_SIZES  = [16, 8]
LEARNING_RATE = 0.001   
MAX_EPOCHS    = 500
BATCH_SIZE    = 32
GRAD_CLIP     = 5.0     
ё
def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

class MLP:
    def __init__(self, layer_sizes: list, random_state: int = 42):
        np.random.seed(random_state)
        self.layer_sizes = layer_sizes
        self.W = []
        self.b = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            self.W.append(np.random.randn(fan_in, layer_sizes[i+1]) * np.sqrt(2.0 / fan_in))
            self.b.append(np.zeros((1, layer_sizes[i+1])))

    def _forward(self, X):
        self._zs = []
        self._as = [X]
        a = X
        for i in range(len(self.W)):
            z = a @ self.W[i] + self.b[i]
            self._zs.append(z)
            a = relu(z) if i < len(self.W) - 1 else softmax(z)
            self._as.append(a)
        return self._as[-1]

    def cross_entropy(y_prob, y_true):
        n = len(y_true)
        return -np.log(y_prob[np.arange(n), y_true] + 1e-9).mean()

    def _backward(self, y_true, lr):
        n = len(y_true)
        delta = self._as[-1].copy()
        delta[np.arange(n), y_true] -= 1
        delta /= n

        for i in reversed(range(len(self.W))):
            dW = self._as[i].T @ delta
            db = delta.sum(axis=0, keepdims=True)

            dW = np.clip(dW, -GRAD_CLIP, GRAD_CLIP)
            db = np.clip(db, -GRAD_CLIP, GRAD_CLIP)

            if i > 0:
                delta = delta @ self.W[i].T * relu_deriv(self._zs[i - 1])
                delta = np.clip(delta, -GRAD_CLIP, GRAD_CLIP)

            self.W[i] -= lr * dW
            self.b[i] -= lr * db

    def fit(self, X, y, lr, epochs, batch_size, verbose=True):
        n = len(X)
        loss_history = []
        acc_history  = []

        for epoch in range(1, epochs + 1):
            idx  = np.random.permutation(n)
            X, y = X[idx], y[idx]

            for start in range(0, n, batch_size):
                Xb = X[start:start + batch_size]
                yb = y[start:start + batch_size]
                self._forward(Xb)
                self._backward(yb, lr)

            prob = self._forward(X)

            if np.isnan(prob).any():
                print(f"  NaN на эпохе {epoch}, обучение остановлено.")
                break

            loss = MLP.cross_entropy(prob, y)
            acc  = np.mean(np.argmax(prob, axis=1) == y)
            loss_history.append(loss)
            acc_history.append(acc)

            if verbose and (epoch % 50 == 0 or epoch == 1):
                print(f"  Эпоха {epoch:4d} | Loss: {loss:.4f} | Accuracy: {acc:.4f}")

        return loss_history, acc_history

    def predict_proba(self, X):
        return self._forward(X)

    def predict(self, X):
        return np.argmax(self._forward(X), axis=1)

def train_model(train_file: str, model_file: str, target_col: str):
    df = pd.read_csv(train_file)
    X  = df.drop(columns=[target_col]).values.astype(float)
    y  = df[target_col].values.astype(int)

    mean = X.mean(axis=0)
    std  = X.std(axis=0);  std[std == 0] = 1
    X_norm = (X - mean) / std

    layer_sizes = [X_norm.shape[1]] + HIDDEN_SIZES + [2]
    print(f"Архитектура сети: {layer_sizes}")
    print(f"lr={LEARNING_RATE}  epochs={MAX_EPOCHS}  batch={BATCH_SIZE}\n")

    model = MLP(layer_sizes, random_state=random_state)
    loss_hist, acc_hist = model.fit(
        X_norm, y,
        lr=LEARNING_RATE,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=True,
    )

    model_data = {
        "weights":      model.W,
        "biases":       model.b,
        "layer_sizes":  layer_sizes,
        "mean":         mean,
        "std":          std,
        "target_col":   target_col,
        "random_state": random_state,
    }
    with open(model_file, "wb") as f:
        pickle.dump(model_data, f)

    print(f"\nИтог: Loss={loss_hist[-1]:.4f}  Accuracy={acc_hist[-1]:.4f} ({acc_hist[-1]*100:.1f}%)")
    print(f"Модель сохранена в {model_file}")
 
    epochs_range = range(1, len(loss_hist) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs_range, loss_hist, color="steelblue", linewidth=1.5)
    ax1.set_title("Кривая обучения — Loss")
    ax1.set_xlabel("Эпоха")
    ax1.set_ylabel("Binary Cross-Entropy")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, acc_hist, color="darkorange", linewidth=1.5)
    ax2.set_title("Кривая обучения — Accuracy")
    ax2.set_xlabel("Эпоха")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("learning_curves.png", dpi=150)
    plt.show()
    print("Графики сохранены в learning_curves.png")


if __name__ == "__main__":
    train_model(train_file, model_file, target_col)
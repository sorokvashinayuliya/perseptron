import pandas as pd
import numpy as np
import pickle

test_file   = "test.csv"
model_file  = "model.pkl"
output_file = "prediction.csv"


def relu(z):
    return np.maximum(0, z)

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

    def predict_proba(self, X):
        return self._forward(X)

    def predict(self, X):
        return np.argmax(self._forward(X), axis=1)


def binary_cross_entropy(y_prob, y_true):
    n = len(y_true)
    return -np.log(y_prob[np.arange(n), y_true] + 1e-9).mean()

def predict_data(test_file: str, model_file: str, output_file: str = None):

    with open(model_file, "rb") as f:
        model_data = pickle.load(f)

    layer_sizes = model_data["layer_sizes"]
    model       = MLP(layer_sizes, random_state=model_data["random_state"])
    model.W     = model_data["weights"]
    model.b     = model_data["biases"]

    mean       = model_data["mean"]
    std        = model_data["std"]
    target_col = model_data["target_col"]

    df     = pd.read_csv(test_file)
    X      = df.drop(columns=[target_col]).values.astype(float)
    y_true = df[target_col].values.astype(int)

    X_normalized = (X - mean) / std

    y_prob = model.predict_proba(X_normalized)   
    y_pred = model.predict(X_normalized)         

    bce = binary_cross_entropy(y_prob, y_true)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy  = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)

    print(f"Тестовая выборка : {len(X)} примеров")
    print(f"Архитектура сети : {layer_sizes}\n")

    print("Вероятности (первые 5 примеров):")
    print(f"  {'P(B=0)':>8}  {'P(M=1)':>8}  {'Прогноз':>7}  {'Истина':>6}")
    for i in range(min(5, len(y_true))):
        print(f"  {y_prob[i,0]:>8.4f}  {y_prob[i,1]:>8.4f}  "
              f"{'M' if y_pred[i]==1 else 'B':>7}  "
              f"{'M' if y_true[i]==1 else 'B':>6}")

    print(f"\nБинарная кросс-энтропия : {bce:.4f}")

    print("\nМатрица ошибок:")
    print(f"                   Прогноз M   Прогноз B")
    print(f"  Истинный M (1):    {tp:6d}      {fn:6d}")
    print(f"  Истинный B (0):    {fp:6d}      {tn:6d}")

    print(f"\nМетрики:")
    print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.1f}%)")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-score  : {f1:.4f}")

    if output_file:
        result_df = df.copy()
        result_df["prob_B"]    = y_prob[:, 0]
        result_df["prob_M"]    = y_prob[:, 1]
        result_df["predicted"] = y_pred
        result_df["correct"]   = (y_pred == y_true).astype(int)
        result_df.to_csv(output_file, index=False)
        print(f"\nПредсказания сохранены в {output_file}")


if __name__ == "__main__":
    predict_data(test_file, model_file, output_file)
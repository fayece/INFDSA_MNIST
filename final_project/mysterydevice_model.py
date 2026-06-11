import numpy as np
from collections import deque

epsilon = 1e-15


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(np.float32)


def softmax(z):
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def one_hot(y, n_classes):
    return np.eye(n_classes)[y]


def sparse_cross_entropy(y_true, y_pred):
    probs = np.clip(y_pred, epsilon, 1.0 - epsilon)
    log_probs = np.log(probs)
    true_log_probs = log_probs[np.arange(len(y_true)), y_true]
    return -np.mean(true_log_probs)



class NeuralNetworkInference:
    def __init__(self, path=None):
        self.weights = []
        self.biases = []
        self.w_min = []
        self.w_max = []
        if path:
            self.load_weights(path)

    def _dequantize(self, i):
        return self.weights[i].astype(np.float32) / 255.0 * (self.w_max[i] - self.w_min[i]) + self.w_min[i]

    def _matmul(self, x, i):
        if not self.w_min:
            return x @ self.weights[i]

        W, w_min, w_max = self.weights[i], self.w_min[i], self.w_max[i]
        out = np.zeros((x.shape[0], W.shape[1]), dtype=np.float32)
        for j, col in enumerate(W.T):
            out[:, j] = x @ (col.astype(np.float32) / 255.0 * (w_max - w_min) + w_min)
        return out

    def forward(self, x):
        current = x
        for i in range(len(self.weights) - 1):
            preac = self._matmul(current, i) + self.biases[i]
            current = relu(preac)

        o_preac = self._matmul(current, -1) + self.biases[-1]
        return softmax(o_preac)

    def predict(self, x):
        x = np.array(x, dtype=np.float32).reshape(1, -1)
        pred = self.forward(x)
        probabilities = pred[0]
        return int(np.argmax(probabilities)), probabilities

    def load_weights(self, path):
        data = np.load(path)
        self.weights = []
        self.biases = []
        self.w_min = []
        self.w_max = []
        i = 0
        while f'w{i}' in data:
            self.weights.append(data[f'w{i}'])
            self.w_min.append(data[f'w{i}_min'][0])
            self.w_max.append(data[f'w{i}_max'][0])
            self.biases.append(data[f'b{i}'])
            i += 1


class NeuralNetworkTraining(NeuralNetworkInference):
    def __init__(self, input_nodes, hidden_layer_sizes, output_nodes, dropout_rates=None):
        super().__init__()

        if dropout_rates is None:
            dropout_rates = [0.0] * len(hidden_layer_sizes)
        elif isinstance(dropout_rates, float):
            dropout_rates = [dropout_rates] * len(hidden_layer_sizes)
        self.dropout_rates = dropout_rates

        layer_sizes = [input_nodes] + hidden_layer_sizes + [output_nodes]
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

        self._best_val_loss = np.inf
        self._epochs_without_improvement = 0
        self._best_snapshot = self._snapshot()

    def _get_weight(self, i):
        if self.w_min:
            return self._dequantize(i)
        return self.weights[i]

    def forward_train(self, x):
        hidden_caches = []
        current = x

        for i in range(len(self.weights) - 1):
            preac = self._matmul(current, i) + self.biases[i]
            ac = relu(preac)

            mask = None
            rate = self.dropout_rates[i]
            if rate > 0:
                mask = (np.random.rand(*ac.shape) > rate) / (1 - rate)
                ac = ac * mask
            hidden_caches.append((preac, ac, mask))
            current = ac

        o_preac = self._matmul(current, -1) + self.biases[-1]
        pred = softmax(o_preac)
        return x, hidden_caches, pred

    def backward(self, y_true, cache):
        x, hidden_caches, pred = cache
        n_hidden = len(hidden_caches)

        weight_gradients = [None] * len(self.weights)
        biases_gradients = [None] * len(self.biases)

        output_gradient = pred - one_hot(y_true, pred.shape[1])
        last_ac = hidden_caches[-1][1]
        weight_gradients[-1] = last_ac.T @ output_gradient
        biases_gradients[-1] = np.sum(output_gradient, axis=0, keepdims=True)

        gradient = output_gradient
        for i in reversed(range(n_hidden)):
            preac, ac, mask = hidden_caches[i]

            gradient = gradient @ self._get_weight(i + 1).T
            if mask is not None:
                gradient = gradient * mask
            gradient = gradient * relu_derivative(preac)

            prev_ac = hidden_caches[i - 1][1] if i > 0 else x
            weight_gradients[i] = prev_ac.T @ gradient
            biases_gradients[i] = np.sum(gradient, axis=0, keepdims=True)

        return weight_gradients, biases_gradients

    def _apply_gradients(self, weight_grads, bias_grads, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * weight_grads[i]
            self.biases[i] -= learning_rate * bias_grads[i]

    def _snapshot(self):
        return [weight.copy() for weight in self.weights], [bias.copy() for bias in self.biases]

    def _restore(self, snapshot):
        weights, biases = snapshot
        self.weights = [w.copy() for w in weights]
        self.biases = [b.copy() for b in biases]

    def train(self, x_train, y_train, learning_rate=0.001, epochs=50, batch_size=32, val_split=0.1, patience=12,
              verbose=True):
        n_val = int(len(x_train) * val_split)
        x_val, y_val = x_train[:n_val], y_train[:n_val]
        x_tr, y_tr = x_train[n_val:], y_train[n_val:]
        n = len(x_tr)

        best_info = None
        stopped_early = False

        for epoch in range(1, epochs + 1):
            indices = np.random.permutation(n)
            x_shuffled = x_tr[indices]
            y_shuffled = y_tr[indices]

            batch_queue = deque(
                (x_shuffled[start:start + batch_size], y_shuffled[start:start + batch_size]) for start in
                range(0, n, batch_size))
            epoch_loss, epoch_correct = 0.0, 0

            while batch_queue:
                x_batch, y_batch = batch_queue.popleft()
                cache = self.forward_train(x_batch)
                _, _, pred = cache

                epoch_loss += sparse_cross_entropy(y_batch, pred) * len(x_batch)
                epoch_correct += np.sum(np.argmax(pred, axis=1) == y_batch)

                weight_grads, bias_grads = self.backward(y_batch, cache)
                self._apply_gradients(weight_grads, bias_grads, learning_rate)

            val_pred = self.forward(x_val)
            val_loss = sparse_cross_entropy(y_val, val_pred)

            if verbose:
                print(
                    f"Epoch {epoch:3d} | Acc: {epoch_correct / n:.4f} | Loss: {epoch_loss / n:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self._epochs_without_improvement = 0
                self._best_snapshot = self._snapshot()
                best_info = (epoch, epoch_correct / n, epoch_loss / n, val_loss)
            else:
                self._epochs_without_improvement += 1

            if self._epochs_without_improvement >= patience:
                stopped_early = True
                break

        self._restore(self._best_snapshot)
        if verbose and best_info:
            if stopped_early: print(f"\nEarly stopping na epoch {epoch}.")
            print(
                f"Hersteld naar epoch {best_info[0]} | Acc: {best_info[1]:.4f} | Loss: {best_info[2]:.4f} | Val Loss: {best_info[3]:.4f}")

    def save_weights(self, path):
        arrays = {}
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            w_min, w_max = weight.min(), weight.max()
            w_quantized = ((weight - w_min) / (w_max - w_min) * 255).astype(np.uint8)
            arrays[f"w{i}"] = w_quantized
            arrays[f"w{i}_min"] = np.array([w_min], dtype=np.float32)
            arrays[f"w{i}_max"] = np.array([w_max], dtype=np.float32)
            arrays[f"b{i}"] = bias.astype(np.float32)
        np.savez_compressed(path, **arrays)

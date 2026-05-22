import numpy as np
from abc import ABC, abstractmethod

epsilon = 1e-15


def one_hot(y, n_classes):
    return np.eye(n_classes)[y]


def relu(z):
    return np.maximum(0, z)


def softmax(z):
    z_shifted = z - np.max(z, axis=1, keepdims=True)  # prevent overflow
    exp_z = np.exp(z_shifted)
    sum_exp_z = np.sum(exp_z, axis=1, keepdims=True)
    softmax_z = exp_z / sum_exp_z
    return softmax_z


# In a real-world application, cross_entropy and sparse_cross_entropy would be merged together, auto-detecting which to use based off the shape of y_true.
# y_true expects one-hot encoded values
def cross_entropy(y_true, y_pred):
    probs = np.clip(y_pred, epsilon, 1.0 - epsilon)
    log_probs = np.log(probs)
    cross_entropy_per_sample = np.sum(y_true * log_probs, axis=1)  # filter out all incorrect classes per row
    loss = -np.mean(cross_entropy_per_sample)

    return loss


# y_true expects class labels (int)
def sparse_cross_entropy(y_true, y_pred):
    probs = np.clip(y_pred, epsilon, 1.0 - epsilon)
    log_probs = np.log(probs)
    true_log_probs = log_probs[np.arange(len(y_true)), y_true]  # pick the log prob of the correct class per sample
    loss = -np.mean(true_log_probs)

    return loss


def forward(x, w1, w2, b1, b2):
    hidden_preactivation = x @ w1 + b1
    hidden_activation = relu(hidden_preactivation)

    output_preactivation = hidden_activation @ w2 + b2
    output_activation = softmax(output_preactivation)

    return x, hidden_preactivation, hidden_activation, output_preactivation, output_activation


def compute_output_gradient(final_prediction, correct_answers):
    return final_prediction - one_hot(correct_answers, final_prediction.shape[1])


def compute_output_gradients(hidden_output, output_gradient):
    gradients = hidden_output.T @ output_gradient
    biases = np.sum(output_gradient, axis=0, keepdims=True)

    return gradients, biases


def compute_hidden_gradient(output_gradient, hidden_to_output_weights):
    return output_gradient @ hidden_to_output_weights.T


def relu_derivative(x):
    return (x > 0).astype(float)


def compute_hidden_gradients(hidden_gradient, hidden_raw_gradient, input_data):
    hidden_activation_derivative = relu_derivative(hidden_raw_gradient)
    hidden_gradient_after_relu = hidden_gradient * hidden_activation_derivative

    weight_gradient = input_data.T @ hidden_gradient_after_relu
    bias_gradient = np.sum(hidden_gradient_after_relu, axis=0, keepdims=True)

    return weight_gradient, bias_gradient


def backward(y_true, cache, w2):
    x, hidden_preactivation, hidden_activation, output_preactivation, final_prediction = cache

    output_gradient = compute_output_gradient(final_prediction, y_true)
    o_weight, o_bias = compute_output_gradients(hidden_activation, output_gradient)

    hidden_gradient = compute_hidden_gradient(output_gradient, w2)
    h_weight, h_bias = compute_hidden_gradients(hidden_gradient, hidden_preactivation, x)

    return o_weight, o_bias, h_weight, h_bias


def train_raw(x_train, y_train, w1, w2, b1, b2, epochs, learning_rate, verbose=True):
    for epoch in range(1, epochs + 1):
        cache = forward(x_train, w1, w2, b1, b2)
        x, hidden_preactivation, hidden_activation, output_preactivation, pred = cache

        loss = sparse_cross_entropy(y_train, pred)

        dw2, db2, dw1, db1 = backward(y_train, cache, w2)

        w1 -= learning_rate * dw1
        w2 -= learning_rate * dw2
        b1 -= learning_rate * db1
        b2 -= learning_rate * db2

        if verbose:
            preds = np.argmax(pred, axis=1)
            acc = np.mean(preds == y_train)
            print(f"Epoch {epoch}, Accuracy: {acc:.4f},  Loss: {loss:.4f}")

    return w1, w2, b1, b2


def train(x_train, y_train, w1, w2, b1, b2, epochs, learning_rate, batch_size=64, verbose=True):
    n = len(x_train)
    for epoch in range(1, epochs + 1):
        indices = np.random.permutation(n)
        x_shuffled = x_train[indices]
        y_shuffled = y_train[indices]

        epoch_loss = 0.0
        epoch_correct = 0

        for start in range(0, n, batch_size):
            x_batch = x_shuffled[start:start + batch_size]
            y_batch = y_shuffled[start:start + batch_size]

            cache = forward(x_batch, w1, w2, b1, b2)
            _, _, _, _, pred = cache

            epoch_loss += sparse_cross_entropy(y_batch, pred) * len(x_batch)
            epoch_correct += np.sum(np.argmax(pred, axis=1) == y_batch)

            dw2, db2, dw1, db1 = backward(y_batch, cache, w2)

            w1 -= learning_rate * dw1
            w2 -= learning_rate * dw2
            b1 -= learning_rate * db1
            b2 -= learning_rate * db2

        if verbose:
            print(f"Epoch {epoch}, Accuracy: {epoch_correct / n:.4f}, Loss: {epoch_loss / n:.4f}")

    return w1, w2, b1, b2


def evaluate(x_test, y_test, w1, w2, b1, b2):
    cache = forward(x_test, w1, w2, b1, b2)
    _, _, _, _, pred = cache

    loss = sparse_cross_entropy(y_test, pred)
    predicted_digits = np.argmax(pred, axis=1)
    acc = np.mean(predicted_digits == y_test)

    print(f"Test Accuracy: {acc:.4f}, Test Loss: {loss:.4f}")

    return acc, loss, predicted_digits


class BaseNeuralNetwork(ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, y_true, cache):
        pass

    @abstractmethod
    def train(self, x_train, y_train, epochs, learning_rate, batch_size, verbose):
        pass

    @abstractmethod
    def evaluate(self, x_test, y_test):
        pass


class NeuralNetwork(BaseNeuralNetwork):
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.w1 = np.random.randn(input_nodes, hidden_nodes) * np.sqrt(2 / input_nodes)
        self.w2 = np.random.randn(hidden_nodes, output_nodes) * np.sqrt(2 / hidden_nodes)
        self.b1 = np.zeros((1, hidden_nodes))
        self.b2 = np.zeros((1, output_nodes))

    def forward(self, x):
        return forward(x, self.w1, self.w2, self.b1, self.b2)

    def backward(self, y_true, cache):
        return backward(y_true, cache, self.w2)

    def _apply_gradients(self, dw1, db1, dw2, db2, learning_rate):
        self.w1 -= learning_rate * dw1
        self.w2 -= learning_rate * dw2
        self.b1 -= learning_rate * db1
        self.b2 -= learning_rate * db2

    def train_raw(self, x_train, y_train, epochs, learning_rate, verbose=True):
        for epoch in range(1, epochs + 1):
            cache = self.forward(x_train)
            _, _, _, _, pred = cache

            loss = sparse_cross_entropy(y_train, pred)
            dw2, db2, dw1, db1 = self.backward(y_train, cache)
            self._apply_gradients(dw1, db1, dw2, db2, learning_rate)

            if verbose:
                acc = np.mean(np.argmax(pred, axis=1) == y_train)
                print(f"Epoch {epoch}, Accuracy: {acc:.4f}, Loss: {loss:.4f}")

    def train(self, x_train, y_train, epochs, learning_rate, batch_size=64, verbose=True):
        n = len(x_train)
        for epoch in range(1, epochs + 1):
            indices = np.random.permutation(n)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0.0
            epoch_correct = 0

            for start in range(0, n, batch_size):
                x_batch = x_shuffled[start:start + batch_size]
                y_batch = y_shuffled[start:start + batch_size]

                cache = self.forward(x_batch)
                _, _, _, _, pred = cache

                epoch_loss += sparse_cross_entropy(y_batch, pred) * len(x_batch)
                epoch_correct += np.sum(np.argmax(pred, axis=1) == y_batch)

                dw2, db2, dw1, db1 = self.backward(y_batch, cache)
                self._apply_gradients(dw1, db1, dw2, db2, learning_rate)

            if verbose:
                print(f"Epoch {epoch}, Accuracy: {epoch_correct / n:.4f}, Loss: {epoch_loss / n:.4f}")

    def evaluate(self, x_test, y_test):
        return evaluate(x_test, y_test, self.w1, self.w2, self.b1, self.b2)


class ImprovedNeuralNetwork(BaseNeuralNetwork):
    def __init__(self, input_nodes, hidden_layer_sizes, output_nodes, dropout_rates=None):
        if dropout_rates is None:
            dropout_rates = [0.0] * len(hidden_layer_sizes)
        elif isinstance(dropout_rates, float):
            dropout_rates = [dropout_rates] * len(hidden_layer_sizes)

        self.dropout_rates = dropout_rates

        layer_sizes = [input_nodes] + hidden_layer_sizes + [output_nodes]
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

        self.bn_layers = [BatchNormLayer(size) for size in hidden_layer_sizes]

        self._best_val_loss = np.inf
        self._epochs_without_improvement = 0
        self._best_snapshot = self._snapshot_weights()

    def forward(self, x, training=True):
        hidden_caches = []
        current = x

        for i, bn in enumerate(self.bn_layers):
            preac = current @ self.weights[i] + self.biases[i]
            batch_out = bn.forward(preac, train=training)
            ac = relu(batch_out)

            mask = None
            rate = self.dropout_rates[i]
            if training and rate > 0:
                mask = (np.random.rand(*ac.shape) > rate) / (1 - rate)
                ac = ac * mask

            hidden_caches.append((preac, batch_out, ac, mask))
            current = ac

        o_preac = current @ self.weights[-1] + self.biases[-1]
        pred = softmax(o_preac)

        return x, hidden_caches, pred

    def backward(self, y_true, cache):
        x, hidden_caches, pred = cache

        output_gradient = pred - one_hot(y_true, pred.shape[1])

        last_ac = hidden_caches[-1][2]
        dw_last = last_ac.T @ output_gradient
        db_last = np.sum(output_gradient, axis=0, keepdims=True)

        weight_gradients = [None] * len(self.weights)
        biases_gradients = [None] * len(self.biases)
        weight_gradients[-1] = dw_last
        biases_gradients[-1] = db_last

        gradient = output_gradient

        for i in reversed(range(len(self.bn_layers))):
            preac, batch_out, batch_ac, mask = hidden_caches[i]

            gradient = gradient @ self.weights[i + 1].T
            if mask is not None:
                gradient = gradient * mask
            gradient = gradient * relu_derivative(batch_out)
            gradient = self.bn_layers[i].backward(gradient)

            prev_ac = hidden_caches[i - 1][2] if i > 0 else x
            weight_gradients[i] = prev_ac.T @ gradient
            biases_gradients[i] = np.sum(gradient, axis=0, keepdims=True)

        return weight_gradients, biases_gradients

    def _apply_gradients(self, weight_grads, bias_grads, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * weight_grads[i]
            self.biases[i] -= learning_rate * bias_grads[i]
        for bn in self.bn_layers:
            bn.apply_gradients(learning_rate)

    def _snapshot_weights(self, epoch=None, accuracy=None, loss=None, val_loss=None):
        weights = [w.copy() for w in self.weights]
        biases = [b.copy() for b in self.biases]
        bn_state = [
            {
                "gamma": bn.gamma.copy(),
                "bias": bn.bias.copy(),
                "running_mean_x": bn.running_mean_x.copy(),
                "running_var_x": bn.running_var_x.copy(),
            }
            for bn in self.bn_layers
        ]
        meta = {"epoch": epoch, "accuracy": accuracy, "loss": loss, "val_loss": val_loss}
        return weights, biases, bn_state, meta

    def _restore_weights(self, snapshot):
        weights, biases, bn_state, meta = snapshot
        self.weights = [w.copy() for w in weights]
        self.biases = [b.copy() for b in biases]
        for bn, state in zip(self.bn_layers, bn_state):
            bn.gamma = state["gamma"].copy()
            bn.bias = state["bias"].copy()
            bn.running_mean_x = state["running_mean_x"].copy()
            bn.running_var_x = state["running_var_x"].copy()
        return meta

    def train(self, x_train, y_train, epochs, learning_rate, batch_size=64,
              val_split=0.1, patience=12, verbose=True):
        n_val = int(len(x_train) * val_split)
        x_val, y_val = x_train[:n_val], y_train[:n_val]
        x_train, y_train = x_train[n_val:], y_train[n_val:]

        stopped_early = False

        for epoch in range(1, epochs + 1):
            epoch_loss, epoch_correct, n = self.run_epoch(x_train, y_train, learning_rate, batch_size)

            _, _, val_pred = self.forward(x_val, training=False)
            val_loss = sparse_cross_entropy(y_val, val_pred)

            if verbose:
                print(f"Epoch {epoch}, Accuracy: {epoch_correct / n:.4f}, "
                      f"Loss: {epoch_loss / n:.4f}, Val Loss: {val_loss:.4f}")

            if self.find_best_epoch(val_loss, epoch, epoch_correct, n, epoch_loss, patience):
                stopped_early = True
                break

        meta = self._restore_weights(self._best_snapshot)
        if verbose:
            if stopped_early:
                print(f"Early stopping at epoch {epoch}. No improvement for {patience} epochs.")
            print(f"Reverting to epoch {meta['epoch']}.\n"
                  f" - Accuracy: {meta['accuracy']:.4f}\n"
                  f" - Loss: {meta['loss']:.4f}\n"
                  f" - Val Loss: {meta['val_loss']:.4f}")


    def run_epoch(self, x_train, y_train, learning_rate, batch_size):
        x_shuffled, y_shuffled, n = self.init_epoch(x_train, y_train)

        epoch_loss = 0.0
        epoch_correct = 0

        for start in range(0, n, batch_size):
            batch_loss, batch_correct = self.train_epoch(x_shuffled, y_shuffled, learning_rate, batch_size, start)
            epoch_loss += batch_loss
            epoch_correct += batch_correct

        return epoch_loss, epoch_correct, n

    def find_best_epoch(self, val_loss, epoch, epoch_correct, n, epoch_loss, patience):
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self._epochs_without_improvement = 0
            self._best_snapshot = self._snapshot_weights(
                epoch=epoch,
                accuracy=epoch_correct / n,
                loss=epoch_loss / n,
                val_loss=val_loss
            )
        else:
            self._epochs_without_improvement += 1

        return self._epochs_without_improvement >= patience

    def init_epoch(self, x_train, y_train):
        n = len(x_train)
        indices = np.random.permutation(n)
        x_shuffled = x_train[indices]
        y_shuffled = y_train[indices]
        return x_shuffled, y_shuffled, n

    def train_epoch(self, x_shuffled, y_shuffled, learning_rate, batch_size, start):
        x_batch = x_shuffled[start:start + batch_size]
        y_batch = y_shuffled[start:start + batch_size]

        cache = self.forward(x_batch, training=True)
        _, _, pred = cache

        batch_loss = sparse_cross_entropy(y_batch, pred) * len(x_batch)
        batch_correct = np.sum(np.argmax(pred, axis=1) == y_batch)

        weight_gradients, biases_gradients = self.backward(y_batch, cache)
        self._apply_gradients(weight_gradients, biases_gradients, learning_rate)

        return batch_loss, batch_correct

    def evaluate(self, x_test, y_test):
        _, _, pred = self.forward(x_test, training=False)

        loss = sparse_cross_entropy(y_test, pred)
        predicted_digits = np.argmax(pred, axis=1)
        acc = np.mean(predicted_digits == y_test)

        print(f"Test Accuracy: {acc:.4f}, Test Loss: {loss:.4f}")
        return acc, loss, predicted_digits


# BatchNormLayer class created by Renan Cunha on GitHub
# https://github.com/renan-cunha/BatchNormalization/blob/master/src/feed_forward/layers.py
class BatchNormLayer:

    def __init__(self, dims: int) -> None:
        self.gamma = np.ones((1, dims), dtype="float32")
        self.bias = np.zeros((1, dims), dtype="float32")

        self.running_mean_x = np.zeros(0)
        self.running_var_x = np.zeros(0)

        # forward params
        self.var_x = np.zeros(0)
        self.stddev_x = np.zeros(0)
        self.x_minus_mean = np.zeros(0)
        self.standard_x = np.zeros(0)
        self.num_examples = 0
        self.mean_x = np.zeros(0)
        self.running_avg_gamma = 0.9

        # backward params
        self.gamma_grad = np.zeros(0)
        self.bias_grad = np.zeros(0)

    def update_running_variables(self) -> None:
        is_mean_empty = np.array_equal(np.zeros(0), self.running_mean_x)
        is_var_empty = np.array_equal(np.zeros(0), self.running_var_x)
        if is_mean_empty != is_var_empty:
            raise ValueError("Mean and Var running averages should be "
                             "initilizaded at the same time")
        if is_mean_empty:
            self.running_mean_x = self.mean_x
            self.running_var_x = self.var_x
        else:
            gamma = self.running_avg_gamma
            self.running_mean_x = gamma * self.running_mean_x + \
                                  (1.0 - gamma) * self.mean_x
            self.running_var_x = gamma * self.running_var_x + \
                                 (1. - gamma) * self.var_x

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        self.num_examples = x.shape[0]
        if train:
            self.mean_x = np.mean(x, axis=0, keepdims=True)
            self.var_x = np.mean((x - self.mean_x) ** 2, axis=0, keepdims=True)
            self.update_running_variables()
        else:
            self.mean_x = self.running_mean_x.copy()
            self.var_x = self.running_var_x.copy()

        self.var_x += epsilon
        self.stddev_x = np.sqrt(self.var_x)
        self.x_minus_mean = x - self.mean_x
        self.standard_x = self.x_minus_mean / self.stddev_x
        return self.gamma * self.standard_x + self.bias

    def backward(self, grad_input: np.ndarray) -> np.ndarray:
        standard_grad = grad_input * self.gamma

        var_grad = np.sum(standard_grad * self.x_minus_mean * -0.5 * self.var_x ** (-3/2),
                          axis=0, keepdims=True)
        stddev_inv = 1 / self.stddev_x
        aux_x_minus_mean = 2 * self.x_minus_mean / self.num_examples

        mean_grad = (np.sum(standard_grad * -stddev_inv, axis=0,
                            keepdims=True) +
                            var_grad * np.sum(-aux_x_minus_mean, axis=0,
                            keepdims=True))

        self.gamma_grad = np.sum(grad_input * self.standard_x, axis=0,
                                 keepdims=True)
        self.bias_grad = np.sum(grad_input, axis=0, keepdims=True)

        return standard_grad * stddev_inv + var_grad * aux_x_minus_mean + \
               mean_grad / self.num_examples

    def apply_gradients(self, learning_rate: float) -> None:
        self.gamma -= learning_rate * self.gamma_grad
        self.bias -= learning_rate * self.bias_grad

import numpy as np
from sklearn.metrics import confusion_matrix


class NeuralNetwork:
    def __init__(self, layer_sizes, activation="relu", output_activation="softmax", loss="cross_entropy",
                 optimizer="gd", lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        optimizer: "gd" (batch gradient descent), "sgd" (stochastic), "adam"
        """
        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self.output_activation_name = output_activation
        self.loss_name = loss
        self.optimizer = optimizer
        self.lr = lr
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.params = self._init_weights()

        # For Adam optimizer
        self.v, self.s, self.t = {}, {}, 0
        if optimizer == "adam":
            for key in self.params:
                self.v[key] = np.zeros_like(self.params[key])
                self.s[key] = np.zeros_like(self.params[key])

    def _init_weights(self):
        np.random.seed(42)  # reproducibility
        params = {}
        for i in range(1, len(self.layer_sizes)):
            params[f"W{i}"] = np.random.randn(self.layer_sizes[i], self.layer_sizes[i-1]) * np.sqrt(2. / self.layer_sizes[i-1])
            params[f"b{i}"] = np.zeros((self.layer_sizes[i], 1))
        return params

    # ---------------- Activation Functions ----------------
    def _activation(self, Z, func):
        if func == "relu":
            return np.maximum(0, Z)
        elif func == "sigmoid":
            return 1 / (1 + np.exp(-Z))
        elif func == "tanh":
            return np.tanh(Z)
        elif func == "softmax":
            expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            return expZ / expZ.sum(axis=0, keepdims=True)
        else:
            raise ValueError("Unknown activation")

    def _activation_derivative(self, A, func):
        if func == "relu":
            return (A > 0).astype(float)
        elif func == "sigmoid":
            return A * (1 - A)
        elif func == "tanh":
            return 1 - A**2
        else:
            raise ValueError("No derivative defined for softmax here")

    # ---------------- Forward Pass ----------------
    def forward(self, X):
        cache = {"A0": X}
        L = len(self.layer_sizes) - 1

        for i in range(1, L):
            W, b = self.params[f"W{i}"], self.params[f"b{i}"]
            Z = W @ cache[f"A{i-1}"] + b
            A = self._activation(Z, self.activation_name)
            cache[f"Z{i}"], cache[f"A{i}"] = Z, A

        # Output layer
        W, b = self.params[f"W{L}"], self.params[f"b{L}"]
        Z = W @ cache[f"A{L-1}"] + b
        A = self._activation(Z, self.output_activation_name)
        cache[f"Z{L}"], cache[f"A{L}"] = Z, A
        return cache

    # ---------------- Loss Functions ----------------
    def _compute_loss(self, Y_pred, Y_true):
        m = Y_true.shape[1]
        if self.loss_name == "cross_entropy":
            eps = 1e-9
            return -np.sum(Y_true * np.log(Y_pred + eps)) / m
        elif self.loss_name == "mse":
            return np.mean((Y_true - Y_pred) ** 2)
        else:
            raise ValueError("Unknown loss function")

    # ---------------- Backpropagation ----------------
    def backward(self, cache, Y_true):
        grads = {}
        m = Y_true.shape[1]
        L = len(self.layer_sizes) - 1
        A_final = cache[f"A{L}"]

        # Output layer gradient
        if self.loss_name == "cross_entropy" and self.output_activation_name == "softmax":
            dZ = A_final - Y_true
        else:  # generic
            dA = -(Y_true - A_final)
            dZ = dA * self._activation_derivative(A_final, self.output_activation_name)

        grads[f"dW{L}"] = (1/m) * dZ @ cache[f"A{L-1}"].T
        grads[f"db{L}"] = (1/m) * np.sum(dZ, axis=1, keepdims=True)

        dA_prev = self.params[f"W{L}"].T @ dZ

        # Hidden layers
        for i in reversed(range(1, L)):
            A = cache[f"A{i}"]
            dZ = dA_prev * self._activation_derivative(A, self.activation_name)
            grads[f"dW{i}"] = (1/m) * dZ @ cache[f"A{i-1}"].T
            grads[f"db{i}"] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            dA_prev = self.params[f"W{i}"].T @ dZ

        return grads

    # ---------------- Update ----------------
    def update(self, grads):
        if self.optimizer == "gd":
            # vanilla gradient descent
            for key in self.params:
                if key.startswith("W") or key.startswith("b"):
                    self.params[key] -= self.lr * grads["d" + key]

        elif self.optimizer == "sgd":
            # stochastic (just like gd but we use mini-batches in fit)
            for key in self.params:
                if key.startswith("W") or key.startswith("b"):
                    self.params[key] -= self.lr * grads["d" + key]

        elif self.optimizer == "adam":
            self.t += 1
            for key in self.params:
                if key.startswith("W") or key.startswith("b"):
                    # momentum
                    self.v[key] = self.beta1 * self.v[key] + (1 - self.beta1) * grads["d" + key]
                    # rmsprop
                    self.s[key] = self.beta2 * self.s[key] + (1 - self.beta2) * (grads["d" + key] ** 2)

                    # bias correction
                    v_corr = self.v[key] / (1 - self.beta1 ** self.t)
                    s_corr = self.s[key] / (1 - self.beta2 ** self.t)

                    # update
                    self.params[key] -= self.lr * v_corr / (np.sqrt(s_corr) + self.eps)

    # ---------------- Training ----------------
    def fit(self, X, Y, epochs=1000, batch_size=None, verbose=100):
        m = X.shape[1]
        for epoch in range(1, epochs+1):
            # Shuffle
            perm = np.random.permutation(m)
            X_shuff, Y_shuff = X[:, perm], Y[:, perm]

            # Mini-batch
            for i in range(0, m, batch_size if batch_size else m):
                X_batch = X_shuff[:, i:i+batch_size] if batch_size else X_shuff
                Y_batch = Y_shuff[:, i:i+batch_size] if batch_size else Y_shuff

                cache = self.forward(X_batch)
                grads = self.backward(cache, Y_batch)
                self.update(grads)

            # Track loss on full dataset
            cache = self.forward(X)
            loss = self._compute_loss(cache[f"A{len(self.layer_sizes)-1}"], Y)
            if epoch % verbose == 0:
                acc = self.accuracy(X, Y)
                print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}, Acc: {acc:.2f}")

    def predict(self, X):
        cache = self.forward(X)
        return np.argmax(cache[f"A{len(self.layer_sizes)-1}"], axis=0)

    def accuracy(self, X, Y_true):
        Y_pred = self.predict(X)
        Y_true_labels = np.argmax(Y_true, axis=0)
        return np.mean(Y_pred == Y_true_labels)

    def confusion(self, X, Y_true):
        Y_pred = self.predict(X)
        Y_true_labels = np.argmax(Y_true, axis=0)
        return confusion_matrix(Y_true_labels, Y_pred)

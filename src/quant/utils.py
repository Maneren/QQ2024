import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import Callback


def accuracy(y_true, y_pred):
    """Calculate prediction accuracy based on sign matching."""
    sign_true = np.sign(y_true)
    sign_pred = np.sign(y_pred)
    return np.mean(sign_true == sign_pred)


def weighted_accuracy(y_true, y_pred):
    """Calculate weighted prediction accuracy based on prediction magnitude."""
    sign_true = np.sign(y_true)
    sign_pred = np.sign(y_pred)
    return np.average(sign_true == sign_pred, weights=np.abs(y_pred))


class PlotLosses(Callback):
    """Callback for tracking and optionally plotting training metrics."""

    def __init__(self, x_train, y_train, x_val, y_val):
        super(PlotLosses, self).__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.i: int = 0

    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}
        self.i = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.i += 1
        loss_train = logs.get("loss", 0.0)
        loss_val = logs.get("val_loss", 0.0)

        print()

        train_pred = self.model.predict(self.x_train)
        val_pred = self.model.predict(self.x_val)

        accuracy_train = accuracy(self.y_train, train_pred)
        accuracy_val = accuracy(self.y_val, val_pred)
        waccuracy_train = weighted_accuracy(self.y_train, train_pred)
        waccuracy_val = weighted_accuracy(self.y_val, val_pred)

        print(
            f"Epoch {self.i}: loss: {loss_train:.4f} val_loss: {loss_val:.4f}\n"
            f"Accuracy train: {accuracy_train:.4f} val: {accuracy_val:.4f}\n"
            f"Weighted accuracy train: {waccuracy_train:.4f} val: {waccuracy_val:.4f}"
        )


class ActivationLogger(Callback):
    """Callback for logging activation values during training."""

    def __init__(self, data):
        super().__init__()
        self.data = data
        self.epoch_activations = []

    def on_epoch_end(self, epoch, logs=None):
        if self.model is not None:
            layer_output = keras.Model(
                inputs=self.model.inputs,
                outputs=self.model.outputs,
            )
            activation_values = layer_output(self.data, training=False)
            self.epoch_activations.append(activation_values.numpy())


class FeatureSensitivityLogger(Callback):
    """Callback for logging feature sensitivity using gradient-based analysis."""

    def __init__(self, nd_data, scalar_data):
        super().__init__()
        self.mean_absolute_sensitivity_over_epochs = []
        self.mean_sensitivity_over_epochs = []
        nd_tensor = tf.convert_to_tensor(nd_data, dtype=tf.float32)
        scalar_tensor = tf.convert_to_tensor(scalar_data, dtype=tf.float32)
        self.inputs = [nd_tensor, scalar_tensor]

    def on_epoch_end(self, epoch, logs=None):
        if self.model is None:
            return

        with tf.GradientTape() as tape:
            tape.watch(self.inputs)
            predictions = self.model(self.inputs)

        gradients = tape.gradient(predictions, self.inputs)
        if gradients is None or len(gradients) < 2:
            return

        nd_sens, scalar_sens = gradients

        if nd_sens is None or scalar_sens is None:
            return

        # Convert IndexedSlices to dense tensors if needed, then to numpy
        if hasattr(nd_sens, "values"):
            nd_tensor = tf.convert_to_tensor(nd_sens)
            nd_numpy = nd_tensor.numpy()
        else:
            nd_tensor = tf.convert_to_tensor(nd_sens)
            nd_numpy = nd_tensor.numpy()

        if hasattr(scalar_sens, "values"):
            scalar_tensor = tf.convert_to_tensor(scalar_sens)
            scalar_numpy = scalar_tensor.numpy()
        else:
            scalar_tensor = tf.convert_to_tensor(scalar_sens)
            scalar_numpy = scalar_tensor.numpy()

        mean_sensitivity = np.mean(
            np.concatenate([scalar_numpy, nd_numpy.sum(axis=2)], axis=1), axis=0
        )
        self.mean_sensitivity_over_epochs.append(mean_sensitivity)

        scalar_abs = np.abs(scalar_numpy)
        nd_abs = np.abs(nd_numpy)

        mean_absolute_sensitivity = np.mean(
            np.concatenate([scalar_abs, nd_abs.sum(axis=2)], axis=1), axis=0
        )

        self.mean_absolute_sensitivity_over_epochs.append(mean_absolute_sensitivity)

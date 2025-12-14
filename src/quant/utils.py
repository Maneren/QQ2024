import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import Callback


def accuracy(y_true, y_pred):
    sign_true = np.sign(y_true)
    sign_pred = np.sign(y_pred)
    return np.mean(sign_true == sign_pred)


def weighted_accuracy(y_true, y_pred):
    sign_true = np.sign(y_true)
    sign_pred = np.sign(y_pred)
    return np.average(sign_true == sign_pred, weights=np.abs(y_pred))


class PlotLosses(Callback):
    def __init__(self, x_train, y_train, x_val, y_val):
        super(PlotLosses, self).__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.accuracy_train = []
        self.accuracy_val = []

        # Enable interactive mode for live updates
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.fig.tight_layout()

    def on_epoch_end(self, epoch, logs={}):
        loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        self.x.append(self.i)
        self.losses.append(loss)
        self.val_losses.append(val_loss)
        self.i += 1

        accuracy_train = accuracy(self.y_train, self.model.predict(self.x_train))
        accuracy_val = accuracy(self.y_val, self.model.predict(self.x_val))
        print(
            f"Accuracy train: {accuracy_train:.4f} val: {accuracy_val:.4f}\n"
            f"MSE loss train: {loss:.4f} val: {val_loss:.4f}"
        )
        self.accuracy_train.append(accuracy_train)
        self.accuracy_val.append(accuracy_val)

        # Clear the current axes and redraw
        self.ax.clear()
        self.ax.set_title("Training and Validation Loss/Accuracy")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss/Accuracy")

        self.ax.plot(self.x, self.losses, label="loss")
        self.ax.plot(self.x, self.val_losses, label="val_loss")

        self.ax.plot(self.x, self.accuracy_train, label="accuracy_train")
        self.ax.plot(self.x, self.accuracy_val, label="accuracy_val")
        self.ax.legend()

        # Update the plot
        plt.locator_params(axis="both", nbins=10)
        plt.draw()
        plt.pause(0.01)  # Small pause to allow plot to update

    def on_train_end(self, logs={}):
        # Turn off interactive mode and show final plot
        plt.ioff()
        plt.show()


class ActivationLogger(Callback):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.epoch_activations = []

    def on_epoch_end(self, epoch, logs=None):
        layer_output = keras.Model(
            inputs=self.model.inputs,
            outputs=self.model.outputs,
        )
        activation_values = layer_output(self.data, training=False)
        self.epoch_activations.append(activation_values.numpy())


class FeatureSensitivityLogger(Callback):
    def __init__(self, nd_data, scalar_data):
        super().__init__()
        self.mean_absolute_sensitivity_over_epochs = []
        self.mean_sensitivity_over_epochs = []
        self.nd_data = nd_data
        self.scalar_data = scalar_data

    def on_epoch_end(self, epoch, logs=None):
        nd_tensor = tf.convert_to_tensor(self.nd_data, dtype=tf.float32)
        scalar_tensor = tf.convert_to_tensor(self.scalar_data, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch([nd_tensor, scalar_tensor])
            predictions = self.model([nd_tensor, scalar_tensor])

        nd_sensitivities, scalar_sensitivities = tape.gradient(
            predictions, [nd_tensor, scalar_tensor]
        )

        scalar_numpy = scalar_sensitivities.numpy()
        nd_numpy = nd_sensitivities.numpy()

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

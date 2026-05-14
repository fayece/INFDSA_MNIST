import tensorflow as tf
import numpy as np


def pre_activation(x, w, b):
    z = np.dot(x, w) + b
    return z


def relu(z):
    return np.maximum(0, z)


def relu_tf(z):
    return tf.maximum(0., z)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_tf(z):
    return 1 / (1 + tf.exp(-z))


def softmax(z):
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)


def softmax_tf(z):
    exp_z = tf.exp(z)
    return exp_z / tf.reduce_sum(exp_z)


def create_nn_no_activation(shape_length: int = 784):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(shape_length,)),

        tf.keras.layers.Dense(256),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.15),

        tf.keras.layers.Dense(64),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.05),

        tf.keras.layers.Dense(10)
    ])

    return model


def create_nn(shape_length: int = 784):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(shape_length,)),

        tf.keras.layers.Dense(256),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(relu_tf),
        tf.keras.layers.Dropout(0.15),

        tf.keras.layers.Dense(64),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(relu_tf),
        tf.keras.layers.Dropout(0.05),

        tf.keras.layers.Dense(10, activation=softmax_tf)
    ])

    return model


def compile_nn(model):
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )


def fit_nn(model, x_train, y_train, epochs=100, batch_size=16, verbose=False, validation_split=0.1):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=12,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping],
        verbose=verbose
    )

    history.early_stopping = early_stopping

    return history

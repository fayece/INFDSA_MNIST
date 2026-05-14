import tensorflow as tf
from infdsa_mnist.activation_functions import pre_activation, relu


def get_hidden_layer_output(model, x_test):
    w, b = model.layers[0].get_weights()
    x = x_test[:10]

    z = pre_activation(x, w, b)

    has_activation = False
    for layer in model.layers[:3]:
        if isinstance(layer, tf.keras.layers.Activation):
            has_activation = True
            break

    if has_activation:
        hidden_layer_output = relu(z)
    else:
        hidden_layer_output = z

    return hidden_layer_output

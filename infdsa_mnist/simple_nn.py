import tensorflow as tf
import numpy as np
import os

def create_nn(shape_length: int = 784):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(shape_length,)),

        tf.keras.layers.Dense(256),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.15),

        tf.keras.layers.Dense(64),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.05),

        tf.keras.layers.Dense(10, activation='softmax'
        )
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


def evaluate_nn(model, x_test, y_test):
    return model.evaluate(x_test, y_test, verbose=0)


def save_nn(model, filename="mnist_model.keras", include_optimizer=True):
    os.makedirs("../models", exist_ok=True)
    filepath = f"../models/{filename}"

    if include_optimizer:
        model.save(filepath)
    else:
        temp_weights = filepath.replace('.keras', '_temp_weights.weights.h5')
        model.save_weights(temp_weights)

        config = model.get_config()
        model_fresh = tf.keras.Sequential.from_config(config)
        model_fresh.load_weights(temp_weights)
        model_fresh.save(filepath)

        os.remove(temp_weights)

        return model_fresh, filepath

    return model, filepath


def quantize_nn(x_train, model, filename="mnist_model_quantized.tflite"):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset():
        indices = np.random.permutation(len(x_train))[:300]
        for i in indices:
            # Yield the random samples one by one
            yield [x_train[i:i+1].astype(np.float32)]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()

    os.makedirs("../models", exist_ok=True)
    filepath = f"../models/{filename}"
    with open(filepath, "wb") as f:
        f.write(tflite_model)

    print(f"Quantized model saved to '{filepath}'")
    return filepath


def evaluate_quantized_nn(tflite_model_path, x_test, y_test):
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_scale, input_zero_point = input_details[0]['quantization']
    is_quantized_input = input_scale > 0

    correct_predictions = 0
    total = len(y_test)

    print(f"Evaluating TFLite model on {total} samples...")

    for i in range(total):
        test_sample = x_test[i:i + 1]

        if is_quantized_input:
            test_sample = test_sample / input_scale + input_zero_point
            test_sample = np.clip(test_sample, 0, 255).astype(np.uint8)
        else:
            test_sample = test_sample.astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], test_sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        predicted = np.argmax(output[0])

        if predicted == y_test[i]:
            correct_predictions += 1

    accuracy = correct_predictions / total
    print(f"Test samples: {total}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    return accuracy


def create_summary_data(filepath_full, filepath_optimized, filepath_quantized, acc, quantized_acc):
    full_kb = os.path.getsize(filepath_full) / 1024
    opt_kb = os.path.getsize(filepath_optimized) / 1024
    quant_kb = os.path.getsize(filepath_quantized) / 1024

    summary_data = [
        ["Full Model", f"{full_kb:.2f} KB", f"{acc * 100:.2f}%"],
        ["Optimized (No Optimizer)", f"{opt_kb:.2f} KB", f"{acc * 100:.2f}%"],
        ["Quantized (uint8)", f"{quant_kb:.2f} KB", f"{quantized_acc * 100:.2f}%"]
    ]

    return summary_data

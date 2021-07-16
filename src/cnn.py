import tensorflow.keras.layers as layers
import tensorflow as tf
from typing import Tuple
from src.metrics import make_keras_model_metrics


def main():
    pass


def create_cnn_model(input_shape: Tuple[int, ...], dropout: float = 0.2, num_convolutions: int = 2, learning_rate: float = 0.000006) -> tf.keras.Model:
    model_metrics = make_keras_model_metrics()

    inp = layers.Input(shape=input_shape)
    x = inp

    for _ in range(num_convolutions):
        x = layers.Conv1D(128, 3)(x)
        x = layers.MaxPooling1D(2)(x)
    x = layers.Flatten()(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate), metrics=model_metrics)
    model.summary()

    return model


if __name__ == '__main__':
    main()

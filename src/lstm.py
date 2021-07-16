import tensorflow.keras.layers as layers
import tensorflow as tf
from typing import Tuple
from src.metrics import make_keras_model_metrics


def main():
    pass


def create_lstm_model(input_shape: Tuple[int, ...], dropout: float = 0.2, lstm_units: int = 64, learning_rate: float = 0.000006) -> tf.keras.Model:
    model_metrics = make_keras_model_metrics()

    inp = layers.Input(shape=input_shape)
    lstm = layers.Dropout(dropout)(layers.Bidirectional(layers.LSTM(lstm_units)))(inp)
    out = layers.Dense(1, activation="sigmoid")(lstm)

    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate), metrics=model_metrics)
    model.summary()

    return model


if __name__ == '__main__':
    main()

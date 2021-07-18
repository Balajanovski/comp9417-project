import tensorflow.keras.layers as layers
import tensorflow as tf
from typing import Tuple
from src.metrics import make_keras_model_metrics
from src.util import load_data_word2vec_deep_learning
from tensorflow.keras.callbacks import EarlyStopping
from src.util import plot_keras_model_learning_curves
from sys import argv
import numpy as np
from src.metrics import get_metrics, print_metrics
from src.util import make_class_weights


def main():
    seq_len = 142
    train, val, test, y_train, y_val, y_test = load_data_word2vec_deep_learning(argv[1], portion_to_load=1.0, balance=True, sequence_length=seq_len)

    model = create_lstm_model((seq_len, 300))
    early_stopping = EarlyStopping(
        monitor="val_loss", verbose=1, patience=5, mode="min", restore_best_weights=True
    )

    history = model.fit(
        train,
        steps_per_epoch=3000,
        epochs=100,
        validation_data=val,
        shuffle=True,
        callbacks=[early_stopping],
        class_weight=make_class_weights(y_train),
        validation_steps=len(y_val)
    )

    plot_keras_model_learning_curves(history, prefix="lstm")

    y_pred = np.array([int(pred > 0.5) for pred in model.predict(test, steps=len(y_test))])
    metrics = get_metrics(y_pred, y_test)
    print_metrics(metrics)


def create_lstm_model(
    input_shape: Tuple[int, ...],
    dropout: float = 0.2,
    lstm_units: int = 64,
    learning_rate: float = 0.000006,
) -> tf.keras.Model:
    model_metrics = make_keras_model_metrics()

    inp = layers.Input(shape=input_shape)
    lstm = layers.Bidirectional(layers.LSTM(lstm_units))(inp)
    x = layers.Dropout(dropout)(lstm)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        metrics=model_metrics,
    )
    model.summary()

    return model


if __name__ == "__main__":
    main()

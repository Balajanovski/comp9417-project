import tensorflow.keras.layers as layers
import tensorflow as tf
from typing import Tuple
from src.metrics import make_keras_model_metrics
from src.util import load_data_word2vec_deep_learning
from tensorflow.keras.callbacks import EarlyStopping
from src.util import plot_keras_model_learning_curves
from sys import argv


def main():
    X_train, X_test, y_train, y_test = load_data_word2vec_deep_learning(argv[1])

    model = create_lstm_model(X_train.shape[1:])
    early_stopping = EarlyStopping(
        monitor="val_loss", verbose=1, patience=5, mode="min", restore_best_weights=True
    )
    history = model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=100,
        validation_split=0.2,
        shuffle=True,
        callbacks=[early_stopping],
    )

    model.evaluate(X_test, y_test)

    plot_keras_model_learning_curves(history, prefix="lstm")


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

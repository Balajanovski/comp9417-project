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
from sklearn.model_selection import train_test_split


def main():
    X_train, X_test, y_train, y_test = load_data_word2vec_deep_learning(argv[1], portion_to_load=1.0, balance=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)

    model = create_lstm_model(X_train.shape[1:])
    early_stopping = EarlyStopping(
        monitor="val_loss", verbose=1, patience=5, mode="min", restore_best_weights=True
    )
    class_weights = make_class_weights(y_train)
    print(f"Class weights: {class_weights}")

    history = model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=100,
        validation_data=(X_val, y_val),
        shuffle=True,
        callbacks=[early_stopping],
        class_weight=class_weights,
    )

    plot_keras_model_learning_curves(history, prefix="lstm")

    y_pred = np.array([int(pred > 0.5) for pred in model.predict(X_test)])
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

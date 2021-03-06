import tensorflow.keras.layers as layers
import tensorflow as tf
from typing import Tuple
from src.metrics import make_keras_model_metrics
from src.util import load_data_word2vec_deep_learning
from tensorflow.keras.callbacks import EarlyStopping
from src.util import plot_keras_model_learning_curves
from sys import argv
from src.util import make_class_weights
import numpy as np
from src.metrics import get_metrics, print_metrics


def main():
    seq_len = 142
    train, val, test, y_train, y_val, y_test = load_data_word2vec_deep_learning(argv[1], portion_to_load=1.0, balance=True, sequence_length=seq_len, portion_test_to_load=0.01)

    model = create_cnn_model((seq_len, 300))
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
        validation_steps=3000,
    )

    plot_keras_model_learning_curves(history, prefix="cnn")

    y_pred = []
    for pred in model.predict(test, steps=len(y_test)):
        y_pred.extend(pred > 0.5)
    y_pred = np.array(y_pred)

    metrics = get_metrics(y_pred, y_test)
    print_metrics(metrics)


def create_cnn_model(
    input_shape: Tuple[int, ...],
    dropout: float = 0.4,
    num_convolutions: int = 3,
    learning_rate: float = 0.000006,
) -> tf.keras.Model:
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
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        metrics=model_metrics,
    )
    model.summary()

    return model


if __name__ == "__main__":
    main()

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
    X_train, X_test, y_train, y_test = load_data_word2vec_deep_learning(argv[1], portion_to_load=0.1)

    model = create_cnn_model(X_train.shape[1:])
    early_stopping = EarlyStopping(
        monitor="val_loss", verbose=1, patience=5, mode="min", restore_best_weights=True
    )
    class_weights = make_class_weights(y_train)

    history = model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=100,
        validation_split=0.2,
        shuffle=True,
        callbacks=[early_stopping],
        class_weight=class_weights,
    )

    plot_keras_model_learning_curves(history, prefix="cnn")

    y_pred = np.array([int(pred > 0.5) for pred in model.predict(X_test)])
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

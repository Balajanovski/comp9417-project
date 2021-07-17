import tensorflow.keras.layers as layers
import tensorflow as tf
from typing import Tuple
from src.metrics import make_keras_model_metrics
from src.util import load_data_word2vec_deep_learning
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os
from plots import PLOTS_FOLDER_PATH


def main():
    X_train, X_test, y_train, y_test = load_data_word2vec_deep_learning()

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

    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.savefig(os.path.join(PLOTS_FOLDER_PATH, "lstm_accuracy_curve.png"))

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.savefig(os.path.join(PLOTS_FOLDER_PATH, "lstm_loss_curve.png"))


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

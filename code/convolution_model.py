from types import SimpleNamespace
import numpy as np
import tensorflow as tf


def get_CNN_model(
    conv_ns=tf.keras.layers,
    norm_ns=tf.keras.layers,
    drop_ns=tf.keras.layers,
    man_conv_ns=tf.keras.layers,
):

    Conv2D = conv_ns.Conv2D
    BatchNormalization = norm_ns.BatchNormalization
    Dropout = drop_ns.Dropout

    model = tf.keras.Sequential(
        [
        # conv layer with 16 filters
        Conv2D(filters=16, kernel_size=(3,3), strides=(2, 2), padding = 'SAME', activation='leaky_relu'),
        # batch normalization
        BatchNormalization(),
        # conv layer with 32 filters
        Conv2D(filters=32, kernel_size=(3,3), strides=(2, 2), padding = 'SAME', activation='leaky_relu'),
        # batch normalization
        BatchNormalization(),
        # max pooling layer
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        # conv layer with 64 filters
        Conv2D(filters=64, kernel_size=(3,3), strides=(2, 2), padding = 'SAME', activation='leaky_relu'),
        # batch normalization
        BatchNormalization(),
        # max pooling layer
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        # conv layer with 128 filters
        Conv2D(filters=128, kernel_size=(3,3), strides=(2, 2), padding = 'SAME', activation='leaky_relu'),
        # batch normalization
        BatchNormalization(),
        # max pooling layer
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        # dropout layer
        Dropout(rate = 0.1),
        # flatten layer
        tf.keras.layers.Flatten(),
        # dropout layer
        Dropout(rate = 0.1),
        # dense layer
        tf.keras.layers.Dense(64, activation='leaky_relu'),
        # dropout layer
        Dropout(rate=0.2),
        # second dense layer
        tf.keras.layers.Dense(32, activation='leaky_relu'),
        Dropout(rate=0.2),
        # final dense layer using softmax
        tf.keras.layers.Dense(10, activation='softmax')
        ]
    )


    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy", 
        metrics=["categorical_accuracy"],
    )

    return SimpleNamespace(model=model, epochs=50, batch_size=100)
from types import SimpleNamespace
import numpy as np
import tensorflow as tf


def get_CNN_model():

    Conv2D = tf.keras.layers.Conv2D
    BatchNormalization = tf.keras.layers.BatchNormalization
    Dropout = tf.keras.layers.Dropout
    
    input_prep_fn = tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(scale=1 / 255),
            tf.keras.layers.Resizing(32, 32),
        ]
    )
    output_prep_fn = tf.keras.layers.CategoryEncoding(
        num_tokens=10, output_mode="one_hot"
    )

    augment_fn = tf.keras.Sequential(
        [            
         tf.keras.layers.RandomTranslation(height_factor=0.2, width_factor=0.2)
        ]
    )

    model = CustomSequential(
        [
        # conv layer with 16 filters
        Conv2D(filters=16, kernel_size=(3,3), strides=(2, 2), padding = 'SAME', activation='leaky_relu'),
        # batch normalization
        BatchNormalization(),
        # conv layer with 100 filters
        Conv2D(filters=100, kernel_size=(3,3), strides=(2, 2), padding = 'SAME', activation='leaky_relu'),
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
        tf.keras.layers.Dense(100, activation='leaky_relu'),
        # dropout layer
        Dropout(rate=0.1),
        # final dense layer using softmax
        tf.keras.layers.Dense(10, activation='softmax')
        ], input_prep_fn=input_prep_fn, output_prep_fn=output_prep_fn, augment_fn=augment_fn
    )


    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy", 
        metrics=["categorical_accuracy"],
    )

    return SimpleNamespace(model=model, epochs=50, batch_size=100)



class CustomSequential(tf.keras.Sequential):
    """
    Subclasses tf.keras.Sequential to allow us to specify preparation functions that
    will modify input and output data.
    :param input_prep_fn: Modifies input images prior to running the forward pass
    :param output_prep_fn: Modifies input labels prior to running forward pass
    :param augment_fn: Augments input images prior to running forward pass
    """

    def __init__(
        self,
        *args,
        input_prep_fn=lambda x: x,
        output_prep_fn=lambda x: x,
        augment_fn=lambda x: x,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.input_prep_fn = input_prep_fn
        self.output_prep_fn = output_prep_fn
        self.augment_fn = augment_fn

    def batch_step(self, data, training=False):

        x_raw, y_raw = data

        x = self.input_prep_fn(x_raw)
        y = self.output_prep_fn(y_raw)
        if training:
            x = self.augment_fn(x)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=training)
            # Compute the loss value (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        if training:
            # Compute gradients
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update and return metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):
        return self.batch_step(data, training=True)

    def test_step(self, data):
        return self.batch_step(data, training=False)

    def predict_step(self, inputs):
        x = self.input_prep_fn(inputs)
        return self(x)
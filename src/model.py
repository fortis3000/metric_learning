import tensorflow as tf
from tensorflow.keras.applications import *
import tensorflow.keras as k

from src.config import MODEL_KIND, EMBEDDING_SIZE, IMG_SIZE


models_dict = {
    "EfficientNetB0": EfficientNetB0,
    "EfficientNetB1": EfficientNetB1,
    "EfficientNetB2": EfficientNetB2,
    "EfficientNetB3": EfficientNetB3,
    "EfficientNetB4": EfficientNetB4,
    "EfficientNetB5": EfficientNetB5,
    "EfficientNetB6": EfficientNetB6,
    "EfficientNetB7": EfficientNetB7,
    "ResNet50": ResNet50,
}


def build_metric_learning_model():
    input = k.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # backbone initialization
    backbone = models_dict[MODEL_KIND](
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        weights="imagenet",
        include_top=False,
        input_tensor=input,
    )

    # backbone.trainable = False  # (un)freeze
    # unfreeze some layers, where 0 - input, -1 - last activation
    for layer in backbone.layers:  # [-13:] for 4 Conv2d layers
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    # embedding model
    # GlobalAveragePooling2D transfer 2D image into 1d vector
    x = k.layers.GlobalAveragePooling2D(name="avg_pool_head")(backbone.output)
    output = k.layers.Dense(
        EMBEDDING_SIZE,
        activation="softplus",
        kernel_regularizer=tf.keras.regularizers.l2(),
        dtype="float32",
    )(x)

    # Compile
    model = k.Model(
        inputs=input,
        outputs=output,  # CHECK ORDER TO TRIPLET LOSS
        name="Metric_learning_model",
    )

    return model

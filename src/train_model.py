import os
import shutil
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa

from src.config_logging import logger
from src.config import (
    DATAPATH,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    MODEL_PATH,
    AUGMENT,
)
from src.dataset import init_dataset
from src.triplet_loss import batch_hard_triplet_loss
from src.model import build_metric_learning_model

tf.keras.backend.set_floatx("float32")


def train_model(
    train_ds: tf.data.Dataset, val_ds: tf.data.Dataset = None, epochs: int = 1
):
    model = build_metric_learning_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tfa.losses.TripletHardLoss(
            margin=1, soft=False, distance_metric="L2"
        ),
    )

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=5, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            verbose=1,
            restore_best_weights=True,
        ),
        # TODO: add Tensorboard
    ]

    model.fit(
        train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds
    )

    return model


def main():
    datasets = tfds.load("StanfordOnlineProducts", data_dir=DATAPATH)
    train_dataset, test_dataset = datasets["train"], datasets["test"]

    os.mkdir(MODEL_PATH)
    for f in [r"src/config.py", r"src/train_model.py"]:
        shutil.copy2(f, MODEL_PATH)

    train_ds = init_dataset(
        train_dataset,
        batch_size=BATCH_SIZE,
        buffer_size=BATCH_SIZE * 2,
        seed=42,
        shuffle=True,
        augment=AUGMENT,
        repeat=False,
        drop_reminder=False,
    )

    test_ds = init_dataset(
        test_dataset,
        batch_size=BATCH_SIZE,
        buffer_size=BATCH_SIZE * 2,
        seed=42,
        shuffle=False,
        augment=False,
        repeat=False,
        drop_reminder=False,
    )

    backbone = train_model(train_ds, test_ds, epochs=EPOCHS)
    backbone.save(os.path.join(MODEL_PATH, "model"))

    logger.info("Model trained successfully")


if __name__ == "__main__":
    main()

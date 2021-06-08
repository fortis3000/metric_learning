import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

from src.config_logging import logger
from src.config import BATCH_SIZE, LEARNING_RATE
from src.dataset import init_dataset
from src.triplet_loss import batch_hard_triplet_loss
from src.model import build_metric_learning_model
from src.triplet_loss_model import TripletLossModel

tf.keras.backend.set_floatx("float32")


def train_model(
    train_ds: tf.data.Dataset, val_ds: tf.data.Dataset = None, epochs: int = 1
):
    model = build_metric_learning_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=batch_hard_triplet_loss,
    )

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=3, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            verbose=1,
            restore_best_weights=True,
        ),
        # TODO: add Tensorboard
    ]

    model.fit(
        train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds
    )

    return model


def save_model(model: tf.keras.Model, filename: str):
    model.save(filename)


if __name__ == "__main__":
    # or env TFDS_DATA_DIR
    datasets = tfds.load(name="StanfordOnlineProducts", data_dir="data")
    train_dataset, test_dataset = datasets["train"], datasets["test"]

    train_ds = init_dataset(
        train_dataset,
        batch_size=BATCH_SIZE,
        buffer_size=BATCH_SIZE * 2,
        seed=42,
        shuffle=False,
        augment=False,
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

    backbone = tf.keras.models.load_model("model_hard_256", compile=False)

    model = TripletLossModel(model=backbone)
    model.set_train_from_ds(train_ds, is_labeled=True)

    logger.info(f"Models train labels shape: {model.train_labels.shape}")

    embeddings_test, labels_test_true = model.predict_embeddings(
        test_ds, is_labeled=True
    )

    logger.info(f"Embeddinds test shape: {embeddings_test.shape}")

    k = 5
    indices_test_pred = model.predict_indices(embeddings=embeddings_test, k=k)
    logger.info(f"Indices shape: {indices_test_pred.shape}")

    labels_test_pred = model.predict_labels_tf(
        model.train_labels, indices_test_pred
    )

    del embeddings_test
    del indices_test_pred

    labels_test_true = labels_test_true.numpy()

    accuracy = TripletLossModel.get_accuracy_cycle(
        labels_test_true, labels_test_pred
    )

    logger.info(f"Accuracy @{k}: {accuracy}")

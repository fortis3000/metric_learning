import os
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

from src.config_logging import logger
from src.config import BATCH_SIZE_EVAL, MODEL_PATH_PREDICT, k_max
from src.dataset import init_dataset
from src.triplet_loss_model import TripletLossModel

tf.keras.backend.set_floatx("float32")

if __name__ == "__main__":
    # or env TFDS_DATA_DIR
    datasets = tfds.load(name="StanfordOnlineProducts", data_dir="data")
    train_dataset, test_dataset = datasets["train"], datasets["test"]

    train_ds = init_dataset(
        train_dataset,
        batch_size=BATCH_SIZE_EVAL,
        buffer_size=BATCH_SIZE_EVAL * 2,
        seed=42,
        shuffle=False,
        augment=False,
        repeat=False,
        drop_reminder=False,
    )

    test_ds = init_dataset(
        test_dataset,
        batch_size=BATCH_SIZE_EVAL,
        buffer_size=BATCH_SIZE_EVAL * 2,
        seed=42,
        shuffle=False,
        augment=False,
        repeat=False,
        drop_reminder=False,
    )

    backbone = tf.keras.models.load_model(MODEL_PATH_PREDICT, compile=False)

    model = TripletLossModel(model=backbone)
    model.set_train_from_ds(train_ds, is_labeled=True)

    logger.info(f"Models train labels shape: {model.train_labels.shape}")

    embeddings_test, labels_test_true = model.predict_embeddings(
        test_ds, is_labeled=True
    )

    logger.info(f"Embeddings test shape: {embeddings_test.shape}")

    indices_test_pred = TripletLossModel.predict_indices(
        embeddings_train=model.train_embeddings,
        embeddings_test=embeddings_test,
        k=k_max,
    )

    logger.info(f"Indices shape: {indices_test_pred.shape}")

    labels_test_pred = model.predict_labels_tf(
        model.train_labels, indices_test_pred
    )

    del embeddings_test
    del indices_test_pred

    pd.DataFrame(labels_test_pred.numpy()).to_csv(
        os.path.join(MODEL_PATH_PREDICT, "preds.csv")
    )
    pd.DataFrame(labels_test_true.numpy()).to_csv(
        os.path.join(MODEL_PATH_PREDICT, "trues.csv")
    )

    # confusion matrix for k=1
    conf_matrix = tf.math.confusion_matrix(
        labels_test_true, labels_test_pred[:, 0]
    )
    pd.DataFrame(conf_matrix.numpy()).to_csv(
        os.path.join(MODEL_PATH_PREDICT, "confusion_matrix.csv")
    )

    true_diag = tf.linalg.diag_part(conf_matrix, k=0)

    true_sum = tf.reduce_sum(conf_matrix, axis=1)
    pred_sum = tf.reduce_sum(conf_matrix, axis=0)

    precision = true_diag / pred_sum
    recall = true_diag / true_sum
    f1_score = 2 * ((precision * recall) / (precision + recall))

    out_metrics = {
        "precision": precision.numpy().tolist(),
        "recall": recall.numpy().tolist(),
        "f1_scores": f1_score.numpy().tolist(),
        "conf_matrix": conf_matrix.numpy().tolist(),
    }

    logger.info(out_metrics)

    accuracy_1 = TripletLossModel.get_accuracy_cycle(
        labels_test_true, labels_test_pred[:, 0]
    )
    accuracy_k = TripletLossModel.get_accuracy_cycle(
        labels_test_true, labels_test_pred
    )

    logger.info(f"Accuracy @1: {accuracy_1}")
    logger.info(f"Accuracy @{k_max}: {accuracy_k}")

import numpy as np
import tensorflow as tf

from src.config_logging import logger
from src.knn import compute_distances_no_loops


class TripletLossModel:
    def __init__(self, model: tf.keras.Model):
        self.model = model

    def _predict_dataset(self, ds: tf.data.Dataset, is_labeled: bool = False):
        embeddings = []

        if is_labeled:
            labels = []
            for i in ds.take(-1):
                embeddings.append(self.model.predict(i[0]))
                labels.append(i[1])

            return (
                tf.cast(tf.concat(embeddings, axis=0), dtype=tf.float64),
                tf.cast(tf.concat(labels, axis=0), dtype=tf.float64),
            )

        else:
            for i in ds.take(-1):
                embeddings.append(self.model.predict(i[0]))

            return tf.cast(tf.concat(embeddings, axis=0), dtype=tf.float64)

    def set_train_from_ds(self, ds: tf.data.Dataset, is_labeled: bool = False):
        """Extracts images and labels from dataset provided.
        Then saves the embedding predicted.
        """
        logger.debug("Setting train set")

        self.train_embeddings, self.train_labels = self._predict_dataset(
            ds, is_labeled=is_labeled
        )

        logger.debug(
            f"Model train embeddings of shape {self.train_embeddings.shape}"
            "were set."
        )
        return True

    def predict_embeddings(
        self, ds: tf.data.Dataset, is_labeled: bool = False
    ):
        """Predicts dataset using base model. Use it for getting labels too."""
        logger.debug("Predicting embeddings")

        return self._predict_dataset(ds, is_labeled=is_labeled)

    @staticmethod
    def predict_indices(
        embeddings_train: tf.Tensor,
        embeddings_test: tf.Tensor,
        bsize: int = 16,
        k: int = 1,
    ):
        """Calculate train set indices for each embedding.

        TODO: return -1 * dists
        """
        logger.debug("Predicting indices")
        # memory issue, calcualting predicted labels step-by-step
        labels_test_indices = []

        n_batches = embeddings_test.shape[0] // bsize

        if embeddings_test.shape[0] % bsize > 0:
            n_batches += 1

        for i in range(n_batches):
            dists, indices = tf.math.top_k(
                -1
                * compute_distances_no_loops(
                    embeddings_train,
                    embeddings_test[i * bsize : (i + 1) * bsize, :],
                ),
                k=k,
                sorted=False,
            )
            labels_test_indices.append(indices)

        labels_test_indices = tf.concat(labels_test_indices, axis=0)

        return labels_test_indices

    @staticmethod
    def predict_labels_np(
        labels_train: np.array, labels_test_indices: np.array
    ):
        logger.debug("Predicting labels")
        return labels_train[labels_test_indices]

    @staticmethod
    def predict_labels_tf(labels_train, labels_test_indices: tf.Tensor):
        logger.debug("Predicting labels")
        return tf.gather(labels_train, labels_test_indices)

    # TODO: fix it
    @staticmethod
    def get_accuracy_np(labels_true, labels_pred):
        equality = np.any(np.equal(labels_true, labels_pred), axis=-1).astype(
            np.int8
        )
        return np.sum(equality, axis=-1) / labels_true.shape[0]

    @staticmethod
    def get_accuracy_cycle(labels_true, labels_pred):
        """Compares labels and calculates accuracy"""
        logger.debug("Getting accuracy")
        assert (
            labels_true.shape[0] == labels_pred.shape[0]
        ), "Provide sequences of the same length"

        mask = np.zeros(labels_pred.shape, dtype=bool)

        if tf.rank(labels_pred) == 2:
            for i in range(labels_pred.shape[0]):
                for j in range(labels_pred.shape[1]):
                    if labels_pred[i, j] == labels_true[i]:
                        mask[i, j] = True

            # could be several pictures in train of the same class
            mask = np.any(mask, axis=-1)

        elif tf.rank(labels_pred) == 1:
            for i in range(labels_pred.shape[0]):
                if labels_pred[i] == labels_true[i]:
                    mask[i] = True
        else:
            raise IndexError("Too many dimensions")

        return np.sum(mask.astype(np.int8)) / len(labels_true)

    @staticmethod
    def get_accuracy_tf(labels_true, labels_pred):
        """Compares labels and calculates accuracy. Quick but takes a lot of memory.

        :param labels_true:
        :param labels_pred:
        :return:
        """
        logger.debug("Getting accuracy")
        assert (
            labels_true.shape[0] == labels_pred.shape[0]
        ), "Provide sequences of the same length"

        # accuracy @ points > 1
        if labels_true.shape[0] != labels_pred.shape[0]:
            mask = tf.equal(
                tf.broadcast_to(labels_true, labels_pred.shape), labels_pred
            )
        else:
            mask = tf.equal(labels_true, labels_pred)

        if tf.rank(mask) > 1:
            mask = tf.reduce_any(mask, axis=-1)

        mask_number = tf.cast(mask, dtype=tf.int8)
        mask_number = tf.reduce_sum(mask_number, axis=-1)

        return mask_number / labels_true.shape[0]

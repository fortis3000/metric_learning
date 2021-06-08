from src.knn import compute_distances_no_loops

import numpy as np
import tensorflow as tf

eps = 1e-8

embedding0 = np.array([[0, 0, 0]], dtype=np.float64)
embedding1 = np.array([[1, 1, 1]], dtype=np.float64)
embedding2 = np.array([[2, 2, 2]], dtype=np.float64)
embedding0_tf = tf.convert_to_tensor(embedding0, dtype=tf.float64)
embedding1_tf = tf.convert_to_tensor(embedding1, dtype=tf.float64)
embedding2_tf = tf.convert_to_tensor(embedding2, dtype=tf.float64)

embedding0_32 = np.array([[0, 0, 0]], dtype=np.float32)
embedding1_32 = np.array([[1, 1, 1]], dtype=np.float32)
embedding2_32 = np.array([[2, 2, 2]], dtype=np.float32)
embedding0_tf_32 = tf.convert_to_tensor(embedding0, dtype=tf.float32)
embedding1_tf_32 = tf.convert_to_tensor(embedding1, dtype=tf.float32)
embedding2_tf_32 = tf.convert_to_tensor(embedding2, dtype=tf.float32)

embedding22_0 = np.array([[0, 0], [0, 0]])
embedding22_1 = np.array([[1, 1], [1, 1]])
embedding22_2 = np.array([[2, 2], [2, 2]])
embedding22_0_tf = tf.convert_to_tensor(embedding22_0, dtype=tf.float64)
embedding22_1_tf = tf.convert_to_tensor(embedding22_1, dtype=tf.float64)
embedding22_2_tf = tf.convert_to_tensor(embedding22_2, dtype=tf.float64)

embedding21_0 = np.array([[0, 0]])
embedding21_1 = np.array([[1, 1]])
embedding21_2 = np.array([[2, 2]])
embedding21_0_tf = tf.convert_to_tensor(embedding21_0, dtype=tf.float64)
embedding21_1_tf = tf.convert_to_tensor(embedding21_1, dtype=tf.float64)
embedding21_2_tf = tf.convert_to_tensor(embedding21_2, dtype=tf.float64)

embedding12_0 = np.array([[0], [0]])
embedding12_1 = np.array([[1], [1]])
embedding12_2 = np.array([[2], [2]])
embedding12_0_tf = tf.convert_to_tensor(embedding12_0, dtype=tf.float64)
embedding12_1_tf = tf.convert_to_tensor(embedding12_1, dtype=tf.float64)
embedding12_2_tf = tf.convert_to_tensor(embedding12_2, dtype=tf.float64)

# TODO add 32 precision tests
# TODO: add failed tests


def compute_dist_numpy(e1, e2):
    return np.sqrt(np.sum((e1 - e2) ** 2, axis=1))


def test_compute_distances_valid_11_numpy():
    assert (
        np.abs(
            compute_dist_numpy(embedding0, embedding0)
            - compute_distances_no_loops(embedding0, embedding0)
        )
        < eps
    )
    assert (
        np.abs(
            compute_dist_numpy(embedding0, embedding1)
            - compute_distances_no_loops(embedding0, embedding1)
        )
        < eps
    )
    assert (
        np.abs(
            compute_dist_numpy(embedding1, embedding0)
            - compute_distances_no_loops(embedding1, embedding0)
        )
        < eps
    )
    assert (
        np.abs(
            compute_dist_numpy(embedding1, embedding1)
            - compute_distances_no_loops(embedding1, embedding1)
        )
        < eps
    )


def test_compute_distances_valid_11_tf():
    assert (
        tf.abs(
            compute_dist_numpy(embedding0_tf, embedding0_tf)
            - compute_distances_no_loops(embedding0_tf, embedding0_tf)
        )
        < eps
    )
    assert (
        tf.abs(
            compute_dist_numpy(embedding0_tf, embedding1_tf)
            - compute_distances_no_loops(embedding0_tf, embedding1_tf)
        )
        < eps
    )
    assert (
        tf.abs(
            compute_dist_numpy(embedding1_tf, embedding0_tf)
            - compute_distances_no_loops(embedding1_tf, embedding0_tf)
        )
        < eps
    )
    assert (
        tf.abs(
            compute_dist_numpy(embedding1_tf, embedding1_tf)
            - compute_distances_no_loops(embedding1_tf, embedding1_tf)
        )
        < eps
    )


def test_compute_distances_valid_22_numpy():
    assert np.all(
        np.less(
            np.abs(
                compute_dist_numpy(embedding22_0, embedding22_0)
                - compute_distances_no_loops(embedding22_0, embedding22_0)
            ),
            eps,
        )
    )
    assert np.all(
        np.less(
            np.abs(
                compute_dist_numpy(embedding22_0, embedding22_1)
                - compute_distances_no_loops(embedding22_0, embedding22_1)
            ),
            eps,
        )
    )
    assert np.all(
        np.less(
            np.abs(
                compute_dist_numpy(embedding22_1, embedding22_0)
                - compute_distances_no_loops(embedding22_1, embedding22_0)
            ),
            eps,
        )
    )
    assert np.all(
        np.less(
            np.abs(
                compute_dist_numpy(embedding22_1, embedding22_1)
                - compute_distances_no_loops(embedding22_1, embedding22_1)
            ),
            eps,
        )
    )


def test_compute_distances_valid_22_tf():
    assert np.all(
        np.less(
            np.abs(
                compute_dist_numpy(embedding22_0_tf, embedding22_0_tf)
                - compute_distances_no_loops(
                    embedding22_0_tf, embedding22_0_tf
                )
            ),
            eps,
        )
    )
    assert np.all(
        np.less(
            np.abs(
                compute_dist_numpy(embedding22_0_tf, embedding22_1_tf)
                - compute_distances_no_loops(
                    embedding22_0_tf, embedding22_1_tf
                )
            ),
            eps,
        )
    )
    assert np.all(
        np.less(
            np.abs(
                compute_dist_numpy(embedding22_1_tf, embedding22_0_tf)
                - compute_distances_no_loops(
                    embedding22_1_tf, embedding22_0_tf
                )
            ),
            eps,
        )
    )
    assert np.all(
        np.less(
            np.abs(
                compute_dist_numpy(embedding22_1_tf, embedding22_1_tf)
                - compute_distances_no_loops(
                    embedding22_1_tf, embedding22_1_tf
                )
            ),
            eps,
        )
    )


def test_compute_distances_valid_21_numpy():
    assert np.all(
        np.less(
            np.abs(
                compute_dist_numpy(embedding21_0, embedding21_0)
                - compute_distances_no_loops(embedding21_0, embedding21_0)
            ),
            eps,
        )
    )
    assert np.all(
        np.less(
            np.abs(
                compute_dist_numpy(embedding21_0, embedding21_1)
                - compute_distances_no_loops(embedding21_0, embedding21_1)
            ),
            eps,
        )
    )
    assert np.all(
        np.less(
            np.abs(
                compute_dist_numpy(embedding21_1, embedding21_0)
                - compute_distances_no_loops(embedding21_1, embedding21_0)
            ),
            eps,
        )
    )
    assert np.all(
        np.less(
            np.abs(
                compute_dist_numpy(embedding21_1, embedding21_1)
                - compute_distances_no_loops(embedding21_1, embedding21_1)
            ),
            eps,
        )
    )


def test_compute_distances_valid_21_tf():
    assert np.all(
        np.less(
            np.abs(
                compute_dist_numpy(embedding21_0_tf, embedding21_0_tf)
                - compute_distances_no_loops(
                    embedding21_0_tf, embedding21_0_tf
                )
            ),
            eps,
        )
    )
    assert np.all(
        np.less(
            np.abs(
                compute_dist_numpy(embedding21_0_tf, embedding21_1_tf)
                - compute_distances_no_loops(
                    embedding21_0_tf, embedding21_1_tf
                )
            ),
            eps,
        )
    )
    assert np.all(
        np.less(
            np.abs(
                compute_dist_numpy(embedding21_1_tf, embedding21_0_tf)
                - compute_distances_no_loops(
                    embedding21_1_tf, embedding21_0_tf
                )
            ),
            eps,
        )
    )
    assert np.all(
        np.less(
            np.abs(
                compute_dist_numpy(embedding21_1_tf, embedding21_1_tf)
                - compute_distances_no_loops(
                    embedding21_1_tf, embedding21_1_tf
                )
            ),
            eps,
        )
    )


def test_compute_distances_valid_12_numpy():
    assert np.all(
        np.less(
            np.abs(
                compute_dist_numpy(embedding12_0, embedding12_0)
                - compute_distances_no_loops(embedding12_0, embedding12_0)
            ),
            eps,
        )
    )
    assert np.all(
        np.less(
            np.abs(
                compute_dist_numpy(embedding12_0, embedding12_1)
                - compute_distances_no_loops(embedding12_0, embedding12_1)
            ),
            eps,
        )
    )
    assert np.all(
        np.less(
            np.abs(
                compute_dist_numpy(embedding12_1, embedding12_0)
                - compute_distances_no_loops(embedding12_1, embedding12_0)
            ),
            eps,
        )
    )
    assert np.all(
        np.less(
            np.abs(
                compute_dist_numpy(embedding12_1, embedding12_1)
                - compute_distances_no_loops(embedding12_1, embedding12_1)
            ),
            eps,
        )
    )


def test_compute_distances_valid_12_tf():
    assert np.all(
        np.less(
            np.abs(
                compute_dist_numpy(embedding12_0_tf, embedding12_0_tf)
                - compute_distances_no_loops(
                    embedding12_0_tf, embedding12_0_tf
                )
            ),
            eps,
        )
    )
    assert np.all(
        np.less(
            np.abs(
                compute_dist_numpy(embedding12_0_tf, embedding12_1_tf)
                - compute_distances_no_loops(
                    embedding12_0_tf, embedding12_1_tf
                )
            ),
            eps,
        )
    )
    assert np.all(
        np.less(
            np.abs(
                compute_dist_numpy(embedding12_1_tf, embedding12_0_tf)
                - compute_distances_no_loops(
                    embedding12_1_tf, embedding12_0_tf
                )
            ),
            eps,
        )
    )
    assert np.all(
        np.less(
            np.abs(
                compute_dist_numpy(embedding12_1_tf, embedding12_1_tf)
                - compute_distances_no_loops(
                    embedding12_1_tf, embedding12_1_tf
                )
            ),
            eps,
        )
    )


def test_compute_distances_cross_types11():
    assert (
        tf.abs(
            compute_distances_no_loops(embedding0, embedding0)
            - compute_distances_no_loops(embedding0_tf, embedding0_tf)
        )
        < eps
    )
    assert (
        tf.abs(
            compute_distances_no_loops(embedding0, embedding1)
            - compute_distances_no_loops(embedding0_tf, embedding1_tf)
        )
        < eps
    )
    assert (
        tf.abs(
            compute_distances_no_loops(embedding1, embedding0)
            - compute_distances_no_loops(embedding1_tf, embedding0_tf)
        )
        < eps
    )
    assert (
        tf.abs(
            compute_distances_no_loops(embedding1, embedding1)
            - compute_distances_no_loops(embedding1_tf, embedding1_tf)
        )
        < eps
    )


def test_compute_distances_cross_types22():
    assert tf.reduce_all(
        tf.less(
            tf.abs(
                compute_distances_no_loops(embedding22_0, embedding22_0)
                - compute_distances_no_loops(
                    embedding22_0_tf, embedding22_0_tf
                )
            ),
            eps,
        )
    )
    assert tf.reduce_all(
        tf.less(
            tf.abs(
                compute_distances_no_loops(embedding22_0, embedding22_1)
                - compute_distances_no_loops(
                    embedding22_0_tf, embedding22_1_tf
                )
            ),
            eps,
        )
    )
    assert tf.reduce_all(
        tf.less(
            tf.abs(
                compute_distances_no_loops(embedding22_1, embedding22_0)
                - compute_distances_no_loops(
                    embedding22_1_tf, embedding22_0_tf
                )
            ),
            eps,
        )
    )
    assert tf.reduce_all(
        tf.less(
            tf.abs(
                compute_distances_no_loops(embedding22_1, embedding22_1)
                - compute_distances_no_loops(
                    embedding22_1_tf, embedding22_1_tf
                )
            ),
            eps,
        )
    )

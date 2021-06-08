import tensorflow as tf


def compute_distances_no_loops(X_train, X_test):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """

    if isinstance(X_train, tf.Tensor):
        X_train = tf.cast(X_train, dtype=tf.float64)
    else:
        X_train = tf.convert_to_tensor(X_train, dtype=tf.float64)

    if isinstance(X_test, tf.Tensor):
        X_test = tf.cast(X_test, dtype=tf.float64)
    else:
        X_test = tf.convert_to_tensor(X_test, dtype=tf.float64)

    # (x - y)^2 = x^2 - 2xy + y^2
    train_sq = tf.reduce_sum(tf.square(X_train), axis=1)
    test_sq = tf.reduce_sum(tf.square(X_test), axis=1)

    matmul = tf.matmul(X_test, X_train, transpose_b=True)

    # broadcasting
    dists = tf.math.sqrt(
        tf.math.add(
            tf.math.add(-2 * matmul, train_sq), tf.expand_dims(test_sq, axis=1)
        )
    )
    return dists

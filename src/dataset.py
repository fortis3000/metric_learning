import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from src.config import IMG_SIZE


def decode_image(image):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE), method="nearest")
    return image


def input_preprocess_train(record):
    """
    Preprocess image and labels after TFRecords reading.
    """
    label = tf.cast(record["super_class_id"], tf.int64)
    image = decode_image(record["image"])
    return image, label


def augment_image(image, label):
    rand_aug = np.random.choice([0, 1, 2, 3])

    if rand_aug == 0:
        image = tf.image.random_brightness(image, max_delta=0.4)
    elif rand_aug == 1:
        image = tf.image.random_contrast(image, lower=0.2, upper=0.5)
    elif rand_aug == 2:
        image = tf.image.random_hue(image, max_delta=0.2)
    else:
        image = tf.image.random_saturation(image, lower=0.2, upper=0.5)

    rand_aug = np.random.choice([0, 1, 2, 3])

    if rand_aug == 0:
        image = tf.image.random_flip_left_right(image)
    elif rand_aug == 1:
        image = tf.image.random_flip_up_down(image)
    elif rand_aug == 2:
        rand_rot = np.random.randn() * 45
        image = tfa.image.rotate(image, rand_rot)
    else:
        image = tfa.image.transform(
            image, [1.0, 1.0, -50, 0.0, 1.0, 0.0, 0.0, 0.0]
        )

    image = tf.image.random_crop(image, size=[100, 100, 3])
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))

    return image, label


def init_dataset(
    dataset,
    batch_size=2,
    buffer_size=1000,
    seed=42,
    shuffle=False,
    augment=False,
    repeat=True,
    drop_reminder=True,
):
    # preparing images and labels for model input
    dataset = dataset.map(
        input_preprocess_train,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    if augment:
        dataset = dataset.map(
            augment_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

    if repeat:
        dataset = dataset.repeat()

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=buffer_size, seed=seed, reshuffle_each_iteration=False
        )

    dataset = dataset.batch(
        batch_size=batch_size, drop_remainder=drop_reminder
    )
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

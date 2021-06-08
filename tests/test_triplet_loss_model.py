import numpy as np

from src.config_logging import logger
from src.triplet_loss_model import TripletLossModel

preds = np.array([[2, 1, 3], [2, 3, 1], [2, 3, 1]])
preds_bad = np.array([[3], [3], [3]])
trues = np.array([1, 1, 1])

preds1d = np.array([[1], [1], [1]])

# Labels truth only 1d
def test_get_accuracy_np():
    assert TripletLossModel.get_accuracy_np(trues, preds) == 1
    assert TripletLossModel.get_accuracy_np(trues, preds_bad) == 0
    assert TripletLossModel.get_accuracy_np(trues, preds1d) == 1


def test_get_accuracy_cycle():
    assert TripletLossModel.get_accuracy_cycle(trues, preds) == 1
    assert TripletLossModel.get_accuracy_cycle(trues, preds_bad) == 0
    assert TripletLossModel.get_accuracy_cycle(trues, preds1d) == 1


def test_get_accuracy_tf():
    assert TripletLossModel.get_accuracy_tf(trues, preds) == 1
    assert TripletLossModel.get_accuracy_tf(trues, preds_bad) == 0
    assert TripletLossModel.get_accuracy_tf(trues, preds1d) == 1


def test_predict_labels_np():
    labels_train = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    test_indices = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 8],
            [8, 9],
            [9, 9],
        ]
    )
    logger.debug(
        TripletLossModel.predict_labels_np(labels_train, test_indices)
    )
    assert np.all(
        np.equal(
            TripletLossModel.predict_labels_np(labels_train, test_indices),
            test_indices,
        )
    )


def test_predict_labels_tf():
    labels_train = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    test_indices = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 8],
            [8, 9],
            [9, 9],
        ]
    )
    logger.debug(
        TripletLossModel.predict_labels_tf(labels_train, test_indices)
    )
    assert np.all(
        np.equal(
            TripletLossModel.predict_labels_tf(labels_train, test_indices),
            test_indices,
        )
    )


def test_predict_indices():

    train = np.array([[0, 0, 0], [1, 1, 1]])
    preds = np.array([[0, 0, 0], [0.9, 0.9, 0.9]])
    answer = np.array([[0], [1]])

    print(TripletLossModel.predict_indices(train, preds))
    assert np.all(
        np.equal(TripletLossModel.predict_indices(train, preds), answer)
    )

import numpy as np
from PIL import Image
import pytest

from utils import crop_resize_image, get_worst_preds


class FakeBatch:
    def __init__(self, classes, filepaths):
        self.classes = classes
        self.filepaths = filepaths
        self.reset_called = False

    def reset(self):
        self.reset_called = True


def test_crop_resize_image_non_square():
    # Create a rectangular image and ensure it is cropped and resized correctly
    img = Image.new("RGB", (300, 200), color=(255, 0, 0))
    resized = crop_resize_image(img, target=(100, 100))
    assert resized.size == (100, 100)


def test_get_worst_preds_thresholds():
    classes = [0, 1, 0]
    filepaths = ["a.jpg", "b.jpg", "c.jpg"]
    batch = FakeBatch(classes, filepaths)
    # Predictions: shape (3,2)
    Y_pred = np.array([
        [0.6, 0.4],
        [0.2, 0.8],
        [0.3, 0.7],
    ])
    # No predictions below 0.1
    rotten = get_worst_preds(batch, Y_pred, threshold=0.1)
    assert batch.reset_called
    assert rotten == []
    # Predictions below 0.5 should yield one worst pred at index 2
    rotten = get_worst_preds(batch, Y_pred, threshold=0.5)
    assert len(rotten) == 1
    fp, true_c, pred_c, conf = rotten[0]
    assert fp == "c.jpg"
    assert true_c == 0
    assert isinstance(pred_c, (int, np.integer))
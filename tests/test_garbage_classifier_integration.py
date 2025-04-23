import os
import numpy as np
import tensorflow as tf
import pytest

from PIL import Image
from garbage.garbage_classifier import GarbageClassifier
from tests.conftest import DummyTracker


@pytest.fixture
def tiny_dataset(tmp_path):
    # Create directory with two classes and 4 images each
    data_dir = tmp_path / "data"
    classes = ["class0", "class1"]
    for cls in classes:
        cls_dir = data_dir / cls
        cls_dir.mkdir(parents=True)
        for i in range(4):
            img = Image.new("RGB", (224, 224), color=(i * 30, i * 30, i * 30))
            img.save(cls_dir / f"{cls}_{i}.jpg")
    return str(data_dir), classes


def test_end_to_end_training(tiny_dataset, monkeypatch):
    data_path, classes = tiny_dataset
    # Stub MobileNetV3Large to a simple sequential model to avoid downloading weights
    def fake_mnv3(**kwargs):
        return tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=kwargs.get("input_shape")),
            tf.keras.layers.Flatten(),
        ])
    monkeypatch.setattr(tf.keras.applications, "MobileNetV3Large", fake_mnv3)
    tracker = DummyTracker()
    classifier = GarbageClassifier(class_names=classes, batch_size=2, tracker=tracker)
    classifier.val_split = 0.5
    # Load dataset
    classifier.load_dataset(train_path=data_path, validation_path=data_path)
    assert classifier.train_batches.samples > 0
    assert classifier.valid_batches.samples > 0
    # Build, compile, and fit one epoch
    classifier.build_model(hidden1_size=4, hidden2_size=2, l2_param=0.0, dropout_factor=0.0, bias_regularizer=None)
    classifier.compile(learning_rate=0.001, loss="categorical_crossentropy")
    history = classifier.fit(epochs=1)
    # Check training occurred
    assert "loss" in history.history
    assert len(history.history["loss"]) == 1
    # Test prediction API on validation set
    preds = classifier.predict(classifier.valid_batches)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[1] == len(classes)
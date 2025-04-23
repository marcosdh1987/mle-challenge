import tensorflow as tf
import pytest

from garbage.garbage_classifier import GarbageClassifier
from tests.conftest import DummyTracker


def test_build_model_and_compile(monkeypatch):
    # Stub MobileNetV3Large to return a minimal sequential model
    def fake_mnv3(**kwargs):
        return tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=kwargs.get("input_shape")),
            tf.keras.layers.Flatten(),
        ])
    monkeypatch.setattr(tf.keras.applications, "MobileNetV3Large", fake_mnv3)
    tracker = DummyTracker()
    classifier = GarbageClassifier(class_names=["a", "b"], batch_size=2, tracker=tracker)
    # Build with test parameters
    classifier.build_model(
        hidden1_size=4,
        hidden2_size=2,
        l2_param=0.01,
        dropout_factor=0.1,
        bias_regularizer="l1",
    )
    # Model should be created
    assert hasattr(classifier, "model")
    # Tracker should have recorded config
    assert tracker.configs, "Tracker did not record any configurations"
    # Compile and check callbacks
    classifier.compile(
        learning_rate=0.001,
        loss="categorical_crossentropy",
        min_lr=1e-5,
        decrease_factor=0.5,
        patience=2,
    )
    assert hasattr(classifier, "_callbacks")
    # Expect ReduceLROnPlateau callback present
    assert any(cb.__class__.__name__ == "ReduceLROnPlateau" for cb in classifier._callbacks)
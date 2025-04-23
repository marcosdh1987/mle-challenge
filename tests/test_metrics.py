import numpy as np
import matplotlib
import pytest

# Use non-interactive backend for plotting tests
matplotlib.use("Agg")

from metrics import print_confusion_matrix, plot_accuracy, plot_loss


class FakeBatch:
    def __init__(self, classes):
        self.classes = classes


def test_print_confusion_matrix(capsys):
    classes = [0, 1, 0, 1]
    Y_pred = np.array([
        [0.9, 0.1],
        [0.4, 0.6],
        [0.3, 0.7],
        [0.2, 0.8],
    ])
    class_names = ["a", "b"]
    batch = FakeBatch(classes)
    # Should not raise and should print report
    print_confusion_matrix(None, batch, Y_pred, class_names, print_report=True)
    captured = capsys.readouterr()
    assert "Classification Report" in captured.out


def test_plot_accuracy_and_loss():
    # Should run without errors
    train_acc = [0.5, 0.6, 0.7]
    val_acc = [0.4, 0.55, 0.65]
    plot_accuracy(train_acc, val_acc)
    plot_loss([1, 0.8, 0.6], [1.2, 1.0, 0.9])
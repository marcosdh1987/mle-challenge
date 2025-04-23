#!/usr/bin/env python3
"""
Command-line entry point for training the garbage classification model.
"""
import argparse
import os
import yaml

from src.garbage.garbage_classifier import GarbageClassifier
from src.garbage.mlflow_tracker import MLFlowTracker


def main():
    parser = argparse.ArgumentParser(
        description="Train the Garbage Classification Model"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/config.yaml",
        help="Path to the YAML config file"
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Dataset configuration
    dataset_cfg = config["dataset"]
    train_path = dataset_cfg["train_path"]
    validation_path = dataset_cfg.get("validation_path", train_path)
    class_names = dataset_cfg["class_names"]
    batch_size = dataset_cfg["batch_size"]

    # Model configuration
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    experiment_name = config.get("experiment", {}).get("name", "garbage-classifier")
    output_cfg = config.get("output", {})

    # Initialize experiment tracker
    tracker = MLFlowTracker(experiment_name)

    # Initialize classifier
    classifier = GarbageClassifier(
        class_names=class_names,
        batch_size=batch_size,
        tracker=tracker
    )
    classifier.val_split = dataset_cfg.get("val_split", classifier.val_split)

    # Load data
    classifier.load_dataset(
        train_path=train_path,
        validation_path=validation_path
    )

    # Build model
    classifier.build_model(
        hidden1_size=model_cfg.get("hidden_sizes", [])[0],
        hidden2_size=model_cfg.get("hidden_sizes", [None, None])[1],
        l2_param=model_cfg.get("l2_param", 0.0),
        dropout_factor=model_cfg.get("dropout", 0.0),
        bias_regularizer=None
    )

    # Compile model
    classifier.compile(
        learning_rate=training_cfg.get("learning_rate", 0.001),
        loss=training_cfg.get("loss", "categorical_crossentropy"),
        min_lr=training_cfg.get("min_lr", 1e-5),
        decrease_factor=training_cfg.get("decrease_factor", 0.1),
        patience=training_cfg.get("patience", 5)
    )

    # Train model
    classifier.fit(epochs=training_cfg.get("epochs", 10))

    # Save model
    model_dir = output_cfg.get("model_dir", "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, output_cfg.get("model_name", "garbage_model.h5"))
    classifier.model.save(model_path)
    tracker.track_artifacts(model_path)
    tracker.finish_run()

    print(f"Training completed. Model saved to {model_path}")


if __name__ == "__main__":
    main()
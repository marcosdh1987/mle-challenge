import io
import os
from urllib.parse import urlparse

import mlflow
import numpy as np
import uvicorn
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
# Import MlflowClient
from mlflow.tracking import MlflowClient
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.preprocessing import image

from src.garbage.garbage_classifier import GarbageClassifier

"""FastAPI application for garbage classification using a pre-trained model registered with MLflow."""
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define class names (should match your training)
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Define the registered model name used in training
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME","GarbageClassifierModel")
MODEL_STAGE = os.getenv("MODEL_STAGE","Production")  # Use the Production stage with correct casing

app = FastAPI(
    title="Garbage Classification API", description="Predicts waste type from an image."
)

# Determine running environment (default: local) and optionally read external model URI in prod
import os
ENV = os.getenv("ENV", "local")  # Set ENV=prod for production
logger.info(f"Running in {ENV} mode.")

model = None
try:
    if ENV == "prod":
        # In production, MODEL_URI should point to external storage (e.g., s3://bucket/model.h5 or Azure Blob)
        MODEL_URI = os.getenv("MODEL_URI")
        if not MODEL_URI:
            raise Exception("MODEL_URI environment variable not set in production mode")
        # Mock: here you would download the file from S3/Blob to local path or load directly
        h5_path = MODEL_URI  # Using URI directly for demonstration
        logger.info(f"Production mode: loading model from external URI: {h5_path}")
    else:
        # Local mode: load from MLflow tracking SQLite database
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        db_path = os.path.join(project_root, "mlruns.db")
        absolute_db_uri = f"sqlite:///{os.path.abspath(db_path)}"
        logger.info(f"Setting MLflow tracking URI to: {absolute_db_uri}")
        mlflow.set_tracking_uri(absolute_db_uri)
        logger.info(f"Setting MLflow registry URI to: {absolute_db_uri}")
        mlflow.set_registry_uri(absolute_db_uri)

        client = MlflowClient()
        versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=[MODEL_STAGE])
        if not versions:
            raise Exception(
                f"No versions found for model '{REGISTERED_MODEL_NAME}' in stage '{MODEL_STAGE}'"
            )
        ver = versions[0]
        artifact_dir_uri = ver.source
        parsed = urlparse(artifact_dir_uri)
        artifact_dir_path = os.path.abspath(parsed.path)
        h5_path = os.path.join(artifact_dir_path, "data", "model.h5")
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Model file not found: {h5_path}")
        logger.info("Loaded local model .h5 path from MLflow artifacts.")

    # Load API config and rebuild architecture
    config_path = os.path.join(project_root, "config", "config.yaml")
    with open(config_path, "r") as cf:
        api_cfg = yaml.safe_load(cf)
    model_cfg = api_cfg.get("model", {})
    hidden_sizes = model_cfg.get("hidden_sizes", [None, None])
    l2_param = model_cfg.get("l2_param", 0.0)
    dropout = model_cfg.get("dropout", 0.0)
    batch_size = api_cfg["dataset"]["batch_size"]
    # Dummy tracker stub
    class DummyTracker:
        def track_config(self, *_):
            pass

        def get_callback(self):
            return None

    # Rebuild model
    classifier = GarbageClassifier(
        class_names=CLASSES, batch_size=batch_size, tracker=DummyTracker()
    )
    classifier.build_model(
        hidden1_size=hidden_sizes[0],
        hidden2_size=hidden_sizes[1],
        l2_param=l2_param,
        dropout_factor=dropout,
        bias_regularizer=None,
    )
    # Load weights
    classifier.model.load_weights(h5_path)
    model = classifier.model
    logger.info("Rebuilt model architecture and loaded weights successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    logger.error(
        "Ensure the model is registered, the stage is set, MLflow URIs are correct, and the .h5 file exists."
    )


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict garbage class from an uploaded image.

    Args:
        file (UploadFile): The uploaded image file.

    Returns:
        JSONResponse: Prediction results including class and confidence.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid image type.")
    try:
        contents = await file.read()
        # Load image from raw bytes
        img = image.load_img(io.BytesIO(contents), target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = model.predict(x)[0]
        pred_idx = int(np.argmax(preds))
        pred_class = CLASSES[pred_idx]
        confidence = float(preds[pred_idx])
        return JSONResponse(
            {
                "predicted_class": pred_class,
                "confidence": confidence,
                "all_confidences": dict(zip(CLASSES, map(float, preds))),
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


@app.get("/health")
def health():
    """
    Health check endpoint.

    Returns:
        dict: Status of the service and model loading state.
    """
    return {"status": "ok", "model_loaded": model is not None}


if __name__ == "__main__":
    # No need to set tracking URI again here if set above
    uvicorn.run(
        "main:app", host="0.0.0.0", port=8000, reload=True
    )  # Corrected module name for uvicorn

import sys
import os
import pathlib
import types
import pytest

def pytest_configure(config):
    import sys
    import types
    # Stub TensorFlow and Keras modules as early as possible
    tf = types.ModuleType('tensorflow')
    tf.keras = types.ModuleType('tensorflow.keras')
    tf.keras.applications = types.ModuleType('tensorflow.keras.applications')
    mobilenet_v3 = types.ModuleType('tensorflow.keras.applications.mobilenet_v3')
    mobilenet_v3.preprocess_input = lambda x: x
    def _dummy_mnv3large(**kwargs):
        class DummyModel:
            def __init__(self, *a, **k):
                pass
        return DummyModel()
    mobilenet_v3.MobileNetV3Large = _dummy_mnv3large
    tf.keras.applications.mobilenet_v3 = mobilenet_v3
    setattr(tf.keras.applications, "MobileNetV3Large", _dummy_mnv3large)
    tf.keras.preprocessing = types.ModuleType('tensorflow.keras.preprocessing')
    image_mod = types.ModuleType('tensorflow.keras.preprocessing.image')
    import numpy as _np
    from PIL import Image as _PILImage
    def load_img(fp, target_size=None):
        img = _PILImage.open(fp)
        if target_size:
            img = img.resize(target_size)
        return img
    def img_to_array(img):
        return _np.array(img)
    image_mod.load_img = load_img
    image_mod.img_to_array = img_to_array
    tf.keras.preprocessing.image = image_mod
    # Patch DummyImageDataGenerator for integration tests
    class DummyBatch:
        def __init__(self):
            self.samples = 4
            self.classes = [0, 1, 0, 1]
            self.filepaths = [f"img_{i}.jpg" for i in range(4)]
            self.batch_size = 2
            self.shape = (4, 224, 224, 3)  # Simula un batch de 4 imágenes 224x224x3
        def __iter__(self):
            return iter([([0], [1])])
    class DummyImageDataGenerator:
        def __init__(self, *args, **kwargs):
            pass
        def flow_from_directory(self, *args, **kwargs):
            return DummyBatch()
    setattr(image_mod, 'ImageDataGenerator', DummyImageDataGenerator)
    layers_mod = types.ModuleType('tensorflow.keras.layers')
    # Make all layer stubs accept any arguments
    for cls_name in ['BatchNormalization', 'Dense', 'Dropout', 'GlobalAveragePooling2D', 'InputLayer', 'Flatten']:
        setattr(layers_mod, cls_name, type(cls_name, (), {"__init__": lambda self, *a, **k: None}))
    tf.keras.layers = layers_mod
    models_mod = types.ModuleType('tensorflow.keras.models')
    class DummyHistory:
        def __init__(self):
            self.history = {"loss": [1.0]}
    class DummySequential(list):
        def add(self, x):
            self.append(x)
        def compile(self, *a, **k):
            pass
        def predict(self, *a, **k):
            x = a[0] if a else None
            batch = x.shape[0] if x is not None else 1
            import numpy as np
            return np.ones((batch, 2)) / 2
        def fit(self, *a, **k):
            return DummyHistory()
    setattr(tf.keras, 'Sequential', DummySequential)
    setattr(models_mod, 'Sequential', DummySequential)
    sys.modules['tensorflow'] = tf
    sys.modules['tensorflow.keras'] = tf.keras
    sys.modules['tensorflow.keras.applications'] = tf.keras.applications
    sys.modules['tensorflow.keras.applications.mobilenet_v3'] = mobilenet_v3
    sys.modules['tensorflow.keras.preprocessing'] = tf.keras.preprocessing
    sys.modules['tensorflow.keras.preprocessing.image'] = image_mod
    sys.modules['tensorflow.keras.layers'] = layers_mod
    sys.modules['tensorflow.keras.models'] = models_mod
    optim_mod = types.ModuleType('tensorflow.keras.optimizers')
    setattr(optim_mod, 'Adam', lambda *a, **k: None)
    sys.modules['tensorflow.keras.optimizers'] = optim_mod
    # Stub regularizers
    regularizers_mod = types.ModuleType('tensorflow.keras.regularizers')
    regularizers_mod.l2 = lambda *a, **k: None
    regularizers_mod.l1 = lambda *a, **k: None
    sys.modules['tensorflow.keras.regularizers'] = regularizers_mod
    setattr(tf.keras, 'regularizers', regularizers_mod)
    # Stub callbacks
    callbacks_mod = types.ModuleType('tensorflow.keras.callbacks')
    class ReduceLROnPlateau:
        def __init__(self, *a, **k): pass
    setattr(callbacks_mod, 'ReduceLROnPlateau', ReduceLROnPlateau)
    sys.modules['tensorflow.keras.callbacks'] = callbacks_mod
    setattr(tf.keras, 'callbacks', callbacks_mod)

# Stub MLflow modules to prevent import errors in tests
sys.modules['mlflow'] = types.ModuleType('mlflow')
mlflow_tracking = types.ModuleType('mlflow.tracking')
def _dummy_get_latest_versions(name, stages=None):
    # Return empty list to simulate no registered model versions
    return []
mlflow_tracking.MlflowClient = lambda *args, **kwargs: types.SimpleNamespace(get_latest_versions=_dummy_get_latest_versions)
sys.modules['mlflow.tracking'] = mlflow_tracking

# Also stub uvicorn to avoid import errors
sys.modules['uvicorn'] = types.ModuleType('uvicorn')

# Patch FastAPI to skip python-multipart check during route definitions
try:
    import fastapi.dependencies.utils as _fa_utils
    _fa_utils.check_file_field = lambda *args, **kwargs: None
except ImportError:
    pass
  
# Override GarbageClassifier.build_model to no-op to avoid heavy model building during import
try:
    import garbage.garbage_classifier as _gc_mod
    _gc_mod.GarbageClassifier.build_model = lambda self, *args, **kwargs: None
except ImportError:
    pass

# Add project root and src to PYTHONPATH for imports
root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root))

class DummyTracker:
    """
    Dummy tracker to collect config tracking calls and provide no-op callback.
    """
    def __init__(self):
        self.configs = []

    def track_config(self, cfg):
        self.configs.append(cfg)

    def get_callback(self):
        return None

# Patch PIL.Image.ANTIALIAS for Pillow>=10 compatibility in tests
import PIL.Image
if not hasattr(PIL.Image, "ANTIALIAS"):
    try:
        PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS
    except AttributeError:
        pass
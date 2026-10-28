"""Background training worker for the Classifier tool."""

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from falcocv.core.classifier.inference import ClassifierInferenceEngine
from falcocv.core.classifier.classifier_memory import ClassifierMemoryBank


class TrainWorker(QObject):
    """Runs classifier training in a background thread."""

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, class_data: dict, class_labels: dict, params: dict, k_nn: int):
        super().__init__()
        self.class_data = class_data
        self.class_labels = class_labels
        self.params = params
        self.k_nn = k_nn
        self.result_engine: ClassifierInferenceEngine | None = None

    def run(self):
        try:
            valid_classes = {cid: imgs for cid, imgs in self.class_data.items() if len(imgs) >= 5}
            if len(valid_classes) < 2:
                self.error.emit("Need >=2 classes with >=5 images each.")
                return

            engine = ClassifierInferenceEngine()
            self.progress.emit(10, "Extracting features...")

            features_by_class: dict[str, np.ndarray] = {}
            total = len(valid_classes)
            for idx, (cid, imgs) in enumerate(valid_classes.items()):
                feats = []
                for img in imgs:
                    try:
                        f = engine.extract_features(img)
                        feats.append(f)
                    except Exception:
                        continue
                if len(feats) < 5:
                    continue
                features_by_class[cid] = np.stack(feats)
                pct = 10 + int((idx + 1) / total * 30)
                label = self.class_labels.get(cid, cid)
                self.progress.emit(pct, f"Extracted: {label}")

            if len(features_by_class) < 2:
                self.error.emit("Need >=2 classes with valid feature extraction.")
                return

            self.progress.emit(50, "Training projection head (triplet loss)...")
            from falcocv.core.classifier.trainer import train_projection_head
            projection = train_projection_head(features_by_class)

            self.progress.emit(75, "Building memory bank...")
            memory_bank = ClassifierMemoryBank()
            for cid, feats in features_by_class.items():
                projected = projection.apply(feats)
                label = self.class_labels.get(cid, cid)
                memory_bank.add_class(cid, label, projected)

            thresholds = memory_bank.calibrate_thresholds()

            engine.projection = projection
            engine.memory_bank = memory_bank
            engine.thresholds = thresholds
            engine.k_nn = self.k_nn
            self.result_engine = engine

            class_summary = ", ".join(
                f"{self.class_labels.get(c, c)} ({len(fs)})"
                for c, fs in features_by_class.items()
            )
            msg = f"Trained {len(features_by_class)} classes: {class_summary}"
            self.progress.emit(100, msg)
            self.finished.emit(msg)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

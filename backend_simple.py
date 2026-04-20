import os
from typing import Dict, Tuple

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ResNet101_binario.keras")
FRONTEND_DIR = os.path.join(BASE_DIR, "Pagina")

IMG_SIZE: Tuple[int, int] = (224, 224)
CLASS_NAMES = ["no_cancer", "cancer"]

app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app)

model = None
model_load_error = None

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as exc:
    model_load_error = str(exc)


def preprocess_image(file_bytes: bytes) -> np.ndarray:
    buffer = np.frombuffer(file_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("No se pudo decodificar la imagen.")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, IMG_SIZE)

    batch = np.expand_dims(image_rgb.astype(np.float32), axis=0)
    batch = tf.keras.applications.resnet.preprocess_input(batch)
    return batch


def decode_prediction(raw_pred: np.ndarray) -> Dict[str, float]:
    raw_pred = np.array(raw_pred)

    if raw_pred.ndim == 2 and raw_pred.shape[1] == 1:
        # En binario con sigmoide, normalmente la salida representa probabilidad de la clase 1.
        # Ajustado al orden actual: clase 0 = no_cancer, clase 1 = cancer.
        p_cancer = float(raw_pred[0, 0])
        p_no_cancer = 1.0 - p_cancer
    elif raw_pred.ndim == 2 and raw_pred.shape[1] >= 2:
        # Modelo con 2 neuronas (softmax o logits)
        probs = raw_pred[0].astype(np.float64)
        probs_sum = probs.sum()
        probs_valid = np.all(probs >= 0.0) and np.all(probs <= 1.0) and abs(probs_sum - 1.0) < 1e-3

        if not probs_valid:
            probs = tf.nn.softmax(probs).numpy()

        p_cancer = float(probs[0])
        p_no_cancer = float(probs[1])
    else:
        raise ValueError("Forma de salida del modelo no soportada.")

    label = "cancer" if p_cancer >= p_no_cancer else "no_cancer"
    confidence = max(p_cancer, p_no_cancer)

    return {
        "label": label,
        "confidence": confidence,
        "prob_cancer": p_cancer,
        "prob_no_cancer": p_no_cancer,
    }


@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/health")
def health():
    if model_load_error:
        return jsonify({"status": "error", "detail": model_load_error}), 500
    return jsonify({"status": "ok", "model": os.path.basename(MODEL_PATH)})


@app.post("/api/predict")
def predict():
    if model_load_error:
        return jsonify({"error": "El modelo no pudo cargarse.", "detail": model_load_error}), 500

    if "image" not in request.files:
        return jsonify({"error": "No se envio el archivo en el campo 'image'."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Nombre de archivo vacio."}), 400

    try:
        image_batch = preprocess_image(file.read())
        pred = model.predict(image_batch, verbose=0)
        decoded = decode_prediction(pred)

        return jsonify(
            {
                "label": decoded["label"],
                "confidence": decoded["confidence"],
                "probabilities": {
                    CLASS_NAMES[0]: decoded["prob_no_cancer"],
                    CLASS_NAMES[1]: decoded["prob_cancer"],
                },
            }
        )
    except Exception as exc:
        return jsonify({"error": "Fallo durante la prediccion.", "detail": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

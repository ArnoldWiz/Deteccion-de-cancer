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
CLASS_NAMES = ["cancer", "no_cancer"]
# Para modelos con salida Dense(1, sigmoid), Keras suele usar etiqueta 1 para la
# segunda clase en orden alfabetico (normalmente "no_cancer").
SIGMOID_POSITIVE_CLASS = "no_cancer"

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
        p_pos = float(np.clip(raw_pred[0, 0], 0.0, 1.0))
        if SIGMOID_POSITIVE_CLASS not in CLASS_NAMES:
            raise ValueError("SIGMOID_POSITIVE_CLASS no existe en CLASS_NAMES.")

        negative_class = next(name for name in CLASS_NAMES if name != SIGMOID_POSITIVE_CLASS)
        probabilities = {
            SIGMOID_POSITIVE_CLASS: p_pos,
            negative_class: 1.0 - p_pos,
        }
    elif raw_pred.ndim == 2 and raw_pred.shape[1] >= 2:
        probs = raw_pred[0].astype(np.float64)
        probs_sum = probs.sum()
        probs_valid = np.all(probs >= 0.0) and np.all(probs <= 1.0) and abs(probs_sum - 1.0) < 1e-3

        if not probs_valid:
            probs = tf.nn.softmax(probs).numpy()

        if probs.shape[0] < len(CLASS_NAMES):
            raise ValueError("La salida del modelo tiene menos clases que CLASS_NAMES.")

        probabilities = {name: float(probs[idx]) for idx, name in enumerate(CLASS_NAMES)}
    else:
        raise ValueError("Forma de salida del modelo no soportada.")

    label = max(probabilities, key=probabilities.get)
    confidence = probabilities[label]

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": probabilities,
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
                "probabilities": decoded["probabilities"],
            }
        )
    except Exception as exc:
        return jsonify({"error": "Fallo durante la prediccion.", "detail": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

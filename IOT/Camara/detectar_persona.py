"""
Detector en vivo para una sola clase: persona.

Requiere:
    pip install ultralytics opencv-python

Uso:
    python detectar_persona.py --model runs/detect/train-5/weights/best.pt

Controles:
    - q: salir
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Detección en vivo con YOLO")
    parser.add_argument(
        "--model",
        default=str(Path(__file__).resolve().parents[2] / "runs" / "detect" / "train-5" / "weights" / "best.pt"),
        help="Ruta al modelo entrenado best.pt",
    )
    parser.add_argument("--camera", type=int, default=0, help="Índice de la cámara")
    parser.add_argument("--conf", type=float, default=0.5, help="Umbral de confianza")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamaño de inferencia")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"No existe el modelo: {model_path}")

    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara {args.camera}. Prueba con --camera 1 o revisa permisos.")

    print(f"Usando modelo: {model_path}")
    print("Presiona 'q' para salir")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("No se pudo leer un frame")
                break

            results = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
            result = results[0]

            detected_persona = False
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls_id in (0, 1):
                        detected_persona = True

                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    label = f"{model.names[cls_id]} {conf:.2f}"
                    color = (0, 255, 0) if cls_id == 0 else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

            status_text = "PERSONA DETECTADA" if detected_persona else "Sin persona"
            status_color = (0, 255, 0) if detected_persona else (0, 0, 255)
            cv2.putText(
                frame,
                status_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                status_color,
                3,
            )

            cv2.imshow("Detector persona", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
